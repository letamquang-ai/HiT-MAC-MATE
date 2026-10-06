"""Perception modules used by HiT-MAC-MATE.

This module preserves the public API and tensor shapes of the original
HiT-MAC perception.py while making noise buffers and recurrent states work
correctly across devices with modern PyTorch.
"""

import math
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class NoisyLinear(nn.Linear):
    """Linear layer with independently sampled Gaussian parameter noise."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.017,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_init = float(sigma_init)
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias:
            self.sigma_bias = nn.Parameter(torch.empty(out_features))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        else:
            self.register_parameter("sigma_bias", None)
            self.register_buffer("epsilon_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # nn.Linear calls this method during its own initialization, before
        # the NoisyLinear-specific parameters have been registered.
        if not hasattr(self, "sigma_weight"):
            super().reset_parameters()
            return

        # Match the initialization used by the original HiT-MAC implementation.
        bound = math.sqrt(3.0 / self.in_features) if self.in_features else 0.0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.sigma_weight, self.sigma_init)
        if self.sigma_bias is not None:
            nn.init.constant_(self.sigma_bias, self.sigma_init)
        with torch.no_grad():
            self.epsilon_weight.zero_()
            if self.epsilon_bias is not None:
                self.epsilon_bias.zero_()

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight + self.sigma_weight * self.epsilon_weight
        bias = self.bias
        if bias is not None and self.sigma_bias is not None and self.epsilon_bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, weight, bias)

    @torch.no_grad()
    def sample_noise(self, generator: Optional[torch.Generator] = None) -> None:
        """Sample noise in-place so buffers retain the module's device."""
        self.epsilon_weight.normal_(generator=generator)
        if self.epsilon_bias is not None:
            self.epsilon_bias.normal_(generator=generator)

    @torch.no_grad()
    def remove_noise(self) -> None:
        self.epsilon_weight.zero_()
        if self.epsilon_bias is not None:
            self.epsilon_bias.zero_()


def xavier_init(layer: nn.Linear) -> nn.Linear:
    """Apply Xavier-uniform weights and zero bias."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class BiRNN(nn.Module):
    """Bidirectional GRU/LSTM encoder compatible with the original API."""

    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = "lstm" in head_name.lower()
        rnn_cls = nn.LSTM if self.lstm else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        ).to(device)
        self.feature_dim = hidden_size * 2
        self.device = torch.device(device)

    def forward(self, x: Tensor, state=None):
        directions = 2
        h0 = x.new_zeros(self.num_layers * directions, x.size(0), self.hidden_size)
        if self.lstm:
            c0 = x.new_zeros(self.num_layers * directions, x.size(0), self.hidden_size)
            initial_state = (h0, c0) if state is None else state
            out, (hn, _cn) = self.rnn(x, initial_state)
        else:
            initial_state = h0 if state is None else state
            out, hn = self.rnn(x, initial_state)
        # The original implementation returns h_n only (not the LSTM cell).
        return out, hn


class RNN(nn.Module):
    """Unidirectional GRU/LSTM encoder compatible with the original API."""

    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = "lstm" in head_name.lower()
        rnn_cls = nn.LSTM if self.lstm else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.feature_dim = hidden_size
        self.device = torch.device(device)

    def forward(self, x: Tensor, state=None):
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        if self.lstm:
            c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
            initial_state = (h0, c0) if state is None else state
            out, (hn, _cn) = self.rnn(x, initial_state)
        else:
            initial_state = h0 if state is None else state
            out, hn = self.rnn(x, initial_state)
        return out, hn


class AttentionLayer(nn.Module):
    """Self-attention over targets.

    Input: x with shape [batch, num_targets, feature_dim].
    Returns z with shape [batch, num_targets, weight_dim] and the summed
    global feature with shape [batch, weight_dim], matching the original API.
    """

    def __init__(self, feature_dim, weight_dim, device):
        super().__init__()
        self.in_dim = feature_dim
        self.device = torch.device(device)
        self.Q = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.K = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.V = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.feature_dim = weight_dim

    def forward(self, x: Tensor):
        if x.ndim != 3:
            raise ValueError(
                f"AttentionLayer expects [batch, num_targets, feature_dim], got {tuple(x.shape)}"
            )
        q = torch.tanh(self.Q(x))
        k = torch.tanh(self.K(x))
        v = torch.tanh(self.V(x))
        weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=2)
        z = torch.bmm(weights, v)
        global_feature = z.sum(dim=1)
        return z, global_feature
