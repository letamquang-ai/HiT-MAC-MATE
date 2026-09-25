from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_net.mlp.gated import GatedMLP


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 4
    use_shared_expert: bool = False  # Whether to use shared experts
    gate_act_fn: str | Callable[[torch.Tensor], torch.Tensor] = (
        "sigmoid"  # Activation function for the experts
    )


class MoE(nn.Module):
    config: MoEConfig

    def __init__(
        self, embed_dim: int = 128, d_ff: int = 512, config: Optional[MoEConfig] = None
    ):
        super().__init__()

        self.config = config if config is None else config
        assert isinstance(self.config, MoEConfig), (
            "config must be an instance of MoEConfig"
        )

        if self.config.use_shared_expert:
            self.shared_expert = GatedMLP(
                embed_dim, d_ff, gate_act_fn=self.config.gate_act_fn
            )
            self.shared_expert_gate = nn.Linear(embed_dim, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

        self.experts = nn.ModuleList(
            [
                GatedMLP(embed_dim, d_ff, gate_act_fn=self.config.gate_act_fn)
                for _ in range(self.config.num_experts - self.config.use_shared_expert)
            ]
        )

        # Gating network (decides which experts to use per token)
        self.gate = nn.Linear(embed_dim, self.config.num_experts, bias=False)

        self._gate_logits: list[torch.Tensor] = []
        self._last_gate_logits: torch.Tensor = None

    def _get_gate_logits(self) -> torch.Tensor:
        return self._gate_logits

    def _store_gate_logits(self, gate_logits: torch.Tensor):
        if self.training:
            self._gate_logits.append(gate_logits)
        self._last_gate_logits = gate_logits

    def _reset_gate_logits(self):
        self._gate_logits = []

    @property
    def gate_logit(self):
        """Returns the most recent gate logits.

        Returns:
            torch.Tensor: The most recent gate logits.
        """
        return self._last_gate_logits

    def forward_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.use_shared_expert:
            shared_expert_output: torch.Tensor = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            )
            return shared_expert_output
        return 0.0

    def forward_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_probs = F.softmax(self.gate_logit, dim=1, dtype=hidden_states.dtype)
        gate_probs = gate_probs.to(hidden_states.dtype)

        expert_outputs = torch.stack(
            [expert(hidden_states) for expert in self.experts], dim=-1
        )  # (*, D, n_experts)

        expert_outputs = torch.matmul(
            expert_outputs,  # (*, D, n_experts)
            gate_probs.unsqueeze(-1),  # (*, n_experts, 1)
        ).squeeze(-1)  # (*, D)

        return expert_outputs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """

        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]

        hidden_states = hidden_states.view(-1, hidden_dim)

        self._store_gate_logits(self.gate(hidden_states))

        final_hidden_states = self.forward_experts(
            hidden_states
        ) + self.forward_shared_expert(hidden_states)

        return final_hidden_states.reshape(original_shape)


