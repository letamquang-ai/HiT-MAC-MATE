"""HiT-MAC low-level executor.

Implements the low-level policy pi^L(a | o, g) from HiT-MAC.

Input:
    x: [B, M, D] target features.
    mask: [B, M] boolean/0-1 goal-conditioned mask.

act() returns:
    action: [B, action_dim]
    logp: [B, 1]
    entropy: [B, 1]
    value: [B, 1]
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class ScaledDotAttention(nn.Module):
    """Scaled dot-product self-attention from HiT-MAC Eq. (1)-(2)."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.q = nn.Linear(self.input_dim, self.hidden_dim)
        self.k = nn.Linear(self.input_dim, self.hidden_dim)
        self.v = nn.Linear(self.input_dim, self.hidden_dim)

        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, M, D], got {tuple(x.shape)}")

        q = torch.tanh(self.q(x))
        k = torch.tanh(self.k(x))
        v = torch.tanh(self.v(x))

        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores / math.sqrt(self.hidden_dim)

        if mask is not None:
            key_mask = mask.to(dtype=torch.bool, device=x.device).unsqueeze(1)
            scores = scores.masked_fill(
                ~key_mask, torch.finfo(scores.dtype).min
            )

        weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            key_mask_f = mask.to(
                dtype=weights.dtype, device=x.device
            ).unsqueeze(1)
            weights = weights * key_mask_f
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom

        attended = torch.bmm(weights, v)
        return attended, weights


class TargetEncoder(nn.Module):
    """Per-target MLP followed by HiT-MAC self-attention."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.attention = ScaledDotAttention(
            self.hidden_dim, self.hidden_dim
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        h = self.mlp(x)
        attended, weights = self.attention(h, mask)
        h = self.norm(h + attended)

        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.to(dtype=h.dtype, device=h.device).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = (h * m).sum(dim=1) / denom

        return h, pooled


class Executor(nn.Module):
    """Goal-conditioned HiT-MAC low-level executor.

    The default input has five MATE target features:
      target_id, normalized distance, normalized relative angle,
      normalized target sight range, loaded bit.

    The action is continuous so the same network can control MATE's
    rotation and zoom dimensions. Rotation-only training can disable zoom
    in the rollout code.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        action_dim: int = 2,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)

        self.encoder = TargetEncoder(self.input_dim, self.hidden_dim)

        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(log_std_init))
        )

        self._init_output(self.actor[-1], gain=0.01)
        self._init_output(self.critic[-1], gain=1.0)

    @staticmethod
    def _init_output(layer: nn.Linear, gain: float) -> None:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    @staticmethod
    def _as_batch(
        x: Tensor, mask: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], bool]:
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, M, D], got {tuple(x.shape)}"
            )

        if mask is not None and (
            mask.ndim != 2 or mask.shape != x.shape[:2]
        ):
            raise ValueError(
                f"mask must have shape [B, M], got {tuple(mask.shape)} "
                f"for x={tuple(x.shape)}"
            )

        return x, mask, squeeze

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x, mask, _ = self._as_batch(x, mask)
        _, pooled = self.encoder(x, mask)

        mean = self.actor(pooled)
        value = self.critic(pooled)
        return mean, value

    def _distribution(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Normal, Tensor]:
        mean, value = self.forward(x, mask)
        std = self.log_std.exp().clamp_min(1e-4)
        return Normal(mean, std), value

    @staticmethod
    def _bound_action(
        normalized: Tensor, low: Tensor, high: Tensor
    ) -> Tensor:
        return low + 0.5 * (normalized + 1.0) * (high - low)

    @staticmethod
    def _normalize_action(
        action: Tensor, low: Tensor, high: Tensor
    ) -> Tensor:
        scale = (high - low).clamp_min(1e-8)
        return 2.0 * (action - low) / scale - 1.0

    @staticmethod
    def _prepare_bounds(
        low: Tensor,
        high: Tensor,
        batch: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        low = torch.as_tensor(low, device=device, dtype=dtype)
        high = torch.as_tensor(high, device=device, dtype=dtype)

        if low.ndim == 1:
            low = low.unsqueeze(0)
        if high.ndim == 1:
            high = high.unsqueeze(0)

        if low.shape[-1] != action_dim or high.shape[-1] != action_dim:
            raise ValueError(
                f"action bounds must have last dimension {action_dim}; "
                f"got low={tuple(low.shape)}, high={tuple(high.shape)}"
            )

        if low.shape[0] == 1 and batch != 1:
            low = low.expand(batch, -1)
        if high.shape[0] == 1 and batch != 1:
            high = high.expand(batch, -1)

        if low.shape != (batch, action_dim) or high.shape != (batch, action_dim):
            raise ValueError(
                f"bounds must broadcast to [{batch}, {action_dim}], "
                f"got low={tuple(low.shape)}, high={tuple(high.shape)}"
            )

        return low, high

    def act(
        self,
        x: Tensor,
        mask: Optional[Tensor],
        low: Tensor,
        high: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample a bounded action and return log-probability, entropy, value."""

        x, mask, squeeze = self._as_batch(x, mask)
        batch = x.shape[0]

        dist, value = self._distribution(x, mask)
        raw = dist.mean if deterministic else dist.rsample()

        normalized = torch.tanh(raw)

        low, high = self._prepare_bounds(
            low, high, batch, self.action_dim, x.device, x.dtype
        )
        action = self._bound_action(normalized, low, high)

        log_det_tanh = torch.log(
            1.0 - normalized.pow(2) + 1e-6
        )
        log_scale = torch.log(
            (high - low).clamp_min(1e-8) * 0.5
        )

        log_prob = (
            dist.log_prob(raw) - log_det_tanh - log_scale
        ).sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        if squeeze:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        x: Tensor,
        mask: Optional[Tensor],
        action: Tensor,
        low: Tensor,
        high: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate supplied environment actions for PPO/A2C updates."""

        x, mask, squeeze = self._as_batch(x, mask)

        dist, value = self._distribution(x, mask)
        low, high = self._prepare_bounds(
            low, high, x.shape[0], self.action_dim, x.device, x.dtype
        )

        action = torch.as_tensor(
            action, device=x.device, dtype=x.dtype
        )
        if action.ndim == 1:
            action = action.unsqueeze(0)

        normalized = self._normalize_action(action, low, high)
        normalized = normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(normalized)

        log_det_tanh = torch.log(
            1.0 - normalized.pow(2) + 1e-6
        )
        log_scale = torch.log(
            (high - low).clamp_min(1e-8) * 0.5
        )

        log_prob = (
            dist.log_prob(raw) - log_det_tanh - log_scale
        ).sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        if squeeze:
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return log_prob, entropy, value


# Compatibility aliases for training scripts.
HiTMACExecutor = Executor
ExecutorNet = Executor


__all__ = [
    "ScaledDotAttention",
    "TargetEncoder",
    "Executor",
    "HiTMACExecutor",
    "ExecutorNet",
]
