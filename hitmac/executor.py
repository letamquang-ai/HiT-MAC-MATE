"""Executor network for HiT-MAC.

This module defines the neural components used by the low-level executor.
Implementation details are intentionally left as TODOs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class ExecutorOutput(NamedTuple):
    """Outputs produced by the executor policy."""

    action_mean: Tensor
    action_log_std: Tensor
    value: Tensor
    log_prob: Tensor | None
    entropy: Tensor | None


class ScaledDotAttention(nn.Module):
    """Scaled dot-product attention over a set of target features."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # TODO: define query, key, and value projections.
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Encode target features, optionally masking invalid targets."""
        raise NotImplementedError


class Executor(nn.Module):
    """Goal-conditioned policy/value network for camera control."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        action_dim: int = 2,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        # TODO: define target encoder, attention, policy, and value heads.
        raise NotImplementedError

    def encode(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Encode the observation and its candidate target features."""
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return action mean, action log standard deviation, and value."""
        raise NotImplementedError

    def get_distribution(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> torch.distributions.Distribution:
        """Construct the policy action distribution."""
        raise NotImplementedError

    def act(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        low: Tensor | None = None,
        high: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample or select an action and return action, log-probability, entropy, value."""
        raise NotImplementedError

    def evaluate_actions(
        self,
        x: Tensor,
        actions: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate stored actions for policy-gradient updates."""
        raise NotImplementedError


__all__ = ["Executor", "ExecutorOutput", "ScaledDotAttention"]
