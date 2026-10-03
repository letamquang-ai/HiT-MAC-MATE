"""HiT-MAC low-level executor (A3C) adapted to MATE.

Implements the executor side described in HiT-MAC:
- goal-conditioned target filtering;
- self-attention encoder;
- one-layer actor and critic;
- pseudo-goal generation every K steps;
- A3C/GAE training with shared parameters.

The original paper uses discrete left/right/stay actions. MATE exposes
continuous camera controls, so this implementation uses a tanh-Gaussian
policy and can optionally train rotation only (the paper-faithful default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class ExecutorConfig:
    hidden_dim: int = 128
    gamma: float = 0.9
    tau: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 5e-4
    grad_clip: float = 50.0
    rollout_steps: int = 20
    goal_period: int = 10
    reward_beta: float = 0.01
    rotation_only: bool = True


class ScaledDotAttention(nn.Module):
    """The scaled dot-product attention from Eq. (1) of HiT-MAC."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(input_dim, hidden_dim)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, M, D], mask: [B, M]
        q = torch.tanh(self.q(x))
        k = torch.tanh(self.k(x))
        v = torch.tanh(self.v(x))

        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(k.shape[-1])

        key_mask = mask[:, None, :].bool()
        scores = scores.masked_fill(
            ~key_mask,
            torch.finfo(scores.dtype).min,
        )

        # Avoid NaNs when a pseudo-goal is empty.
        empty = ~mask.any(dim=1)

        # IMPORTANT: do not use scores[empty] = 0.0
        scores = torch.where(
            empty[:, None, None],
            torch.zeros_like(scores),
            scores,
        )

        weights = F.softmax(scores, dim=-1)

        weights = weights * mask[:, None, :].to(weights.dtype)

        attended = torch.bmm(weights, v)  # [B, M, H]

        # Average over valid query targets.
        denom = (
            mask.sum(dim=1, keepdim=True)
            .to(attended.dtype)
            .clamp_min(1.0)
        )

        pooled = attended.sum(dim=1) / denom  # [B, H]

        # IMPORTANT: do not use pooled[empty] = 0.0
        pooled = torch.where(
            empty[:, None],
            torch.zeros_like(pooled),
            pooled,
        )

        return pooled


class ExecutorNet(nn.Module):
    """Shared executor policy/value network."""

    def __init__(self, target_feature_dim: int = 5, hidden_dim: int = 128, action_dim: int = 1):
        super().__init__()
        self.encoder = ScaledDotAttention(target_feature_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        nn.init.normal_(self.actor.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.normal_(self.critic.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        h = self.encoder(x, mask)
        mean = torch.tanh(self.actor(h))
        value = self.critic(h).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def act(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        deterministic: bool = False,
    ):
        mean, std, value = self(x, mask)
        dist = Normal(mean, std)

        if deterministic:
            z = mean
        else:
            z = dist.rsample()

        squashed = torch.tanh(z)
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        action = squashed * scale + bias

        # Log-probability of tanh-squashed Gaussian.
        log_prob = dist.log_prob(z) - torch.log(
            1.0 - squashed.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value


def build_target_features(observation: np.ndarray, goal: np.ndarray, num_targets: int) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float64)
    from mate import constants as consts
    slices = consts.camera_observation_slices_of(int(round(obs[0])), int(round(obs[1])), int(round(obs[2])))
    self_state = obs[slices["self_state"]]
    target_block = obs[slices["opponent_states_with_mask"]].reshape(num_targets, consts.TARGET_STATE_DIM_PUBLIC + 1)
    camera_xy = self_state[:2]
    max_range = max(float(self_state[6]), 1e-6)
    features = np.zeros((num_targets, 5), dtype=np.float32)
    sensed = target_block[:, consts.TARGET_STATE_DIM_PUBLIC] > 0.5
    deltas = target_block[:, :2] - camera_xy
    distances = np.linalg.norm(deltas, axis=1)
    selected = sensed & (np.asarray(goal) > 0.5) & (distances < max_range)
    if not selected.any():
        return features, selected.astype(np.bool_)
    orientation = float(np.degrees(np.arctan2(self_state[4], self_state[3])))
    target_angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
    relative_angles = ((target_angles - orientation + 180.0) % 360.0) - 180.0
    idx = np.flatnonzero(selected)
    features[idx, 0] = idx / max(num_targets - 1, 1)
    features[idx, 1] = distances[idx] / max_range
    features[idx, 2] = relative_angles[idx] / 180.0
    features[idx, 3] = target_block[idx, 2] / max_range
    features[idx, 4] = target_block[idx, 3]
    return features, selected.astype(np.bool_)


def generate_pseudo_goals(observations: np.ndarray, num_targets: int) -> np.ndarray:
    observations = np.asarray(observations)
    from mate import constants as consts
    goals = np.zeros((len(observations), num_targets), dtype=np.float32)
    for i, obs in enumerate(observations):
        slices = consts.camera_observation_slices_of(int(round(obs[0])), int(round(obs[1])), int(round(obs[2])))
        self_state = obs[slices["self_state"]]
        target_block = obs[slices["opponent_states_with_mask"]].reshape(num_targets, consts.TARGET_STATE_DIM_PUBLIC + 1)
        camera_xy = self_state[:2]
        max_range = float(self_state[6])
        sensed = target_block[:, consts.TARGET_STATE_DIM_PUBLIC] > 0.5
        distances = np.linalg.norm(target_block[:, :2] - camera_xy, axis=1)
        valid = sensed & (distances < max_range)
        goals[i, valid] = 1.0
        if not valid.any() and sensed.any():
            goals[i, int(np.argmin(np.where(sensed, distances, np.inf)))] = 1.0
    return goals


def goal_conditioned_reward(observation: np.ndarray, next_observation: np.ndarray, goal: np.ndarray, action: np.ndarray, beta: float = 0.01) -> float:
    from mate import constants as consts
    obs = np.asarray(next_observation, dtype=np.float64)
    slices = consts.camera_observation_slices_of(int(round(obs[0])), int(round(obs[1])), int(round(obs[2])))
    self_state = obs[slices["self_state"]]
    num_targets = int(round(obs[1]))
    target_block = obs[slices["opponent_states_with_mask"]].reshape(num_targets, consts.TARGET_STATE_DIM_PUBLIC + 1)
    indices = np.flatnonzero(np.asarray(goal) > 0.5)
    if indices.size == 0:
        tracking_reward = 0.0
    else:
        targets = target_block[indices]
        sensed = targets[:, consts.TARGET_STATE_DIM_PUBLIC] > 0.5
        delta = targets[:, :2] - self_state[:2]
        distance = np.linalg.norm(delta, axis=1)
        orientation = float(np.degrees(np.arctan2(self_state[4], self_state[3])))
        target_angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        relative_angle = np.abs(((target_angle - orientation + 180.0) % 360.0) - 180.0)
        alpha_max = max(float(self_state[5]), 1e-6)
        covered = sensed & (distance < max(float(self_state[6]), 1e-6)) & (relative_angle < alpha_max)
        tracking_reward = float(np.where(covered, 1.0 - relative_angle / alpha_max, -1.0).mean())
    rotation_step = max(float(abs(self_state[7])), 1e-6)
    rotation_cost = abs(float(np.asarray(action)[0])) / rotation_step
    return tracking_reward - beta * rotation_cost

