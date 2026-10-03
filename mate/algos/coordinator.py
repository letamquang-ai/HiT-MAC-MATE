"""HiT-MAC high-level coordinator actor and AMC critic."""
from __future__ import annotations
import math
from torch import nn
import torch
from torch.distributions import Bernoulli

class ScaledDotAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(input_dim, hidden_dim)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        q, k, v = (torch.tanh(layer(x)) for layer in (self.q, self.k, self.v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        if mask is None:
            weights = torch.softmax(scores, dim=-1)
        else:
            key_mask = mask.unsqueeze(-2).bool()
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1) * key_mask.to(scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.matmul(weights, v)

class AMCCritic(nn.Module):
    """Attention-based marginal contribution critic, using fixed feature order."""
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = ScaledDotAttention(feature_dim, hidden_dim)
        self.phi = nn.Sequential(nn.Linear(feature_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor, valid: torch.Tensor | None = None):
        batch, count, _ = features.shape
        if count == 0:
            return features.new_zeros(batch)

        if valid is None:
            valid = torch.ones(batch, count, dtype=torch.bool, device=features.device)
        else:
            valid = valid.bool()

        # Compute all prefix attentions in one causal pass instead of
        # recomputing attention separately for every prefix.
        key_mask = valid[:, None, :].expand(batch, count, count)
        causal = torch.tril(
            torch.ones(count, count, dtype=torch.bool, device=features.device),
            diagonal=-1,
        )
        attention_mask = key_mask & causal.unsqueeze(0)

        q, k, v = (
            torch.tanh(layer(features))
            for layer in (self.attention.q, self.attention.k, self.attention.v)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        # Position zero has an empty prefix and is not used as a context.
        scores[:, 0, :] = 0.0
        weights = torch.softmax(scores, dim=-1)
        weights = weights * attention_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.matmul(weights, v)

        # Context for contribution i is the mean attended representation
        # over valid positions strictly before i.
        prefix_sum = torch.cumsum(attended, dim=1)
        prefix_count = torch.cumsum(valid.to(features.dtype), dim=1)
        context = torch.zeros(
            batch, count, self.attention.q.out_features,
            dtype=features.dtype, device=features.device,
        )
        if count > 1:
            context[:, 1:] = prefix_sum[:, :-1] / prefix_count[:, :-1].unsqueeze(-1).clamp_min(1.0)

        contribution = self.phi(torch.cat((context, features), dim=-1)).squeeze(-1)
        return (contribution * valid.to(features.dtype)).sum(dim=1)

class CoordinatorNet(nn.Module):
    """Produces a Bernoulli camera-target goal map and team-value estimate.

    Input: observation [B, cameras, targets, feature_dim]; optional valid mask [B,cameras,targets].
    Output of act: goals [B,cameras,targets], log_prob [B], entropy [B], value [B].
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, attention_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, attention_dim), nn.ReLU())
        self.attention = ScaledDotAttention(attention_dim, attention_dim)
        self.actor = nn.Linear(attention_dim, 1)
        self.critic = AMCCritic(attention_dim, hidden_dim)
        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, observation: torch.Tensor, mask: torch.Tensor | None = None):
        if observation.ndim != 4:
            raise ValueError("observation must be [batch, cameras, targets, features]")
        b, n, m, _ = observation.shape
        encoded = self.encoder(observation)
        flat = encoded.reshape(b*n, m, -1)
        flat_mask = None if mask is None else mask.reshape(b*n, m).bool()
        attended = self.attention(flat, flat_mask).reshape(b, n, m, -1)
        probs = torch.sigmoid(self.actor(attended).squeeze(-1))
        critic_features = attended.reshape(b, n*m, -1)
        critic_mask = None if mask is None else mask.reshape(b, n*m).bool()
        value = self.critic(critic_features, critic_mask)
        return probs, value

    def act(self, observation: torch.Tensor, mask: torch.Tensor | None = None, sample: bool = True):
        probs, value = self(observation, mask)
        dist = Bernoulli(probs=probs.clamp(1e-6, 1-1e-6))
        goals = dist.sample() if sample else (probs >= 0.5).to(probs.dtype)
        valid = 1.0 if mask is None else mask.to(probs.dtype)
        goals = goals * valid
        log_prob = (dist.log_prob(goals) * valid).sum(dim=(-2,-1))
        entropy = (dist.entropy() * valid).sum(dim=(-2,-1))
        return goals, log_prob, entropy, value

__all__ = ["ScaledDotAttention", "AMCCritic", "CoordinatorNet"]