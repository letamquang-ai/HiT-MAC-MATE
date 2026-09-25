from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from common_net.moe.base import MoE


class ImportanceLoss(nn.Module):
    """
    Encourages balanced importance among experts.
    Formula:
        L_importance = CV(importance)^2
    where importance = sum of gate probabilities per expert.
    """

    def __init__(self, eps=1e-9):
        super(ImportanceLoss, self).__init__()
        self.eps = eps

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        gate_logits: [batch_size, num_experts]
        """
        gate_logits = gate_logits.view(
            -1, gate_logits.shape[-1]
        )  # Flatten to 2D if necessary

        gate_probs = F.softmax(gate_logits, dim=-1)  # Softmax over experts
        importance = gate_probs.sum(dim=0)  # Sum over batch

        mean_importance = importance.mean()
        var_importance = ((importance - mean_importance) ** 2).mean()

        cv_squared = var_importance / (mean_importance**2 + self.eps)
        return cv_squared


class LoadLoss(nn.Module):
    """
    Encourages balanced load among experts.
    Formula:
        L_load = CV(load)^2
    where load = sum of gate selections per expert (hard selection).
    """

    def __init__(self, eps=1e-9):
        super(LoadLoss, self).__init__()
        self.eps = eps

    def forward(self, gate_logits):
        """
        gate_logits: [batch_size, num_experts]
        """
        gate_logits = gate_logits.view(
            -1, gate_logits.shape[-1]
        )  # Flatten to 2D if necessary

        gate_probs = F.softmax(gate_logits, dim=-1)
        # For load: sum over batch selections
        load = gate_probs.sum(dim=0)

        mean_load = load.mean()
        var_load = ((load - mean_load) ** 2).mean()

        cv_squared = var_load / (mean_load**2 + self.eps)
        return cv_squared


class CapacityLoss(nn.Module):
    """
    Penalizes experts that exceed their capacity.
    """

    def __init__(self, capacity: int = 10):
        super(CapacityLoss, self).__init__()
        self.capacity = capacity

    def forward(self, expert_assignments: torch.Tensor) -> torch.Tensor:
        """
        expert_assignments: [batch_size] tensor of expert indices (hard assignments)
        """
        num_experts = expert_assignments.max().item() + 1
        load_per_expert = torch.zeros(num_experts, device=expert_assignments.device)

        for expert in range(num_experts):
            load_per_expert[expert] = (expert_assignments == expert).sum()

        overload = torch.clamp(load_per_expert - self.capacity, min=0)
        capacity_loss = (overload**2).mean()

        return capacity_loss


class AllGateLoss(nn.Module):
    """
    Wrapper that combines multiple gate losses.
    """

    def __init__(
        self, importance_weight=1.0, load_weight=1.0, eps=1e-9, return_dict=False
    ):
        super(AllGateLoss, self).__init__()
        self.importance_loss = ImportanceLoss(eps=eps)
        self.load_loss = LoadLoss(eps=eps)

        self.importance_weight = importance_weight
        self.load_weight = load_weight
        self.return_dict = return_dict

    def forward(self, gate_logits):
        imp_loss = self.importance_loss(gate_logits)
        load_loss = self.load_loss(gate_logits)
        loss = self.importance_weight * imp_loss + self.load_weight * load_loss

        if self.return_dict:
            return loss, {"importance_loss": imp_loss, "load_loss": load_loss}
        else:
            return loss


class MoEGateLossManager(nn.Module):
    def __init__(self, module: nn.Module, criterion: Optional[Callable] = None):
        super().__init__()
        self.criterion = criterion if criterion is not None else AllGateLoss()

        self.losses: list[MoE] = []

        self._find_moe_layers(module)

    def _find_moe_layers(self, module: nn.Module):
        """Find all MoE layers in a module recursively."""
        for name, submodule in module.named_modules():
            if isinstance(submodule, MoE):
                self.register_moe(submodule)

    def register_moe(self, moe_layer: MoE):
        self.losses.append(moe_layer)

    def extend(self, other: "MoEGateLossManager"):
        self.losses.extend(other.losses)

    def forward(self):
        total_gate_loss = 0.0
        for moe in self.losses:
            gate_logits = moe._get_gate_logits()
            for logit in gate_logits:
                total_gate_loss += self.criterion(logit)

            moe._reset_gate_logits()

        return total_gate_loss

