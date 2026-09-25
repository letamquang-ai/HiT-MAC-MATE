from dataclasses import dataclass

import torch
import torch.nn.functional as F

from common_net.moe.base import MoE, MoEConfig



@dataclass(frozen=True)
class TopKMoEConfig(MoEConfig):
    top_k: int = 1
    norm_topk_prob: bool = False  # Whether to normalize the top-k probabilities


class TopKMoE(MoE):
    config: TopKMoEConfig
    def __init__(self, embed_dim = 128, d_ff = 512, config = None):
        config = config if config is not None else TopKMoEConfig()
        super().__init__(embed_dim, d_ff, config)

    def forward_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (*, D)
        # gate_probs: (*, n_experts)

        top_k = self.config.top_k

        if top_k > self.config.num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than num_experts ({self.config.num_experts})"
            )

        gate_probs = F.softmax(self.gate_logit, dim=1, dtype=hidden_states.dtype)
        gate_probs = gate_probs.to(hidden_states.dtype)
        gate_probs, selected_experts = torch.topk(gate_probs, top_k, dim=-1)

        if self.config.norm_topk_prob:
            gate_probs /= gate_probs.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.config.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(
                -1, hidden_states.shape[-1]
            )
            current_hidden_states = (
                expert_layer(current_state) * gate_probs[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        return final_hidden_states
