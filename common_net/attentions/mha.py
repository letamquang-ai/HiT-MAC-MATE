import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from common_net.pos_embedding.base import BasePosEmbedding, IdentityPosEmbedding

from ..common import Gated, ZeroCenteredRMSNorm
from .base_attn import AttentionBase, ScaledDotProductAttention


@dataclass(frozen=True)
class MHAConfig:
    num_heads: int = 1
    attn_cls: type[AttentionBase] = ScaledDotProductAttention
    attn_kwargs: Optional[dict] = None
    positional_embedding_cls: type[BasePosEmbedding] = IdentityPosEmbedding
    positional_embedding_kwargs: Optional[dict] = None
    gated: bool = False
    gate_act_fn: str | Callable[[torch.Tensor], torch.Tensor] = "sigmoid"
    gate_operator_fn: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "*"
    num_k_heads: Optional[int] = None
    num_v_heads: Optional[int] = None
    kdim: Optional[int] = None
    vdim: Optional[int] = None
    qk_norm_cls: type[nn.Module] = ZeroCenteredRMSNorm
    qk_norm_kwargs: Optional[dict] = None
    bias: bool = False
    device: Optional[torch.device | str] = None
    dtype: Optional[torch.dtype] = None


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module with support for group query attention and gated attention."""

    def __init__(
        self,
        embed_dim: int,
        config: MHAConfig = MHAConfig(),
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = config.num_heads
        self.num_k_heads = (
            config.num_k_heads if config.num_k_heads is not None else config.num_heads
        )
        self.num_v_heads = (
            config.num_v_heads if config.num_v_heads is not None else config.num_heads
        )
        self.kdim = config.kdim if config.kdim is not None else embed_dim
        self.vdim = config.vdim if config.vdim is not None else embed_dim

        factory_kwargs = {"device": config.device, "dtype": config.dtype}

        self.head_dim = embed_dim // self.num_heads

        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by the number of heads ({self.num_heads})."
            )

        if self.num_heads == self.num_k_heads and self.num_heads == self.num_v_heads:
            pass  # normal attention
        elif self.num_heads > max(
            self.num_k_heads, self.num_v_heads
        ):  # group query Attention
            if self.num_heads % self.num_k_heads != 0:
                raise ValueError(
                    f"Number of attention heads must be multiple of number of key heads, get H={self.num_heads}, Hk={self.num_k_heads}"
                )
            if self.num_heads % self.num_v_heads != 0:
                raise ValueError(
                    f"Number of attention heads must be multiple of number of value heads, get H={self.num_heads}, Hk={self.num_v_heads}"
                )
        else:  # to be update
            raise NotImplementedError(
                "Only support Group Query Attention for now (Hk == Hv < Hq), get Hq={}, Hk={}, Hv={}".format(
                    self.num_heads, self.num_k_heads, self.num_v_heads
                )
            )

        self.positional_embedding = config.positional_embedding_cls(
            **(config.positional_embedding_kwargs or {})
        )

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=config.bias, **factory_kwargs
        )
        self.k_proj = nn.Linear(
            self.kdim,
            self.num_k_heads * self.head_dim,
            bias=config.bias,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            self.vdim,
            self.num_v_heads * self.head_dim,
            bias=config.bias,
            **factory_kwargs,
        )

        if config.gated:
            self.gate = Gated(
                embed_dim,
                embed_dim,
                bias=config.bias,
                gate_act_fn=config.gate_act_fn,
                gate_operator_fn=config.gate_operator_fn,
            )
        else:
            self.gate = None

        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=config.bias, **factory_kwargs
        )

        attn_kwargs = config.attn_kwargs if config.attn_kwargs is not None else {}
        self.attn = config.attn_cls(**attn_kwargs)

        qk_norm_kwargs = (
            config.qk_norm_kwargs if config.qk_norm_kwargs is not None else {}
        )

        self.q_norm = config.qk_norm_cls(self.head_dim, **qk_norm_kwargs)
        self.k_norm = config.qk_norm_cls(self.head_dim, **qk_norm_kwargs)
        self._reset_parameters()

    def _reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(_init_weights)

    def expand_3d(self, x: torch.Tensor, name: str) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 1:
            warnings.warn(
                f"{name} is a 1D tensor with shape (E,), automatically unsqueeze to (1, 1, E)"
            )

            return x.unsqueeze(0).unsqueeze(0)

        elif x.dim() == 2:
            warnings.warn(
                f"{name} is a 2D tensor with shape (N, E), automatically unsqueeze to (1, N, E)"
            )

            return x.unsqueeze(0)

        elif x.dim() == 3:
            return x

        else:
            raise ValueError(
                f"{name} must be a 3D tensor with shape (B, N, E), 2D tensor with shape (N, E) or 1D tensor with shape (E,), but got shape {original_shape}"
            )

    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        causal=False,
    ):
        # Q: (B, N, E)
        # K: (B, Nkv, E)
        # V: (B, Nkv, E)

        Q = self.expand_3d(Q, "Q")
        if K is None:
            K = Q
        if V is None:
            V = K
        K = self.expand_3d(K, "K")
        V = self.expand_3d(V, "V")

        if Q.size(0) != K.size(0) or Q.size(0) != V.size(0):
            raise ValueError(
                f"Batch size of Q, K and V must be the same, but got {Q.size(0)}, {K.size(0)}, {V.size(0)}"
            )

        Nkv = K.size(1)
        if Nkv != (Nv := V.size(1)):
            raise ValueError(
                f"Number of key tokens (Nk={Nkv}) must be equal to number of value tokens (Nv={Nv})"
            )

        B, N, _ = Q.size()

        H = self.num_heads
        Hk = self.num_k_heads
        Hv = self.num_v_heads
        D = self.head_dim

        # E = H*D

        # Project to multi-head
        Q_proj: torch.Tensor = self.q_proj(Q)  # (B, N, H * D)
        K_proj: torch.Tensor = self.k_proj(K)  # (B, Nkv, Hk * D)
        V_proj: torch.Tensor = self.v_proj(V)  # (B, Nkv, Hv * D)

        Q_proj = Q_proj.view(B, N, H, D).transpose(1, 2)  # (B, H , N, D)
        K_proj = K_proj.view(B, Nkv, Hk, D).transpose(1, 2)  # (B, Hk, Nkv, D)
        V_proj = V_proj.view(B, Nkv, Hv, D).transpose(1, 2)  # (B, Hv, Nkv, D)

        Q_proj = self.q_norm(Q_proj)  # (B, H, N, D)
        K_proj = self.k_norm(K_proj)  # (B, Hk, Nkv, D)

        Q_proj = self.positional_embedding(Q_proj)  # (B, H, N, D)
        K_proj = self.positional_embedding(K_proj)  # (B, Hk, Nkv, D)
        # V_proj does not need positional embedding

        # Attention
        attn_out, attn_weights = self.attn(
            Q_proj, K_proj, V_proj, causal=causal
        )  # (B, H, N, D)

        attn_out: torch.Tensor = (
            attn_out.transpose(1, 2).contiguous().view(B, N, H * D)
        )  # (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D) or (B, N, E)

        if self.gate is not None:
            attn_out = self.gate(
                Q, attn_out
            )  # attn_out = attn_out o act(Q @ W)  -> (B, N, E)

        out = self.out_proj(attn_out)  # (B, N, E)
        return out, attn_weights


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        q_latent_dim: int | None = None,
        kv_latent_dim: int | None = None,
        config: MHAConfig = MHAConfig(),
    ) -> None:
        super().__init__()

        raise NotImplementedError(
            "This class is still under development, use MultiHeadAttention instead for now"
        )

        if q_latent_dim is None:
            self.q_latent_dim = embed_dim // 8
        if kv_latent_dim is None:
            self.kv_latent_dim = embed_dim // 16

        self.config = config

        head_dim = embed_dim // config.num_heads

        self.Wq_d = nn.Linear(embed_dim, self.q_latent_dim, bias=config.bias)
        self.W_qk = nn.Linear(
            self.q_latent_dim, config.num_heads * self.kv_latent_dim, bias=config.bias
        )

        self.Wkv_d = nn.Linear(embed_dim, self.kv_latent_dim, bias=config.bias)
        self.Wv_u = nn.Linear(
            self.kv_latent_dim, config.num_heads * head_dim, bias=config.bias
        )

        self.Wo = nn.Linear(config.num_heads * head_dim, embed_dim, bias=config.bias)

    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        causal=False,
    ):
        # Q: (B, N, E)
        # K: (B, Nkv, E)
        # V: (B, Nkv, E)

        B, N, _ = Q.size()
        Nkv = K.size(1)

        C_q = self.Wq_d(Q)  # (B, N, q_latent_dim)
        C_kv = self.Wkv_d(K)  # (B, Nkv, kv_latent_dim)

        C_qW_qk = self.W_qk(C_q)  # (B, N, H*kv_latent_dim)
        C_qW_qk = C_qW_qk.view(B, N, self.config.num_heads, -1).transpose(
            1, 2
        )  # (B, H, N, kv_latent_dim)

        scores = torch.matmul(
            C_qW_qk.transpose(1, 2), C_kv.transpose(-2, -1)[:, None, ...]
        ) / (math.sqrt(self.kv_latent_dim))  # (B, H, N, Nkv)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, N, Nkv)

        V_out = (
            self.Wv_u(C_kv).view(B, Nkv, self.config.num_heads, -1).transpose(1, 2)
        )  # (B, H, Nkv, head_dim)

        output = (
            torch.matmul(attn_weights, V_out.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
            .view(B, N, -1)
        )  # (B, H, N, head_dim) -> (B, N, H*head_dim)

        return output
