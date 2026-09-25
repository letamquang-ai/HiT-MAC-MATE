import math

import torch

from .base import BasePosEmbedding


class SinusoidalPositionalEmbedding(BasePosEmbedding):
    """
    Standard sinusoidal positional embedding.

    Supports input shapes:
        - (B, N, D)
        - (B, H, N, D)

    Assumes:
        - sequence dimension is -2
        - embedding dimension is -1

    Returns:
        Positional embedding broadcastable to x:
        - (1, N, D) for input (B, N, D)
        - (1, 1, N, D) for input (B, H, N, D)
    """

    def __init__(self, base: float = 10000.0):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (3, 4):
            raise ValueError(
                f"Expected x with shape (B, N, D) or (B, H, N, D), got {tuple(x.shape)}"
            )

        *leading, N, D = x.shape
        device = x.device
        dtype = x.dtype

        position = torch.arange(N, device=device, dtype=torch.float32)  # (N,)
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=torch.float32)
            * (-math.log(self.base) / D)
        )  # (ceil(D/2),)

        pe = torch.zeros(N, D, device=device, dtype=torch.float32)  # (N, D)
        pe[:, 0::2] = torch.sin(position[:, None] * div_term[None, :])
        pe[:, 1::2] = torch.cos(
            position[:, None] * div_term[None, : pe[:, 1::2].shape[1]]
        )

        shape = [1] * len(leading) + [N, D]
        return x + pe.view(*shape).to(dtype)
