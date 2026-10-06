import torch
from .base import BasePosEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate pairs in the last dimension:
    (x1, x2) -> (-x2, x1)
    """
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {d}")

    x = x.view(*x.shape[:-1], d // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply rotary positional embeddings.

    Supported shapes:
        (B, N, D)
        (B, H, N, D)

    Assumes:
        - sequence dimension is -2
        - feature dimension is -1
    """
    if x.ndim not in (3, 4):
        raise ValueError(
            f"Expected x to have shape (B, N, D) or (B, H, N, D), got {tuple(x.shape)}"
        )

    *leading, N, D = x.shape

    if D % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got D={D}")

    device = x.device
    dtype = x.dtype
    half_d = D // 2

    # (D/2,)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_d, device=device, dtype=torch.float32) / half_d)
    )

    # (N,)
    pos = torch.arange(N, device=device, dtype=torch.float32)

    # (N, D/2)
    freqs = torch.outer(pos, inv_freq)

    # (N, D)
    freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)

    # reshape to broadcast over x
    # for (B, N, D)   -> (1, N, D)
    # for (B, H, N, D)-> (1, 1, N, D)
    shape = [1] * len(leading) + [N, D]
    cos = freqs.cos().to(dtype).view(*shape)  # (1, N, D) or (1, 1, N, D)
    sin = freqs.sin().to(dtype).view(*shape)  # (1, N, D) or (1, 1, N, D)

    return x * cos + rotate_half(x) * sin  # (B, N, D) or (B, H, N, D)


class RotaryEmbedding(BasePosEmbedding):
    """
    Module that applies rotary positional embedding to input tensor.

    Args:
        base: RoPE base
    """

    def __init__(self, base: float = 10000.0):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_rope(x, base=self.base)
