import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split last dim into pairs and apply:
    (x1, x2) -> (-x2, x1)
    """
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2)


def rotary_emb(
    x: torch.Tensor,
    base: float = 10000.0,
    seq_dim: int = 1,
) -> torch.Tensor:
    """
    Apply rotary positional embedding to x.

    Args:
        x: Tensor of shape (B, N, D)
        base: RoPE base
        seq_dim: sequence dimension, default=1 for (B, N, D)

    Returns:
        Tensor of same shape as x
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape (B, N, D), got {tuple(x.shape)}")

    B, N, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"Embedding dimension D must be even, got D={D}")

    device = x.device
    dtype = x.dtype

    half_d = D // 2

    # Frequencies for even positions
    inv_freq = 1.0 / (base ** (torch.arange(0, half_d, device=device, dtype=torch.float32) / half_d))

    # Positions: [N]
    positions = torch.arange(N, device=device, dtype=torch.float32)

    # Angles: [N, D/2]
    angles = torch.outer(positions, inv_freq)

    # Expand each angle to match pairwise dims: [N, D]
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

    cos = angles.cos().to(dtype).unsqueeze(0)  # [1, N, D]
    sin = angles.sin().to(dtype).unsqueeze(0)  # [1, N, D]

    return x * cos + rotate_half(x) * sin

def apply_rope_qk(q: torch.Tensor, k: torch.Tensor, base: float = 10000.0):
    return rotary_emb(q, base=base), rotary_emb(k, base=base)


def main():
    x = torch.randn(2, 4, 8)  # (B, N, D)
    y = apply_rope_qk(x, x)
    print(y[0].shape)  # Should be (2, 4, 8)
    print(y[1].shape)  # Should be (2, 4, 8)

    pass

if __name__ == "__main__":
    main()
