import torch.nn as nn
import torch


class BasePosEmbedding(nn.Module):
    """
    Base class for positional embeddings.

    Subclasses should implement the forward method to return a positional embedding
    tensor that can be added to the input tensor.

    Supported input shapes:
        - (B, N, D)
        - (B, H, N, D)

    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class IdentityPosEmbedding(BasePosEmbedding):
    """
    Positional embedding that returns the input tensor unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
