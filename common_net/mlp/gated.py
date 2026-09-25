import torch
import torch.nn as nn
from typing import Optional
from common_net.common import Gated

class GatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        bias: bool = False,
        gate_act_fn: str = "sigmoid",
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim  #

        self.up = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.gate = Gated(
            input_dim,
            hidden_dim,
            bias=bias,
            gate_act_fn=gate_act_fn,
            gate_operator_fn="*",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)  # y = fc(x)
        gate_output = self.gate(x, y)  # y' = y o act(x@W(+B))
        return self.down(gate_output)  # out = fc(y)

