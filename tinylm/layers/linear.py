from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class PackedLinear(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dims_out: list[int],
        bias: bool = True,
        skip_bias_add: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.skip_bias_add = skip_bias_add
        self.weight = nn.Parameter(torch.empty(sum(dims_out), dim_in, dtype=dtype))
        self.bias: Optional[nn.Parameter] = None
        if bias:
            self.bias = nn.Parameter(torch.empty(sum(dims_out), dtype=dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(x, self.weight, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class QKVPackedLinear(PackedLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        dims_out = [
            num_heads * head_size,  # q_proj
            num_kv_heads * head_size,  # k_proj
            num_kv_heads * head_size,  # v_proj
        ]
        super().__init__(hidden_size, dims_out, bias, skip_bias_add=skip_bias_add)
