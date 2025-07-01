from typing import Optional

import torch
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from torch import nn

from tinylm.layers.base import AcceleratedModule


class RMSNorm(AcceleratedModule[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.compile
    def _rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float)
        norm = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x.mul_(norm).to(dtype).mul_(self.weight)
        return x

    @torch.compile
    def _add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        x = x.to(torch.float).add_(residual.to(torch.float))
        residual = x.to(dtype)
        norm = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x.mul_(norm).to(dtype).mul_(self.weight)
        return x, residual

    def forward_torch(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._rms_forward(x)
        else:
            return self._add_rms_forward(x, residual)

    def forward_flashinfer(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rmsnorm(x, self.weight.data, self.eps)
        else:
            fused_add_rmsnorm(x, residual, self.weight.data, self.eps)
            return x, residual
