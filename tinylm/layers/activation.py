import torch
from flashinfer.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from torch.nn import functional as F

from tinylm.layers.base import AcceleratedModule


class GeluAndMul(AcceleratedModule[torch.Tensor]):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        assert approximate in ("none", "tanh")
        if approximate == "none":
            self.op = gelu_and_mul
        else:
            self.op = gelu_tanh_and_mul
        self.approximate = approximate

    @torch.compile
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_flashinfer(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SiluAndMul(AcceleratedModule[torch.Tensor]):
    @torch.compile
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_flashinfer(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)
