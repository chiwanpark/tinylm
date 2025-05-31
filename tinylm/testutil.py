from typing import Optional

import torch

# we borrow these values from PyTorch's values.
# see https://github.com/pytorch/pytorch/blob/d6edefefbf3688d9aa75c16acb7d3452bab0b380/test/test_transformers.py#L67 for details.
_default_atol = {
    torch.float16: 1e-3,
    torch.float32: 1e-5,
    torch.bfloat16: 1e-3,
}

_default_rtol = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 1.6e-2,
}


def allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> bool:
    assert a.shape == b.shape, f"Shapes do not match: {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"Tensor types do not match: {a.dtype} vs {b.dtype}"
    if atol is None:
        atol = _default_atol[a.dtype]
    if rtol is None:
        rtol = _default_rtol[a.dtype]
    return torch.allclose(a, b, atol=atol, rtol=rtol)
