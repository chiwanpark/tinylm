import pytest
import torch
from torch import nn

from tinylm.layers.linear import PackedLinear
from tinylm.testutil import allclose


def build_packed_linear(ref1: nn.Linear, ref2: nn.Linear) -> PackedLinear:
    assert ref1.weight.shape == ref2.weight.shape
    if ref1.bias is not None:
        assert ref2.bias is not None
        assert ref1.bias.shape == ref2.bias.shape
    layer = PackedLinear(
        dim_in=ref1.in_features,
        dims_out=[ref1.out_features, ref2.out_features],
        bias=ref1.bias is not None,
        skip_bias_add=ref1.bias is None,
        dtype=ref1.weight.dtype,
    ).to(ref1.weight.device)
    layer.weight.data.copy_(torch.cat([ref1.weight, ref2.weight], dim=0))
    if layer.bias is not None:
        layer.bias.data.copy_(torch.cat([ref1.bias, ref2.bias], dim=0))
    return layer


@pytest.mark.parametrize("skip_bias_add", [True, False])
@pytest.mark.parametrize("dim", [16, 64, 256])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [1, 4, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_linear(
    skip_bias_add: bool,
    dim: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    is_nvidia: bool,
) -> None:
    l1 = nn.Linear(dim, dim // 2, bias=not skip_bias_add, dtype=dtype).cuda()
    l2 = nn.Linear(dim, dim // 2, bias=not skip_bias_add, dtype=dtype).cuda()
    l_packed = build_packed_linear(l1, l2).cuda()
    x = torch.randn(batch_size, seq_len, dim).cuda().to(dtype)

    out, _ = l_packed(x)
    assert allclose(out, torch.cat([l1(x), l2(x)], dim=-1))
