import pytest
import torch

from tinylm.config import TinyLMConfig
from tinylm.layers.activation import GeluAndMul
from tinylm.testutil import allclose


@pytest.mark.parametrize("dim", [16, 64, 256])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [1, 4, 16])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_GeluAndMul(
    dim: int,
    batch_size: int,
    seq_len: int,
    approximate: str,
    dtype: torch.dtype,
    is_nvidia: bool,
    config: TinyLMConfig,
) -> None:
    x = torch.randn(batch_size, seq_len, 2 * dim).cuda().to(dtype)
    layer = GeluAndMul(approximate=approximate)
    expected_output = layer.forward_torch(x)
    output = layer(x)

    assert output.shape == expected_output.shape
    assert allclose(expected_output, output)
