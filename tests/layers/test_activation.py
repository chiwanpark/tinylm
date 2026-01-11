from jax import numpy as jnp

from tinylm.layers.activation import GeluAndMul, SiluAndMul
from tinylm.testutil import allclose


def test_gelu_and_mul_correctness() -> None:
    x = jnp.array([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=jnp.bfloat16)
    layer = GeluAndMul(approximate=False)
    output = layer(x)

    expected = jnp.array([[0.4766, 7.8125], [35, 0]], dtype=jnp.bfloat16)
    assert allclose(expected, output)


def test_silu_and_mul_correctness() -> None:
    x = jnp.array([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=jnp.bfloat16)
    layer = SiluAndMul()
    output = layer(x)

    expected = jnp.array([[0.8086, 7.0312], [34.7500, 0.1187]], dtype=jnp.bfloat16)
    assert allclose(expected, output)
