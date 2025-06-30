import torch

from tinylm.layers.rotary_embedding import get_rotary_embedding
from tinylm.testutil import allclose


def test_rotary_embedding_correctness():
    batch_size = 2
    head_size = 2
    seq_len = 2
    num_heads = 1
    device = torch.device("cpu")
    dtype = torch.bfloat16

    rotary_emb = get_rotary_embedding(
        head_size=head_size,
        rotary_dim=head_size,
        max_position_embeddings=seq_len,
        base=1000.0,
    )
    rotary_emb = rotary_emb.to(device=device)

    num_tokens = batch_size * seq_len
    input_size = num_tokens * head_size * num_heads
    shape = num_tokens, num_heads, head_size
    pos = torch.arange(seq_len, dtype=torch.long, device=device).repeat(batch_size)
    q = 1 + torch.arange(input_size, dtype=dtype, device=device).reshape(shape)
    k = 16 - torch.arange(input_size, dtype=dtype, device=device).reshape(shape)

    q_rotated, k_rotated = rotary_emb(pos, q, k)

    q_rotated_ref = torch.tensor(
        [[[1.0000, 2.0000]], [[-1.7422, 4.6875]], [[5.0000, 6.0000]], [[-2.9531, 10.1875]]],
        dtype=dtype,
        device=device,
    )
    k_rotated_ref = torch.tensor(
        [[[16.0000, 15.0000]], [[-3.3750, 18.7500]], [[12.0000, 11.0000]], [[-2.1719, 13.2500]]],
        dtype=dtype,
        device=device,
    )
    assert allclose(q_rotated, q_rotated_ref)
    assert allclose(k_rotated, k_rotated_ref)
