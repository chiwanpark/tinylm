from functools import lru_cache

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size

        # build the cosine and sine cache
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _apply_rotary_embedding(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * cos + x2 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        q_shape = query.shape
        q_rotated = query.view(num_tokens, -1, self.head_size)
        q_rotated = self._apply_rotary_embedding(q_rotated, cos, sin)
        q_rotated = q_rotated.view(q_shape)

        k_shape = key.shape
        k_rotated = key.view(num_tokens, -1, self.head_size)
        k_rotated = self._apply_rotary_embedding(k_rotated, cos, sin)
        k_rotated = k_rotated.view(k_shape)

        return q_rotated, k_rotated


@lru_cache(maxsize=1)
def get_rotary_embedding(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
) -> RotaryEmbedding:
    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
    )
