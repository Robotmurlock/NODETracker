"""
Implementation of multi-head attention from "Attention is all you need" paper
"""
import math
from typing import Optional

import torch
from torch import nn


class DotProductAttention(nn.Module):
    """
    Dot-scaled product attention.
    """
    NEG_INF = -999_999

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, _, _, dim = k.shape

        k_t = torch.transpose(k, 2, 3)
        scaled_dot_product = (q @ k_t) / math.sqrt(dim)  # q @ k.T / sqrt(dim)

        if mask is not None:
            scaled_dot_product = torch.masked_fill(scaled_dot_product, mask=mask, value=self.NEG_INF)

        score = torch.softmax(scaled_dot_product, dim=-1)  # softmax(q @ k.T / sqrt(dim))
        return score @ v  # softmax(q @ k.T / sqrt(dim)) @ v


class MultiHeadAttention(nn.Module):
    """
    Implementation of multi-head attention without projections.
    Base for SelfAttention and CrossAttention.
    Ref: https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, n_heads: int):
        super().__init__()
        self._n_heads = n_heads
        self._attention = DotProductAttention()

    def _multi_head_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Multi head attention without projections.

        Args:
            q: Query
            k: Keys
            v: Values

        Returns:
            Attention result
        """
        q, k, v = self._split(q), self._split(k), self._split(v)
        o = self._attention(q, k, v)
        o = self._concat(o)
        return o

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits input tensor dimension by number of head (over batch and temporal dimension)

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        batch_size, time_len, full_dim = x.shape  # B x T x (H * D)
        assert full_dim % self._n_heads == 0, \
            f'Dimension ({full_dim}) is not divisible by number of heads ({self._n_heads}).'

        head_dim = full_dim // self._n_heads
        x = x.reshape(batch_size, time_len, self._n_heads, head_dim)    # B x T x H x D
        x = torch.transpose(x, 1, 2).contiguous()  # B x H x T x D
        return x

    def _concat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concatenates input tensor that was previously split.

        Args:
            x: Split tensor

        Returns:
            Combined tensor
        """
        batch_size, n_heads, time_len, head_dim = x.shape  # B x H x T x D
        assert n_heads == self._n_heads

        x = torch.transpose(x, 1, 2)  # B x T x H x D
        x = x.reshape(batch_size, time_len, n_heads * head_dim) # B x T x (H * D)
        return x


class MultiHeadSelfAttention(MultiHeadAttention):
    """
    Implementation of self multi-head attention.
    """
    def __init__(self, n_heads: int, dim: int):
        """
        Args:
            n_heads: Number of attention heads
            dim: Input/output dimension
        """
        super().__init__(n_heads=n_heads)
        self._n_heads = n_heads
        self._w_q = nn.Linear(dim, dim)
        self._w_k = nn.Linear(dim, dim)
        self._w_v = nn.Linear(dim, dim)
        self._w_o = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._w_q(x), self._w_k(x), self._w_v(x)  # project input
        o = self._multi_head_attention(q, k, v)
        o = self._w_o(o)  # project output
        return o


class MultiHeadCrossAttention(MultiHeadAttention):
    """
    Implementation of cross multi-head attention.
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self._multi_head_attention(q, k, v)


class TemporalFirstMultiHeadSelfAttention(nn.Module):
    """
    Wrapper for `MultiHeadSelfAttention` in case of "batch_first=False" input.
    """
    def __init__(self, n_heads: int, dim: int):
        """
        Args:
            n_heads: Number of attention heads
            dim: Input/output dimension
        """
        super().__init__()
        self._mhsa = MultiHeadSelfAttention(n_heads=n_heads, dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 0, 1).contiguous()
        x = self._mhsa(x)
        x = torch.transpose(x, 0, 1).contiguous()
        return x


class TemporalFirstMultiHeadCrossAttention(nn.Module):
    """
    Wrapper for `MultiHeadCrossAttention` in case of "batch_first=False" input.
    """
    def __init__(self, n_heads: int):
        """
        Args:
            n_heads: Number of attention heads
        """
        super().__init__()
        self._mhca = MultiHeadCrossAttention(n_heads=n_heads)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k, v = [torch.transpose(value, 0, 1).contiguous() for value in [q, k, v]]
        x = self._mhca(q, k, v)
        x = torch.transpose(x, 0, 1)
        return x


def run_test() -> None:
    mha = MultiHeadSelfAttention(4, 16)
    x = torch.randn(10, 3, 16)
    o = mha(x)
    print(f'Output shape: {o.shape}')

    tf_mhsa = TemporalFirstMultiHeadSelfAttention(4, 16)
    x = torch.randn(3, 10, 16)
    o = tf_mhsa(x)
    print(f'Output shape (: {o.shape}')


if __name__ == '__main__':
    run_test()
