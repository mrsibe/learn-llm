"""因果 MHA/GQA：手写数学参考，以及 PyTorch SDPA 后端。"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import GPTConfig
from .layers import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.kv_heads
        self.head_dim = cfg.n_embd // cfg.n_head
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.backend = cfg.attention_backend
        self.qkv = nn.Linear(cfg.n_embd, cfg.n_embd + 2 * self.kv_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = (
            RotaryEmbedding(self.head_dim, cfg.rope_theta)
            if cfg.position_type == "rope"
            else None
        )
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split((C, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = [
            t.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) for t in (k, v)
        ]
        if self.rope is not None:
            q, k = self.rope(q, k)
        if self.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.drop.p if self.training else 0.0,
                enable_gqa=self.n_head != self.n_kv_heads,
            )
        else:
            # 连续的一组 Query heads 共享一个 KV head；参考实现显式展开。
            repeats = self.n_head // self.n_kv_heads
            if repeats > 1:
                k, v = [t.repeat_interleave(repeats, dim=1) for t in (k, v)]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
            y = self.drop(F.softmax(scores, dim=-1)) @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(y))
