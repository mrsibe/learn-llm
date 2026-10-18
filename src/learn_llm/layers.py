"""手写归一化、位置旋转和 FFN；不包含缓存或框架注册表。"""

import torch
import torch.nn.functional as F
from torch import nn

from .config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 低精度输入在 FP32 中计算平方均值，避免 FP16/BF16 的数值问题。
        # 保留 FP64，便于参考计算与 gradcheck。
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(
            work.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def build_norm(cfg: GPTConfig):
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.n_embd, cfg.norm_eps)
    return nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps, bias=cfg.bias)


class RotaryEmbedding(nn.Module):
    """对相邻的偶/奇维度旋转；完整序列的位置从 0 开始，不实现 KV Cache。"""

    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE 要求 head_dim 为偶数")
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, q, k):
        # 按需构造 FP32 频率，避免 module.to(bfloat16) 把持久频率表降精度。
        frequency = self.theta ** (
            -torch.arange(0, self.head_dim, 2, device=q.device, dtype=torch.float32)
            / self.head_dim
        )
        positions = torch.arange(q.size(-2), device=q.device, dtype=torch.float32)
        angles = positions[:, None] * frequency[None, :]
        cos, sin = angles.cos(), angles.sin()

        def rotate(x):
            work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            even, odd = work[..., ::2], work[..., 1::2]
            c, s = cos.to(work.dtype), sin.to(work.dtype)
            rotated = torch.stack(
                (even * c - odd * s, even * s + odd * c), dim=-1
            ).flatten(-2)
            return rotated.to(x.dtype)

        return rotate(q), rotate(k)


class GELUMLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.ffn_size, bias=cfg.bias)
        self.proj = nn.Linear(cfg.ffn_size, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.ffn_size, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, cfg.ffn_size, bias=cfg.bias)
        self.proj = nn.Linear(cfg.ffn_size, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))
