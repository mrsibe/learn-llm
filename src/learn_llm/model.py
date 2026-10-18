"""共享 Pre-Norm Decoder 骨架；配置决定 GPT-2 或 Llama 风格组件。"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .attention import CausalSelfAttention
from .config import GPTConfig
from .layers import GELUMLP, SwiGLU, build_norm


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = build_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = build_norm(cfg)
        self.mlp = GELUMLP(cfg) if cfg.mlp_type == "gelu" else SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = (
            nn.Embedding(cfg.block_size, cfg.n_embd)
            if cfg.position_type == "learned"
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = build_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # 所有残差分支输出投影（包括 SwiGLU）都按深度缩放初始化。
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _B, T = idx.shape
        if not 0 < T <= self.cfg.block_size:
            raise ValueError(
                f"序列长度必须在 1..{self.cfg.block_size} 之间，实际为 {T}"
            )
        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1]]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
