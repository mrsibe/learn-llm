import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.backend = cfg.attention_backend
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # QKV 合一
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = [
            t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        ]
        if self.backend == "sdpa":
            # SDPA 不会自动根据 eval() 关闭 dropout，需要显式传入 0。
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.drop.p if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
            y = self.drop(F.softmax(att, dim=-1)) @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # 权重绑定

        self.apply(self._init_weights)
        # GPT-2 初始化：残差分支的输出投影再按 1/sqrt(2*n_layer) 缩放，
        # 避免残差流方差随深度累积。只命中 attn.proj.weight / mlp.proj.weight。
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
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1]]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
