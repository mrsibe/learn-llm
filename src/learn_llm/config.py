"""共享的 Decoder-only 模型与单卡训练配置；通过具体组件做架构消融。"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    attention_backend: str = "manual"
    norm_type: str = "layernorm"
    position_type: str = "learned"
    mlp_type: str = "gelu"
    n_kv_heads: int | None = None  # None 使用 n_head，即 MHA；1 为 MQA
    ffn_hidden_size: int | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bias: bool = True
    tie_embeddings: bool = True

    @property
    def kv_heads(self):
        return self.n_head if self.n_kv_heads is None else self.n_kv_heads

    @property
    def ffn_size(self):
        if self.ffn_hidden_size is not None:
            return self.ffn_hidden_size
        if self.mlp_type == "gelu":
            return 4 * self.n_embd
        # SwiGLU 有三组矩阵，使用约 8/3*d 而不是 4*d，并向上对齐到 64。
        return math.ceil((8 * self.n_embd / 3) / 64) * 64

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd 必须能被 n_head 整除")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        if self.attention_backend not in ("manual", "sdpa"):
            raise ValueError("attention_backend 只能是 manual 或 sdpa")
        for name, choices in (
            ("norm_type", ("layernorm", "rmsnorm")),
            ("position_type", ("learned", "rope")),
            ("mlp_type", ("gelu", "swiglu")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} 必须是 {choices} 之一")
        for name in ("n_kv_heads", "ffn_hidden_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} 必须为正整数或 null")
        if self.n_head % self.kv_heads:
            raise ValueError("n_head 必须能被 n_kv_heads 整除")
        if self.position_type == "rope" and (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE 要求 head_dim 为偶数")
        for name in ("norm_eps", "rope_theta"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} 必须为有限正数")
        if type(self.bias) is not bool or type(self.tie_embeddings) is not bool:
            raise ValueError("bias 和 tie_embeddings 必须为布尔值")


@dataclass
class TrainConfig:
    data_dir: str = "data/tinystories"
    out_dir: str = "out/gpt2"
    steps: int = 1000
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    eval_every: int = 100
    eval_batches: int = 10
    seed: int = 1337
    device: str = "auto"

    def __post_init__(self):
        for name in ("steps", "batch_size", "grad_accum", "eval_every", "eval_batches"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr 必须 > 0，weight_decay 必须 >= 0")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 只能是 auto、cpu 或 cuda")


@dataclass
class ExperimentConfig:
    model: GPTConfig = field(default_factory=GPTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - {"model", "train"}:
        raise ValueError("YAML 顶层只允许 model 和 train")
    return ExperimentConfig(
        GPTConfig(**raw.get("model", {})), TrainConfig(**raw.get("train", {}))
    )
