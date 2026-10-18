"""仅定义 GPT-2 和单卡训练配置，不预留其他架构的空接口。"""

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
