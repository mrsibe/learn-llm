"""复用 PyTorch 的 AdamW、AMP、GradScaler 和调度器，显式定义训练策略。"""

import math

import torch

from .config import TrainConfig


class Precision:
    def __init__(self, device: str, requested="auto"):
        if requested == "auto":
            requested = (
                "fp32"
                if device == "cpu"
                else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
            )
        if requested not in ("fp32", "bf16", "fp16"):
            raise ValueError("未知精度模式")
        if device == "cpu" and requested != "fp32":
            raise ValueError("CPU 训练仅支持 fp32")
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("当前 CUDA 设备不支持 BF16")
        self.name = requested
        self.device = device
        self.dtype = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[requested]
        self.scaler = torch.amp.GradScaler(device, enabled=requested == "fp16")

    def autocast(self):
        return torch.autocast(
            self.device, dtype=self.dtype, enabled=self.name != "fp32"
        )


def build_optimizer(model, cfg: TrainConfig):
    groups = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def learning_rate(update: int, cfg: TrainConfig):
    """update 为从 0 开始的成功优化器更新索引；计划不依赖运行的 steps。"""
    if cfg.scheduler == "constant":
        return cfg.lr
    if update < cfg.warmup_steps:
        return cfg.lr * (update + 1) / cfg.warmup_steps
    if update >= cfg.decay_steps - 1:
        return cfg.min_lr
    progress = (update - cfg.warmup_steps) / (cfg.decay_steps - cfg.warmup_steps - 1)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def build_scheduler(optimizer, cfg: TrainConfig):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: learning_rate(update, cfg) / cfg.lr
    )


def optimizer_step(model, optimizer, precision: Precision, grad_clip: float):
    """FP16 溢出时由 GradScaler 跳过更新并降低 scale；返回是否真正更新。"""
    scaler = precision.scaler
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        grad_clip if grad_clip else math.inf,
        error_if_nonfinite=not scaler.is_enabled(),
    )
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
    return updated, norm.detach().item()
