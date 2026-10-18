"""完整训练状态与原子保存；不实现旧版本 checkpoint 迁移。"""

import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 2
RESUME_FIELDS = (
    "batch_size",
    "grad_accum",
    "lr",
    "weight_decay",
    "seed",
    "dtype",
    "beta1",
    "beta2",
    "grad_clip",
    "scheduler",
    "warmup_steps",
    "decay_steps",
    "min_lr",
    "eval_batches",
    "eval_seed",
)


def rng_state(device):
    numpy = np.random.get_state()
    state = {
        "rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
    }
    if device == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["rng"])
    random.setstate(state["python_rng"])
    numpy = state["numpy_rng"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    if device == "cuda" and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"])


def atomic_save(path: Path, payload):
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "config",
        "train_config",
        "step",
        "optimizer_updates",
        "tokens_seen",
        "best_val_loss",
        "val_loss",
        "best_step",
        "validation_starts",
        "precision",
        "data_meta",
        "tokenizer_json",
        "rng",
        "python_rng",
        "numpy_rng",
        "environment",
        "origin_environment",
    }
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != FORMAT_VERSION
        or required - checkpoint.keys()
    ):
        raise ValueError("不支持的或不完整的 V2 checkpoint")
    if (
        type(checkpoint["step"]) is not int
        or type(checkpoint["optimizer_updates"]) is not int
        or not 0 <= checkpoint["optimizer_updates"] <= checkpoint["step"]
        or checkpoint["scheduler"].get("last_epoch") != checkpoint["optimizer_updates"]
    ):
        raise ValueError("checkpoint 更新计数与 scheduler 状态不一致")
    return checkpoint


def preserve_best(checkpoint, source: Path, out: Path):
    """恢复 latest/best 的部分写入，也支持将完整历史最佳状态迁移到新输出目录。"""

    def identity(state):
        return (
            state["config"],
            state["data_meta"],
            state["tokenizer_json"],
            state["precision"],
            {key: state["train_config"][key] for key in RESUME_FIELDS},
        )

    best = checkpoint if checkpoint["step"] == checkpoint["best_step"] else None
    target = None
    for path in dict.fromkeys(
        ((source.parent / "best.pt").resolve(), (out / "best.pt").resolve())
    ):
        if not path.exists():
            continue
        candidate = (
            checkpoint if path.resolve() == source.resolve() else load_checkpoint(path)
        )
        if (
            identity(candidate) != identity(checkpoint)
            or candidate["step"] != candidate["best_step"]
            or candidate["val_loss"] != candidate["best_val_loss"]
        ):
            raise ValueError(f"历史 best checkpoint 无效或属于其他实验：{path}")
        if path.resolve() == (out / "best.pt").resolve():
            target = candidate
        if (
            best is None
            or candidate["best_val_loss"] < best["best_val_loss"]
            or (
                candidate["best_val_loss"] == best["best_val_loss"]
                and candidate["best_step"] == best["best_step"]
            )
        ):
            best = candidate
    if best is not None and (
        not math.isfinite(best["best_val_loss"])
        or best["best_val_loss"] > checkpoint["best_val_loss"]
    ):
        best = None
    if best is None:
        if checkpoint["best_step"]:
            raise ValueError("历史 best 权重缺失，不能安全续训；请恢复原始 best.pt")
        return float("inf"), 0
    if (
        best["best_val_loss"] == checkpoint["best_val_loss"]
        and best["best_step"] != checkpoint["best_step"]
    ):
        raise ValueError("历史 best 的 step 与 checkpoint 不一致")
    if target is not best:
        out.mkdir(parents=True, exist_ok=True)
        atomic_save(out / "best.pt", best)
    return best["best_val_loss"], best["best_step"]


def validate_resume(checkpoint, model_cfg, cfg, data, precision):
    if checkpoint["config"] != asdict(model_cfg):
        raise ValueError("续训模型配置与 checkpoint 不一致")
    if (
        checkpoint["data_meta"] != data.meta
        or checkpoint["tokenizer_json"] != data.tokenizer_json
    ):
        raise ValueError("续训数据或 tokenizer 与 checkpoint 不一致")
    for name in RESUME_FIELDS:
        if checkpoint["train_config"][name] != getattr(cfg, name):
            raise ValueError(f"续训不能修改 {name}")
    if checkpoint["precision"] != precision.name:
        raise ValueError("续训实际精度模式与 checkpoint 不一致")
    if cfg.steps <= checkpoint["step"]:
        raise ValueError("续训 steps 必须大于 checkpoint 已完成步数")
