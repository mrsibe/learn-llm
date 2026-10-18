"""单卡训练生命周期：AMP、调度、固定验证、latest/best 恢复与可选 Profiling。"""

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import (
    FORMAT_VERSION,
    atomic_save,
    load_checkpoint,
    preserve_best,
    restore_rng,
    rng_state,
    validate_resume,
)
from .config import ExperimentConfig, GPTConfig, TrainConfig, load_config
from .data import TokenData
from .evaluate import evaluate, validate_windows, validation_starts
from .metadata import environment_info
from .model import GPT
from .optim import Precision, build_optimizer, build_scheduler, optimizer_step


def make_profiler(out, device, active_steps):
    if not active_steps:
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0, warmup=1, active=active_steps, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out / "profile")),
        record_shapes=True,
        profile_memory=True,
    )


def train(model_cfg: GPTConfig, cfg: TrainConfig, resume=False, verify_data=False):
    cfg.__post_init__()
    model_cfg.__post_init__()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % 2**32)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = Precision(device, cfg.dtype)
    data = TokenData(cfg.data_dir, model_cfg.block_size, verify=verify_data)
    model_cfg = replace(model_cfg, vocab_size=data.vocab_size)
    out = Path(cfg.out_dir)
    path = out / "ckpt.pt"
    if (path.exists() or (out / "best.pt").exists()) and not resume:
        raise FileExistsError(f"{out} 已有 checkpoint，请用 --resume 或更换输出目录")
    source = path if resume is True else Path(resume) if resume else None
    checkpoint = load_checkpoint(source) if resume else None
    if checkpoint:
        validate_resume(checkpoint, model_cfg, cfg, data, precision)

    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    starts = validation_starts(data, cfg.batch_size, cfg.eval_batches, cfg.eval_seed)
    start = tokens_seen = updates = best_step = 0
    best_loss, last_val_loss = float("inf"), None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        precision.scaler.load_state_dict(checkpoint["scaler"])
        start, tokens_seen, updates = (
            checkpoint["step"],
            checkpoint["tokens_seen"],
            checkpoint["optimizer_updates"],
        )
        starts = checkpoint["validation_starts"]
        validate_windows(data, starts)
        if len(starts) != cfg.batch_size * cfg.eval_batches:
            raise ValueError("checkpoint 验证窗口数量不匹配")
        best_loss, best_step = preserve_best(checkpoint, source, out)
        last_val_loss = checkpoint["val_loss"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(asdict(ExperimentConfig(model_cfg, cfg)))
    )
    environment = environment_info(device) | {"dtype": precision.name}
    (out / "metadata.json").write_text(json.dumps(environment, indent=2) + "\n")
    origin_environment = checkpoint["origin_environment"] if checkpoint else environment
    tokens_per_step = cfg.batch_size * cfg.grad_accum * model_cfg.block_size
    print(
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 参数 | {device}/{precision.name} | "
        f"{tokens_per_step:,} tokens/step | 计划 {cfg.steps * tokens_per_step:,} tokens "
        f"({cfg.steps * tokens_per_step / len(data.splits['train']):.2f} token 等效 epochs)"
    )

    with (
        SummaryWriter(
            log_dir=str(out / "events"), purge_step=start + 1 if resume else None
        ) as writer,
        make_profiler(out, device, cfg.profile_steps) as profiler,
    ):
        # 所有初始化之后再恢复，避免初始化消耗训练的 RNG 流。
        if checkpoint:
            restore_rng(checkpoint, device)
            checkpoint = None  # 状态已恢复，不长期保留 CPU 上的模型/优化器副本。
        for step in range(start, cfg.steps):
            model.train()
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            lr = optimizer.param_groups[0]["lr"]

            def region(name):
                return (
                    torch.profiler.record_function(name)
                    if profiler is not None
                    else nullcontext()
                )

            for _ in range(cfg.grad_accum):
                with region("data"):
                    x, y = data.batch("train", cfg.batch_size, device)
                with region("forward"), precision.autocast():
                    _, loss = model(x, y)
                with region("backward"):
                    precision.scaler.scale(loss / cfg.grad_accum).backward()
                loss_sum += loss.detach() / cfg.grad_accum
            train_loss = loss_sum.item()
            if not math.isfinite(train_loss):
                raise RuntimeError(f"step {step + 1}: loss 非有限值")
            with region("optimizer"):
                updated, grad_norm = optimizer_step(
                    model, optimizer, precision, cfg.grad_clip
                )
                if updated:
                    updates += 1
                    scheduler.step()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens_seen += tokens_per_step  # 包括 FP16 跳过更新时实际读取的 tokens
            metrics = {
                "loss/train": train_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm,
                "train/tokens_seen": tokens_seen,
                "train/optimizer_updates": updates,
                "train/skipped_updates": step + 1 - updates,
                "train/token_epochs": tokens_seen / len(data.splits["train"]),
                "perf/step_ms": elapsed * 1000,
                "perf/tokens_per_s": tokens_per_step / elapsed,
            }
            if precision.scaler.is_enabled():
                metrics["train/grad_scale"] = precision.scaler.get_scale()
            if device == "cuda":
                metrics |= {
                    "perf/peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "perf/peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            for tag, value in metrics.items():
                writer.add_scalar(tag, value, step + 1)
            if (step + 1) % 10 == 0:
                print(
                    f"step {step + 1}: loss {train_loss:.4f}, lr {lr:.2e}, "
                    f"{tokens_per_step / elapsed:.0f} tok/s, skipped={step + 1 - updates}",
                    flush=True,
                )
            if profiler is not None:
                profiler.step()

            improved = False
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                result = evaluate(model, data, starts, cfg.batch_size, precision)
                last_val_loss = result["loss"]
                improved = last_val_loss < best_loss
                if improved:
                    best_loss, best_step = last_val_loss, step + 1
                writer.add_scalar("loss/val", last_val_loss, step + 1)
                if result["perplexity"] is not None:
                    writer.add_scalar("val/perplexity", result["perplexity"], step + 1)
                print(
                    f"step {step + 1}: val loss {last_val_loss:.4f}, best={best_loss:.4f} @ {best_step}",
                    flush=True,
                )

            if improved or (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
                writer.flush()
                payload = {
                    "format_version": FORMAT_VERSION,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": precision.scaler.state_dict(),
                    "precision": precision.name,
                    "config": asdict(model_cfg),
                    "train_config": asdict(cfg),
                    "data_meta": data.meta,
                    "tokenizer_json": data.tokenizer_json,
                    "validation_starts": starts,
                    "step": step + 1,
                    "optimizer_updates": updates,
                    "tokens_seen": tokens_seen,
                    "val_loss": last_val_loss,
                    "best_val_loss": best_loss,
                    "best_step": best_step,
                    "environment": environment,
                    "origin_environment": origin_environment,
                    "data_verified": verify_data,
                    **rng_state(device),
                }
                atomic_save(path, payload)
                if improved:
                    atomic_save(out / "best.pt", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gpt2_42m.yaml")
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("--steps", type=int, help="总训练步数，独立于固定 decay_steps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16", "fp16"))
    parser.add_argument("--profile-steps", type=int)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="默认恢复 out/ckpt.pt，也可指定路径",
    )
    parser.add_argument("--verify-data", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {
        k: v
        for k, v in {
            "data_dir": args.data,
            "out_dir": args.out,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "dtype": args.dtype,
            "profile_steps": args.profile_steps,
        }.items()
        if v is not None
    }
    train(
        config.model,
        replace(config.train, **overrides),
        resume=args.resume,
        verify_data=args.verify_data,
    )


if __name__ == "__main__":
    main()
