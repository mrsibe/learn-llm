"""合成 token 的独立性能测量；不包含下载、数据读取、验证或 checkpoint。"""

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .config import GPTConfig, TrainConfig, load_config
from .metadata import environment_info
from .model import GPT

MODES = ("forward", "forward_backward", "train_step")


def benchmark(
    model_cfg: GPTConfig,
    cfg: TrainConfig,
    *,
    mode="train_step",
    warmup=5,
    steps=20,
    dtype="auto",
    loss_readback="per-step",
):
    if mode not in MODES or warmup < 1 or steps < 1:
        raise ValueError("mode 无效，warmup 和 steps 必须 >= 1")
    if loss_readback not in ("per-step", "per-microbatch"):
        raise ValueError("loss_readback 无效")
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if dtype == "auto":
        dtype = "bf16" if bf16 else "fp32"
    if dtype not in ("fp32", "bf16") or (dtype == "bf16" and not bf16):
        raise ValueError("仅支持 fp32 或支持 CUDA BF16 的设备")
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.empty_cache()
    model = GPT(model_cfg).to(device)
    model.train(mode != "forward")
    groups = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    optimizer = (
        torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
        if mode == "train_step"
        else None
    )
    accum = cfg.grad_accum if mode == "train_step" else 1
    # 整个测量复用同一组 GPU 输入，不将随机数生成/主机拷贝计入吞吐。
    inputs = [
        torch.randint(
            model_cfg.vocab_size,
            (cfg.batch_size, model_cfg.block_size + 1),
            device=device,
        )
        for _ in range(accum)
    ]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def iteration(measure):
        events = []
        phases = {"forward_ms": 0.0, "backward_ms": 0.0}

        def phase(name, fn):
            if not measure:
                return fn()
            if device == "cuda":
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                value = fn()
                end.record()
                events.append((name, begin, end))
            else:
                begin = time.perf_counter()
                value = fn()
                phases[name] += (time.perf_counter() - begin) * 1000
            return value

        sync()
        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        loss_sum = (
            0.0 if loss_readback == "per-microbatch" else torch.zeros((), device=device)
        )
        for tokens in inputs:

            def forward(tokens=tokens):
                with (
                    torch.set_grad_enabled(mode != "forward"),
                    torch.autocast(
                        device_type=device,
                        dtype=torch.bfloat16,
                        enabled=dtype == "bf16",
                    ),
                ):
                    # targets 非空才计算全部位置；避免把仅生成最后一个 logit 的路径当训练前向。
                    return model(tokens[:, :-1], tokens[:, 1:])[1]

            loss = phase("forward_ms", forward)
            # 保留旧训练循环作为同步开销的诊断对照，不作为推荐实现。
            if loss_readback == "per-microbatch" and not torch.isfinite(loss):
                raise RuntimeError("Non-finite benchmark loss")
            if mode != "forward":
                phase("backward_ms", lambda loss=loss: (loss / accum).backward())
            if loss_readback == "per-microbatch":
                loss_sum += loss.detach().item() / accum
            else:
                loss_sum += loss.detach() / accum
        value = loss_sum if isinstance(loss_sum, float) else loss_sum.item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite benchmark loss")
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        sync()
        elapsed = (time.perf_counter() - start) * 1000
        for name, begin, end in events:
            phases[name] += begin.elapsed_time(end)
        return elapsed, phases

    for _ in range(warmup):
        iteration(False)
    sync()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples = [iteration(True) for _ in range(steps)]
    times = [sample[0] for sample in samples]
    tokens_per_iteration = cfg.batch_size * model_cfg.block_size * accum
    return {
        "mode": mode,
        "dtype": dtype,
        "device": device,
        "loss_readback": loss_readback,
        "model": asdict(model_cfg),
        "train": asdict(cfg),
        "warmup": warmup,
        "steps": steps,
        "effective_grad_accum": accum,
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_per_iteration": tokens_per_iteration,
        "tokens_per_s": tokens_per_iteration * 1000 / statistics.mean(times),
        "step_ms_mean": statistics.mean(times),
        "step_ms_median": statistics.median(times),
        "step_ms_stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "forward_ms": statistics.mean(p["forward_ms"] for _, p in samples),
        "backward_ms": statistics.mean(p["backward_ms"] for _, p in samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if device == "cuda"
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gpt2_42m.yaml")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--backend", choices=("manual", "sdpa", "both"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--loss-readback", choices=("per-step", "per-microbatch"), default="per-step"
    )
    parser.add_argument(
        "--output", type=Path, help="可选 JSON 输出路径，拒绝覆盖已有文件"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = replace(
        config.train,
        **{
            k: v
            for k, v in {
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "device": args.device,
            }.items()
            if v is not None
        },
    )
    model_cfg = replace(
        config.model,
        block_size=args.seq_len
        if args.seq_len is not None
        else config.model.block_size,
    )
    modes = MODES if args.mode == "all" else (args.mode,)
    backends = (
        ("manual", "sdpa")
        if args.backend == "both"
        else (args.backend or model_cfg.attention_backend,)
    )
    results = []
    for backend in backends:
        for mode in modes:
            result = benchmark(
                replace(model_cfg, attention_backend=backend),
                cfg,
                mode=mode,
                warmup=args.warmup,
                steps=args.steps,
                dtype=args.dtype,
                loss_readback=args.loss_readback,
            )
            results.append(result)
            print(
                f"{backend:6s} {mode:16s}: {result['tokens_per_s']:.0f} tok/s, "
                f"{result['step_ms_mean']:.2f} ms, peak={result['peak_allocated_bytes']}",
                flush=True,
            )
    payload = {
        "environment": environment_info(results[0]["device"]),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
