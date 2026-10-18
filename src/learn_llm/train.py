"""手写单卡训练循环：AdamW、AMP、梯度累积、验证和完整状态恢复。"""

import argparse
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .config import ExperimentConfig, GPTConfig, TrainConfig, load_config
from .data import TokenData
from .model import GPT


def train(model_cfg: GPTConfig, cfg: TrainConfig, resume=False):
    cfg.__post_init__()
    model_cfg.__post_init__()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=use_bf16
        )

    data = TokenData(cfg.data_dir, model_cfg.block_size)
    # 少量语料可能训不满 32K 词表，实际模型维度必须跟随 tokenizer。
    model_cfg = replace(model_cfg, vocab_size=data.vocab_size)
    out = Path(cfg.out_dir)
    path = out / "ckpt.pt"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} 已存在，请用 --resume 或更换输出目录")
    checkpoint = (
        torch.load(path, map_location="cpu", weights_only=True) if resume else None
    )
    if checkpoint:
        if checkpoint["config"] != asdict(model_cfg):
            raise ValueError("续训模型配置与 checkpoint 不一致")
        if (
            checkpoint["tokenizer_json"] != data.tokenizer_json
            or checkpoint["data_meta"] != data.meta
        ):
            raise ValueError("续训数据或 tokenizer 与 checkpoint 不一致")
        # 允许延长训练、改变输出位置/验证频率，不静默改变优化过程。
        for name in ("batch_size", "grad_accum", "lr", "weight_decay", "seed"):
            if checkpoint["train_config"][name] != getattr(cfg, name):
                raise ValueError(f"续训不能修改 {name}")
        if cfg.steps <= checkpoint["step"]:
            raise ValueError("续训 steps 必须大于 checkpoint 已完成步数")

    model = GPT(model_cfg).to(device)
    groups = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
    start, tokens_seen = 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, tokens_seen = checkpoint["step"], checkpoint["tokens_seen"]
        torch.set_rng_state(checkpoint["rng"])
        if device == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(asdict(ExperimentConfig(model_cfg, cfg)))
    )
    tokens_per_step = cfg.batch_size * cfg.grad_accum * model_cfg.block_size
    print(
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 参数 | {device} | {tokens_per_step:,} tokens/step"
    )

    with SummaryWriter(
        log_dir=str(out / "events"), purge_step=start if resume else None
    ) as writer:
        for step in range(start, cfg.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(cfg.grad_accum):
                x, y = data.batch("train", cfg.batch_size, device)
                with autocast():
                    _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"step {step + 1}: loss 非有限值")
                (loss / cfg.grad_accum).backward()
                train_loss += loss.detach().item() / cfg.grad_accum
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            tokens_seen += tokens_per_step
            writer.add_scalar("loss/train", train_loss, step + 1)
            writer.add_scalar("train/tokens_seen", tokens_seen, step + 1)
            if (step + 1) % 10 == 0:
                print(f"step {step + 1}: train loss {train_loss:.4f}", flush=True)

            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                model.eval()
                # 固定窗口，验证不扰动训练随机数流。
                generator = torch.Generator().manual_seed(42)
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(cfg.eval_batches):
                        x, y = data.batch("val", cfg.batch_size, device, generator)
                        with autocast():
                            _, loss = model(x, y)
                        val_loss += loss.item() / cfg.eval_batches
                if not math.isfinite(val_loss):
                    raise RuntimeError("验证 loss 非有限值，不保存 checkpoint")
                print(f"step {step + 1}: val loss {val_loss:.4f}", flush=True)
                writer.add_scalar("loss/val", val_loss, step + 1)
                writer.flush()
                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(model_cfg),
                    "train_config": asdict(cfg),
                    "data_meta": data.meta,
                    "tokenizer_json": data.tokenizer_json,
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "val_loss": val_loss,
                    "rng": torch.get_rng_state(),
                }
                if device == "cuda":
                    payload["cuda_rng"] = torch.cuda.get_rng_state()
                temporary = path.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gpt2_42m.yaml")
    parser.add_argument("--data", help="覆盖 train.data_dir")
    parser.add_argument("--out", help="覆盖 train.out_dir")
    parser.add_argument("--steps", type=int, help="覆盖总优化步数，续训也按总数算")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {
        name: value
        for name, value in {
            "data_dir": args.data,
            "out_dir": args.out,
            "steps": args.steps,
            "batch_size": args.batch_size,
        }.items()
        if value is not None
    }
    train(config.model, replace(config.train, **overrides), resume=args.resume)


if __name__ == "__main__":
    main()
