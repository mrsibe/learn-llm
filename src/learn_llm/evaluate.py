"""固定窗口的 validation loss / perplexity，亦可独立评估 checkpoint。"""

import argparse
import json
import math

import torch

from .checkpoint import load_checkpoint
from .config import GPTConfig
from .data import TokenData
from .model import GPT
from .optim import Precision


def validation_starts(data, batch_size, batches, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        len(data.splits["val"]) - data.block_size,
        (batch_size * batches,),
        generator=generator,
    )


def validate_windows(data, starts):
    if (
        not isinstance(starts, torch.Tensor)
        or starts.device.type != "cpu"
        or starts.ndim != 1
        or starts.dtype != torch.int64
        or not starts.numel()
        or starts.min() < 0
        or starts.max() >= len(data.splits["val"]) - data.block_size
    ):
        raise ValueError("验证窗口索引无效")


@torch.no_grad()
def evaluate(model, data, starts, batch_size, precision):
    validate_windows(data, starts)
    training = model.training
    model.eval()
    total = torch.zeros((), device=precision.device)
    try:
        for chunk in starts.split(batch_size):
            x, y = data.batch_at("val", chunk, precision.device)
            with precision.autocast():
                _, loss = model(x, y)
            total += loss.detach() * len(chunk)
    finally:
        model.train(training)
    loss = (total / len(starts)).item()
    if not math.isfinite(loss):
        raise RuntimeError("验证 loss 非有限值，不保存 checkpoint")
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None
    return {
        "loss": loss,
        "perplexity": perplexity,
        "windows": len(starts),
        "tokens": len(starts) * data.block_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="out/gpt2/ckpt.pt")
    parser.add_argument("--data", default=None, help="默认使用 checkpoint 的数据路径")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--verify-data", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须 > 0")
    checkpoint = load_checkpoint(args.ckpt)
    cfg = GPTConfig(**checkpoint["config"])
    data = TokenData(
        args.data or checkpoint["train_config"]["data_dir"],
        cfg.block_size,
        verify=args.verify_data,
    )
    if (
        data.meta != checkpoint["data_meta"]
        or data.tokenizer_json != checkpoint["tokenizer_json"]
    ):
        raise ValueError("评估数据或 tokenizer 与 checkpoint 不一致")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = Precision(device)
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    result = evaluate(
        model, data, checkpoint["validation_starts"], args.batch_size, precision
    )
    print(json.dumps(result | {"precision": precision.name}, allow_nan=False))


if __name__ == "__main__":
    main()
