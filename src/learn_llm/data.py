"""自己实现 token packing / 随机窗口；存储与计算复用 NumPy、PyTorch。"""

import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

EOT = "<|endoftext|>"
DTYPE = "<u2"  # 小端 uint16；支持最多 65536 个 token


def write_tokens(texts, tokenizer: Tokenizer, path: Path, max_tokens=0):
    """文档末尾加 EOT，再连续拼接；不 padding，允许跨文档窗口。"""
    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("max_tokens 必须为非负整数")
    eot = tokenizer.token_to_id(EOT)
    if eot is None or tokenizer.get_vocab_size() > 65536:
        raise ValueError("tokenizer 必须包含 EOT，且词表不能超过 uint16 上限")
    total = 0
    digest = hashlib.sha256()
    texts = iter(texts)
    with path.open("wb") as f:
        while batch := list(islice(texts, 256)):
            ids = []
            for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                ids.extend(encoding.ids)
                ids.append(eot)
            if max_tokens:
                ids = ids[: max_tokens - total]
            block = np.asarray(ids, dtype=DTYPE).tobytes()
            f.write(block)
            digest.update(block)
            total += len(ids)
            if max_tokens and total >= max_tokens:
                break
    return total, digest.hexdigest()


class TokenData:
    def __init__(self, directory: str | Path, block_size: int, verify=False):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text())
        self.tokenizer_json = (directory / "tokenizer.json").read_text()
        tokenizer = Tokenizer.from_str(self.tokenizer_json)
        self.vocab_size = tokenizer.get_vocab_size()
        if self.meta["dtype"] != DTYPE or self.meta["vocab_size"] != self.vocab_size:
            raise ValueError("数据 dtype 或词表大小与 tokenizer 不一致")
        self.block_size = block_size
        self.splits = {}
        for split in ("train", "val"):
            path = directory / f"{split}.bin"
            size = path.stat().st_size
            if size % 2 or size < (block_size + 1) * 2:
                raise ValueError(
                    f"{path} 格式错误或太短，至少需要 {block_size + 1} tokens"
                )
            if size // 2 != self.meta[f"{split}_tokens"]:
                raise ValueError(f"{path} 的 token 数与 meta.json 不一致")
            if verify:
                expected = self.meta.get(f"{split}_sha256")
                if expected is None:
                    raise ValueError(f"meta.json 缺少 {split}_sha256，请重新准备数据")
                with path.open("rb") as f:
                    actual = hashlib.file_digest(f, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError(f"{path} SHA256 不一致，数据可能已损坏")
            self.splits[split] = np.memmap(path, mode="r", dtype=DTYPE)

    def batch(self, split: str, batch_size: int, device: str, generator=None):
        tokens = self.splits[split]
        starts = torch.randint(
            len(tokens) - self.block_size, (batch_size,), generator=generator
        )
        return self.batch_at(split, starts, device)

    def batch_at(self, split, starts, device):
        # 固定验证集合与随机训练窗口复用同一个 packing 语义。
        tokens = self.splits[split]
        windows = np.stack([tokens[i : i + self.block_size + 1] for i in starts])
        windows = torch.from_numpy(windows.astype(np.int64))
        if device == "cuda":
            windows = windows.pin_memory().to(device, non_blocking=True)
        else:
            windows = windows.to(device)
        return windows[:, :-1], windows[:, 1:]
