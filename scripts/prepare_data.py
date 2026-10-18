"""用 HF Datasets 接入 TinyStories，用 HF Tokenizers 训练 byte-level BPE。"""

import argparse
import gc
import json
import tempfile
from contextlib import closing
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from learn_llm.data import DTYPE, EOT, write_tokens

REPO = "roneneldan/TinyStories"


def texts(dataset, limit=0):
    iterator = iter(dataset)
    try:
        rows = islice(iterator, limit) if limit else iterator
        for row in rows:
            yield row["text"]
    finally:
        # 提前截断 streaming 时主动释放底层 parquet/HTTP 迭代器。
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def train_tokenizer(documents, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        documents,
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return tokenizer


def prepare_data(
    out: Path, vocab_size=32768, tokenizer_docs=20000, max_docs=0, streaming=False
):
    if out.exists():
        raise FileExistsError(f"{out} 已存在，请换一个输出目录")
    if not 257 <= vocab_size <= 65536 or tokenizer_docs <= 0 or max_docs < 0:
        raise ValueError(
            "vocab-size 须为 257..65536；tokenizer-docs > 0；max-docs >= 0"
        )
    # 正式训练使用 Arrow 缓存；调试可按需读取远程分片，避免先缓存完整数据集。
    dataset = load_dataset(REPO, streaming=streaming)
    limit = min(tokenizer_docs, max_docs) if max_docs else tokenizer_docs
    print("训练 tokenizer（只读取训练集）...", flush=True)
    with closing(texts(dataset["train"], limit)) as documents:
        tokenizer = train_tokenizer(documents, vocab_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        work = Path(temp) / "prepared"
        work.mkdir()
        tokenizer.save(str(work / "tokenizer.json"))
        meta = {
            "dataset": REPO,
            "dtype": DTYPE,
            "vocab_size": tokenizer.get_vocab_size(),
            "preparation": {
                "streaming": streaming,
                "max_docs": max_docs,
                "tokenizer_docs": limit,
            },
        }
        for split, name in (("train", "train"), ("validation", "val")):
            print(f"编码 {name}...", flush=True)
            # 每次调用 texts 都产生新迭代器，不复用 tokenizer 已消费的训练流。
            with closing(texts(dataset[split], max_docs)) as documents:
                count, digest = write_tokens(documents, tokenizer, work / f"{name}.bin")
            meta[f"{name}_tokens"] = count
            meta[f"{name}_sha256"] = digest
            print(f"{name}: {count:,} tokens", flush=True)
        (work / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        work.rename(out)
    if streaming:
        # 提前结束的 Arrow reader 可能留有 Python 引用环。先回收它们，避免
        # 其后台 I/O 线程到解释器退出阶段才释放（Datasets/PyArrow 流式路径）。
        del dataset
        gc.collect()
    print(f"完成：{out}，实际词表 {tokenizer.get_vocab_size()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/tinystories")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--tokenizer-docs", type=int, default=20000)
    parser.add_argument(
        "--max-docs", type=int, default=0, help="每个 split 的编码文档上限；0 为全部"
    )
    parser.add_argument(
        "--streaming", action="store_true", help="按需读取，配合 --max-docs 快速调试"
    )
    args = parser.parse_args()
    try:
        prepare_data(
            Path(args.out),
            args.vocab_size,
            args.tokenizer_docs,
            args.max_docs,
            streaming=args.streaming,
        )
    finally:
        if args.streaming:
            gc.collect()


if __name__ == "__main__":
    main()
