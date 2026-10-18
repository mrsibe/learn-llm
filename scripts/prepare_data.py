"""用 HF Datasets 接入 TinyStories，用 HF Tokenizers 训练 byte-level BPE。"""

import argparse
import json
import tempfile
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from learn_llm.data import DTYPE, EOT, write_tokens

REPO = "roneneldan/TinyStories"


def texts(dataset, limit=0):
    rows = islice(dataset, limit) if limit else iter(dataset)
    return (row["text"] for row in rows)


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


def prepare_data(out: Path, vocab_size=32768, tokenizer_docs=20000, max_docs=0):
    if out.exists():
        raise FileExistsError(f"{out} 已存在，请换一个输出目录")
    if not 257 <= vocab_size <= 65536 or tokenizer_docs <= 0 or max_docs < 0:
        raise ValueError(
            "vocab-size 须为 257..65536；tokenizer-docs > 0；max-docs >= 0"
        )
    # 使用 Datasets 默认 Arrow 磁盘缓存/内存映射，不把完整语料加载到 RAM。
    # 这里不用在线 streaming：下载/缓存一次，之后多次训练复用本地 token 文件。
    dataset = load_dataset(REPO)
    limit = min(tokenizer_docs, max_docs) if max_docs else tokenizer_docs
    print("训练 tokenizer（只读取训练集）...", flush=True)
    tokenizer = train_tokenizer(texts(dataset["train"], limit), vocab_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        work = Path(temp) / "prepared"
        work.mkdir()
        tokenizer.save(str(work / "tokenizer.json"))
        meta = {
            "dataset": REPO,
            "dtype": DTYPE,
            "vocab_size": tokenizer.get_vocab_size(),
        }
        for split, name in (("train", "train"), ("validation", "val")):
            print(f"编码 {name}...", flush=True)
            count, digest = write_tokens(
                texts(dataset[split], max_docs), tokenizer, work / f"{name}.bin"
            )
            meta[f"{name}_tokens"] = count
            meta[f"{name}_sha256"] = digest
            print(f"{name}: {count:,} tokens", flush=True)
        (work / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        work.rename(out)
    print(f"完成：{out}，实际词表 {tokenizer.get_vocab_size()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/tinystories")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--tokenizer-docs", type=int, default=20000)
    parser.add_argument(
        "--max-docs", type=int, default=0, help="每个 split 的编码文档上限；0 为全部"
    )
    args = parser.parse_args()
    prepare_data(Path(args.out), args.vocab_size, args.tokenizer_docs, args.max_docs)


if __name__ == "__main__":
    main()
