"""HF 数据集 -> 训练/复用 BPE -> 有 token 预算的 train.bin / val.bin。"""

import argparse
import gc
import hashlib
import json
import tempfile
from contextlib import closing
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from learn_llm.data import DTYPE, EOT, write_tokens

REPO = "roneneldan/TinyStories"
SOURCES = {
    "tinystories": (REPO, None),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
}


def texts(dataset, limit=0):
    iterator = iter(dataset)
    try:
        for row in islice(iterator, limit) if limit else iterator:
            yield row["text"]
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def is_validation(text, fraction, seed):
    digest = hashlib.sha256(f"{seed}\0{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def partition_texts(dataset, split, limit, val_fraction, split_seed):
    """FineWeb 无官方验证集，按文档内容哈希划分；相同文本不会跨集合。"""
    count = 0
    with closing(texts(dataset)) as documents:
        for text in documents:
            if is_validation(text, val_fraction, split_seed) == (split == "val"):
                yield text
                count += 1
                if limit and count >= limit:
                    return


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
    out: Path,
    vocab_size=32768,
    tokenizer_docs=20000,
    max_docs=0,
    streaming=False,
    *,
    dataset_name="tinystories",
    tokenizer_path=None,
    val_fraction=0.01,
    split_seed=42,
    train_tokens=0,
    val_tokens=0,
    revision=None,
):
    if out.exists():
        raise FileExistsError(f"{out} 已存在，请换一个输出目录")
    if (
        not 257 <= vocab_size <= 65536
        or tokenizer_docs <= 0
        or min(max_docs, train_tokens, val_tokens, split_seed) < 0
    ):
        raise ValueError(
            "词表须为 257..65536；tokenizer-docs > 0；文档/token 预算与种子须 >= 0"
        )
    if dataset_name not in SOURCES or not 0 < val_fraction < 1:
        raise ValueError("未知数据集或 val-fraction 不在 (0, 1) 内")
    repo, subset = SOURCES[dataset_name]
    kwargs = {"streaming": streaming}
    if subset:
        kwargs["name"] = subset
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset(repo, **kwargs)

    def documents(split, limit=0):
        if dataset_name == "tinystories":
            return texts(dataset["train" if split == "train" else "validation"], limit)
        return partition_texts(dataset["train"], split, limit, val_fraction, split_seed)

    try:
        limit = min(tokenizer_docs, max_docs) if max_docs else tokenizer_docs
        if tokenizer_path:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            print("训练 tokenizer（只读取训练集）...", flush=True)
            with closing(documents("train", limit)) as source:
                tokenizer = train_tokenizer(source, vocab_size)
        if tokenizer.token_to_id(EOT) is None or tokenizer.get_vocab_size() > 65536:
            raise ValueError("tokenizer 必须包含 EOT，且词表适合 uint16")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent) as temp:
            work = Path(temp) / "prepared"
            work.mkdir()
            tokenizer.save(str(work / "tokenizer.json"))
            meta = {
                "dataset": repo,
                "subset": subset,
                "revision": revision,
                "dtype": DTYPE,
                "vocab_size": tokenizer.get_vocab_size(),
                "validation_split": "official"
                if dataset_name == "tinystories"
                else {
                    "method": "text_sha256",
                    "fraction": val_fraction,
                    "seed": split_seed,
                },
                "preparation": {
                    "streaming": streaming,
                    "max_docs": max_docs,
                    "tokenizer_docs": 0 if tokenizer_path else limit,
                    "reused_tokenizer": bool(tokenizer_path),
                    "train_token_limit": train_tokens,
                    "val_token_limit": val_tokens,
                },
            }
            # 两个独立迭代器：实现保持简单；FineWeb 的流式模式会扫描源数据两遍。
            for split, budget in (("train", train_tokens), ("val", val_tokens)):
                print(f"编码 {split}...", flush=True)
                with closing(documents(split, max_docs)) as source:
                    count, digest = write_tokens(
                        source, tokenizer, work / f"{split}.bin", max_tokens=budget
                    )
                if not count:
                    raise ValueError(f"{split} 为空，请增加文档预算或调整划分比例")
                meta[f"{split}_tokens"], meta[f"{split}_sha256"] = count, digest
                print(f"{split}: {count:,} tokens", flush=True)
            (work / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            work.rename(out)
    finally:
        if streaming:
            # 提前结束的 Arrow reader 可能持有引用环，须在解释器退出前释放。
            del dataset
            gc.collect()
    print(f"完成：{out}，实际词表 {tokenizer.get_vocab_size()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, default="tinystories")
    parser.add_argument("--out", default=None)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--tokenizer-docs", type=int, default=20000)
    parser.add_argument("--tokenizer", type=Path, help="复用已有 tokenizer.json")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument(
        "--train-tokens", type=int, default=0, help="0 为不限；预算可能截断最后一篇文档"
    )
    parser.add_argument("--val-tokens", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--revision", help="可选的 Hugging Face 数据集 commit/tag")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()
    prepare_data(
        Path(args.out or f"data/{args.dataset}"),
        args.vocab_size,
        args.tokenizer_docs,
        args.max_docs,
        args.streaming,
        dataset_name=args.dataset,
        tokenizer_path=args.tokenizer,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
