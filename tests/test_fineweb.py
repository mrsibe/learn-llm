import json

import numpy as np
import pytest
from datasets import Dataset
from tokenizers import Tokenizer

from learn_llm.data import DTYPE, EOT, TokenData, write_tokens
from scripts import prepare_data


def source():
    return Dataset.from_dict(
        {
            "text": [
                f"Story number {i}. A little cat made a friend. " * 3
                for i in range(100)
            ]
            * 2
        }
    )


def test_fineweb_partition_precedes_tokenizer_training(tmp_path, monkeypatch):
    dataset = source()
    calls = []

    def load(repo, **kwargs):
        calls.append((repo, kwargs))
        return {"train": dataset}

    monkeypatch.setattr(prepare_data, "load_dataset", load)
    real_train = prepare_data.train_tokenizer
    sampled = []

    def train(texts, vocab):
        sampled.extend(texts)
        assert all(not prepare_data.is_validation(text, 0.5, 42) for text in sampled)
        return real_train(sampled, vocab)

    monkeypatch.setattr(prepare_data, "train_tokenizer", train)
    out = tmp_path / "fineweb"
    prepare_data.prepare_data(
        out,
        vocab_size=300,
        tokenizer_docs=4,
        max_docs=6,
        dataset_name="fineweb-edu",
        val_fraction=0.5,
        revision="pinned",
    )
    assert len(sampled) == 4
    assert calls == [
        (
            "HuggingFaceFW/fineweb-edu",
            {"name": "sample-10BT", "streaming": False, "revision": "pinned"},
        )
    ]
    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
    partitions = {}
    for split in ("train", "val"):
        ids = np.fromfile(out / f"{split}.bin", dtype=DTYPE).tolist()
        partitions[split] = set(
            tokenizer.decode(ids, skip_special_tokens=False).split(EOT)
        ) - {""}
    assert partitions["train"] and partitions["val"]
    assert not partitions["train"] & partitions["val"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["subset"] == "sample-10BT" and meta["revision"] == "pinned"
    TokenData(out, 8, verify=True)


def test_reuse_tokenizer_and_exact_token_budgets(token_data, tmp_path, monkeypatch):
    dataset = source().to_iterable_dataset()
    monkeypatch.setattr(
        prepare_data, "load_dataset", lambda *args, **kwargs: {"train": dataset}
    )
    monkeypatch.setattr(
        prepare_data, "train_tokenizer", lambda *args: pytest.fail("应复用 tokenizer")
    )
    out = tmp_path / "fineweb"
    prepare_data.prepare_data(
        out,
        dataset_name="fineweb-edu",
        streaming=True,
        tokenizer_path=token_data / "tokenizer.json",
        val_fraction=0.5,
        train_tokens=256,
        val_tokens=128,
    )
    data = TokenData(out, 8, verify=True)
    assert len(data.splits["train"]) == 256 and len(data.splits["val"]) == 128
    assert (
        Tokenizer.from_str(data.tokenizer_json).get_vocab()
        == Tokenizer.from_file(str(token_data / "tokenizer.json")).get_vocab()
    )


def test_write_tokens_rejects_negative_budget(token_data, tmp_path):
    tokenizer = Tokenizer.from_file(str(token_data / "tokenizer.json"))
    with pytest.raises(ValueError):
        write_tokens(["abc"], tokenizer, tmp_path / "bad.bin", max_tokens=-1)
