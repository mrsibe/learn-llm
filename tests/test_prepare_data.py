import json

import pytest
from datasets import Dataset

from learn_llm.data import TokenData
from scripts import prepare_data


@pytest.mark.parametrize("streaming", [False, True])
def test_preparation_restarts_iterators(tmp_path, monkeypatch, streaming):
    calls = []
    reads = {"train": 0, "validation": 0}
    closed = {"train": 0, "validation": 0}

    class Documents:
        def __init__(self, split):
            self.split = split

        def __iter__(self):
            try:
                for _ in range(10):
                    reads[self.split] += 1
                    yield {"text": "A little cat played in the garden. " * 4}
            finally:
                closed[self.split] += 1

    def load(repo, **kwargs):
        calls.append(kwargs)
        return {split: Documents(split) for split in reads}

    monkeypatch.setattr(prepare_data, "load_dataset", load)
    out = tmp_path / "prepared"
    prepare_data.prepare_data(
        out, vocab_size=300, tokenizer_docs=2, max_docs=4, streaming=streaming
    )
    assert calls == [{"streaming": streaming}]
    assert reads == {"train": 6, "validation": 4}
    assert closed == {"train": 2, "validation": 1}
    TokenData(out, block_size=8, verify=True)


def test_real_datasets_streaming_matches_cached(tmp_path, monkeypatch):
    dataset = Dataset.from_dict({"text": ["One day a bird found a friend. " * 4] * 8})

    def load(repo, streaming=False):
        return {
            name: dataset.to_iterable_dataset() if streaming else dataset
            for name in ("train", "validation")
        }

    monkeypatch.setattr(prepare_data, "load_dataset", load)
    for streaming in (False, True):
        prepare_data.prepare_data(
            tmp_path / str(streaming),
            vocab_size=300,
            tokenizer_docs=4,
            max_docs=6,
            streaming=streaming,
        )
    for name in ("train.bin", "val.bin", "tokenizer.json"):
        assert (tmp_path / "False" / name).read_bytes() == (
            tmp_path / "True" / name
        ).read_bytes()
    meta = json.loads((tmp_path / "True/meta.json").read_text())
    assert meta["preparation"]["streaming"] is True


def test_preparation_failure_does_not_publish_partial_data(tmp_path, monkeypatch):
    dataset = Dataset.from_dict({"text": ["A little cat."] * 3})
    monkeypatch.setattr(
        prepare_data, "load_dataset", lambda *args, **kwargs: {"train": dataset}
    )
    out = tmp_path / "partial"
    with pytest.raises(KeyError, match="validation"):
        prepare_data.prepare_data(out, vocab_size=300)
    assert not out.exists()
