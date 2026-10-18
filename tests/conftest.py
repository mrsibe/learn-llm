import pytest
import torch
from datasets import Dataset

from scripts import prepare_data


@pytest.fixture(scope="session", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def token_data(tmp_path, monkeypatch):
    # 测试不下载远程语料，Datasets 和 Tokenizers 仍使用真实实现。
    texts = [
        "Once upon a time, a little cat found a friend. " * 4,
        "A dog and a bird played together in the garden. " * 4,
    ] * 8
    splits = {
        name: Dataset.from_dict({"text": texts}) for name in ("train", "validation")
    }
    monkeypatch.setattr(prepare_data, "load_dataset", lambda repo: splits)
    out = tmp_path / "data"
    prepare_data.prepare_data(out, vocab_size=300, tokenizer_docs=8, max_docs=8)
    return out
