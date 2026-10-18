from dataclasses import replace

import numpy as np
import pytest
import torch

from learn_llm.config import GPTConfig, TrainConfig
from learn_llm.data import DTYPE, TokenData
from learn_llm.model import GPT
from learn_llm.train import train


def configs(token_data, tmp_path):
    return (
        GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16),
        TrainConfig(
            data_dir=str(token_data),
            out_dir=str(tmp_path / "out"),
            steps=1,
            batch_size=1,
            grad_accum=2,
            eval_every=1,
            eval_batches=1,
            device="cpu",
        ),
    )


@pytest.mark.parametrize("split", ["train", "val"])
def test_same_length_corruption_requires_hash_check(token_data, tmp_path, split):
    TokenData(token_data, block_size=8, verify=True)
    path = token_data / f"{split}.bin"
    token = np.fromfile(path, dtype=DTYPE, count=1)[0]
    with path.open("r+b") as f:
        f.write(np.asarray([(int(token) + 1) % 200], dtype=DTYPE).tobytes())
    # 默认长度检查不能发现等长修改；显式 verify 才扫描文件内容。
    TokenData(token_data, block_size=8)
    with pytest.raises(ValueError, match="SHA256"):
        TokenData(token_data, block_size=8, verify=True)
    model, cfg = configs(token_data, tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        train(model, cfg, verify_data=True)
    assert not (tmp_path / "out").exists()


def test_resume_rejects_model_and_tokenizer_mismatch(token_data, tmp_path):
    model, cfg = configs(token_data, tmp_path)
    train(model, cfg, verify_data=True)
    resumed = replace(cfg, steps=2)
    with pytest.raises(ValueError, match="模型配置"):
        train(replace(model, n_layer=2), resumed, resume=True)
    # 合法 JSON、相同词表，但 checkpoint 保存的 tokenizer JSON 不再一致。
    path = token_data / "tokenizer.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="tokenizer"):
        train(model, resumed, resume=True)


def test_failed_checkpoint_write_preserves_previous_file(
    token_data, tmp_path, monkeypatch
):
    model, cfg = configs(token_data, tmp_path)
    train(model, cfg)
    path = tmp_path / "out/ckpt.pt"
    before = path.read_bytes()

    def fail_save(payload, temporary):
        temporary.write_bytes(b"partial checkpoint")
        raise OSError("simulated full disk")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="full disk"):
        train(model, replace(cfg, steps=2), resume=True)
    assert path.read_bytes() == before


def test_nonfinite_loss_stops_before_optimizer_update(
    token_data, tmp_path, monkeypatch
):
    model, cfg = configs(token_data, tmp_path)
    forward = GPT.forward

    def nonfinite(self, *args):
        logits, loss = forward(self, *args)
        return logits, loss * float("nan")

    updates = []
    monkeypatch.setattr(GPT, "forward", nonfinite)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self: updates.append(True))
    with pytest.raises(RuntimeError, match="loss 非有限值"):
        train(model, cfg)
    assert updates == []
    assert not (tmp_path / "out/ckpt.pt").exists()
