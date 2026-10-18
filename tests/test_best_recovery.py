from dataclasses import replace

import pytest
import torch

from learn_llm.checkpoint import load_checkpoint
from learn_llm.config import GPTConfig, TrainConfig
from learn_llm.train import train


def setup(token_data, tmp_path):
    model = GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16)
    cfg = TrainConfig(
        data_dir=str(token_data),
        out_dir=str(tmp_path / "source"),
        steps=1,
        batch_size=1,
        grad_accum=1,
        eval_every=1,
        eval_batches=1,
        device="cpu",
    )
    return model, cfg


def compare_weights(left, right):
    for key in left["model"]:
        torch.testing.assert_close(
            left["model"][key], right["model"][key], rtol=0, atol=0
        )


def test_best_write_failure_recovers_from_latest(token_data, tmp_path, monkeypatch):
    model, cfg = setup(token_data, tmp_path)
    score = {"loss": 2.0, "perplexity": 7.0}
    monkeypatch.setattr("learn_llm.train.evaluate", lambda *args: dict(score))
    train(model, cfg)
    save = torch.save

    def fail_best(payload, path):
        if path.name == "best.tmp":
            path.write_bytes(b"partial")
            raise OSError("best write failed")
        return save(payload, path)

    score["loss"] = 1.0
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_best)
        with pytest.raises(OSError, match="best write"):
            train(model, replace(cfg, steps=2), resume=True)
    true_best = load_checkpoint(tmp_path / "source/ckpt.pt")
    assert true_best["step"] == true_best["best_step"] == 2
    assert load_checkpoint(tmp_path / "source/best.pt")["best_val_loss"] == 2.0
    score["loss"] = 3.0
    train(model, replace(cfg, steps=3), resume=True)
    recovered = load_checkpoint(tmp_path / "source/best.pt")
    compare_weights(recovered, true_best)
    assert recovered["best_val_loss"] == 1.0
    assert load_checkpoint(tmp_path / "source/ckpt.pt")["best_step"] == 2


def test_resume_into_new_output_preserves_historical_best(
    token_data, tmp_path, monkeypatch
):
    model, cfg = setup(token_data, tmp_path)
    score = {"loss": 1.0, "perplexity": 3.0}
    monkeypatch.setattr("learn_llm.train.evaluate", lambda *args: dict(score))
    train(model, cfg)
    expected = load_checkpoint(tmp_path / "source/best.pt")
    score["loss"] = 2.0
    train(model, replace(cfg, steps=2), resume=True)
    train(
        model,
        replace(cfg, steps=3, out_dir=str(tmp_path / "new")),
        resume=tmp_path / "source/ckpt.pt",
    )
    actual = load_checkpoint(tmp_path / "new/best.pt")
    compare_weights(actual, expected)
    assert actual["step"] == 1
    # latest 本身不是历史最佳，且原 best 缺失时拒绝无依据迁移。
    (tmp_path / "source/best.pt").unlink()
    with pytest.raises(ValueError, match="历史 best"):
        train(
            model,
            replace(cfg, steps=3, out_dir=str(tmp_path / "missing")),
            resume=tmp_path / "source/ckpt.pt",
        )
