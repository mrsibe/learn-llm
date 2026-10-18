import json
from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

from learn_llm.checkpoint import load_checkpoint
from learn_llm.config import GPTConfig, TrainConfig, load_config
from learn_llm.data import TokenData
from learn_llm.evaluate import evaluate, validation_starts
from learn_llm.evaluate import main as evaluate_cli
from learn_llm.optim import Precision, learning_rate
from learn_llm.train import train


def small_configs(token_data, tmp_path, **kwargs):
    model = GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16)
    cfg = TrainConfig(
        data_dir=str(token_data),
        out_dir=str(tmp_path / "train"),
        steps=2,
        batch_size=2,
        grad_accum=2,
        eval_every=1,
        eval_batches=2,
        device="cpu",
        scheduler="cosine",
        warmup_steps=1,
        decay_steps=8,
        **kwargs,
    )
    return model, cfg


def test_warmup_cosine_boundaries_and_fixed_horizon():
    cfg = TrainConfig(
        lr=0.001, min_lr=0.0001, scheduler="cosine", warmup_steps=2, decay_steps=8
    )
    rates = [learning_rate(i, cfg) for i in range(12)]
    assert rates[0] == pytest.approx(0.0005)
    assert rates[1] == rates[2] == cfg.lr
    assert rates[7:] == [cfg.min_lr] * 5
    assert rates[2:8] == sorted(rates[2:8], reverse=True)
    assert rates == [learning_rate(i, replace(cfg, steps=50000)) for i in range(12)]
    assert learning_rate(0, replace(cfg, scheduler="constant")) == cfg.lr
    no_warmup = replace(cfg, warmup_steps=0)
    assert learning_rate(0, no_warmup) == cfg.lr
    assert learning_rate(7, no_warmup) == cfg.min_lr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheduler": "unknown"},
        {"scheduler": "cosine", "warmup_steps": 25000},
        {"scheduler": "cosine", "min_lr": 1},
        {"dtype": "unknown"},
        {"grad_clip": -1},
        {"beta1": 1},
        {"beta2": float("nan")},
        {"profile_steps": -1},
        {"save_every": 0},
        {"warmup_steps": True},
    ],
)
def test_training_config_validation(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_cpu_precision_and_explicit_errors():
    assert Precision("cpu", "auto").name == "fp32"
    assert not Precision("cpu", "fp32").scaler.is_enabled()
    for dtype in ("fp16", "bf16"):
        with pytest.raises(ValueError):
            Precision("cpu", dtype)


def test_fixed_validation_windows_weighted_mean(token_data):
    data = TokenData(token_data, 8)
    before = torch.get_rng_state()
    starts = validation_starts(data, 1, 5, 42)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)

    class Average(nn.Module):
        def forward(self, x, targets):
            return None, x[:, 0].float().mean() / 100

    model = Average().train()
    result = evaluate(model, data, starts, 2, Precision("cpu"))
    expected = data.batch_at("val", starts, "cpu")[0][:, 0].float().mean().item() / 100
    assert result["loss"] == pytest.approx(expected)
    assert result["tokens"] == 40
    assert model.training
    assert evaluate(model, data, starts, 1, Precision("cpu"))["loss"] == pytest.approx(
        result["loss"]
    )
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


def test_latest_best_and_resume_plan_validation(token_data, tmp_path, monkeypatch):
    model, cfg = small_configs(token_data, tmp_path)
    values = iter((1.0, 1.5))
    monkeypatch.setattr(
        "learn_llm.train.evaluate",
        lambda *args: {"loss": next(values), "perplexity": 3.0},
    )
    train(model, cfg)
    latest, best = (
        load_checkpoint(tmp_path / f"train/{name}.pt") for name in ("ckpt", "best")
    )
    assert latest["step"] == 2 and best["step"] == 1
    assert latest["best_val_loss"] == best["best_val_loss"] == 1.0
    assert latest["best_step"] == 1
    assert latest["scheduler"]["last_epoch"] == latest["optimizer_updates"] == 2
    assert latest["precision"] == "fp32" and latest["scaler"] == {}
    for field, value in (
        ("decay_steps", 9),
        ("warmup_steps", 0),
        ("min_lr", 0),
        ("scheduler", "constant"),
        ("eval_batches", 3),
        ("eval_seed", 43),
    ):
        with pytest.raises(ValueError, match=field):
            train(model, replace(cfg, steps=3, **{field: value}), resume=True)
    # best.pt 也是完整训练状态，可以显式恢复；不是仅权重快照。
    monkeypatch.setattr(
        "learn_llm.train.evaluate", lambda *args: {"loss": 0.5, "perplexity": 2.0}
    )
    train(model, replace(cfg, steps=3), resume=tmp_path / "train/best.pt")
    # 相同 loss 不覆盖更早的最佳点。
    assert load_checkpoint(tmp_path / "train/ckpt.pt")["best_step"] == 2


def test_profile_emits_trace(token_data, tmp_path):
    model, cfg = small_configs(token_data, tmp_path, profile_steps=1)
    train(model, cfg)
    traces = list((tmp_path / "train/profile").glob("*.pt.trace.json"))
    assert traces
    names = {
        event["name"] for event in json.loads(traces[0].read_text())["traceEvents"]
    }
    assert {"forward", "backward", "optimizer", "data"} <= names


@pytest.mark.parametrize("source", ["tinystories", "fineweb_edu"])
def test_formal_configs_match_training_budget(source):
    gpt = load_config(f"configs/gpt2_42m_{source}.yaml")
    llama = load_config(f"configs/llama_42m_{source}.yaml")
    assert (
        gpt.train.steps
        * gpt.train.batch_size
        * gpt.train.grad_accum
        * gpt.model.block_size
        == 409_600_000
    )
    left, right = asdict(gpt.train), asdict(llama.train)
    left.pop("out_dir")
    right.pop("out_dir")
    assert left == right
    assert gpt.train.scheduler == "cosine"
    assert gpt.train.decay_steps == 25000 and gpt.train.warmup_steps == 500
    assert gpt.train.eval_batches == 50 and gpt.train.eval_every == 500


def test_standalone_evaluate_uses_checkpoint_windows(
    token_data, tmp_path, monkeypatch, capsys
):
    model, cfg = small_configs(token_data, tmp_path)
    train(model, cfg)
    saved = load_checkpoint(tmp_path / "train/best.pt")
    capsys.readouterr()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--ckpt",
            str(tmp_path / "train/best.pt"),
            "--batch-size",
            "1",
            "--verify-data",
        ],
    )
    evaluate_cli()
    result = json.loads(capsys.readouterr().out)
    assert result["windows"] == cfg.eval_batches * cfg.batch_size
    assert result["loss"] == pytest.approx(saved["val_loss"], rel=1e-6)


def test_invalid_validation_windows_rejected_before_resume(token_data, tmp_path):
    model, cfg = small_configs(token_data, tmp_path)
    train(model, cfg)
    path = tmp_path / "train/ckpt.pt"
    state = load_checkpoint(path)
    state["validation_starts"][0] = -1
    torch.save(state, path)
    with pytest.raises(ValueError, match="验证窗口"):
        train(model, replace(cfg, steps=3), resume=True)


def test_incomplete_checkpoint_rejected(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"format_version": 2, "model": {}}, path)
    with pytest.raises(ValueError, match="不完整"):
        load_checkpoint(path)
