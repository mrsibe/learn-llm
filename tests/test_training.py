from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer

from learn_llm.config import GPTConfig, TrainConfig
from learn_llm.generate import main as generate
from learn_llm.train import train


def test_train_resume_matches_uninterrupted(token_data, tmp_path, monkeypatch, capsys):
    cfg = GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.1)
    training = TrainConfig(
        data_dir=str(token_data),
        out_dir=str(tmp_path / "resumed"),
        steps=1,
        batch_size=2,
        grad_accum=2,
        eval_every=1,
        eval_batches=1,
        device="cpu",
    )
    train(cfg, training)
    train(cfg, replace(training, steps=3), resume=True)
    train(cfg, replace(training, out_dir=str(tmp_path / "straight"), steps=3))
    resumed = torch.load(tmp_path / "resumed/ckpt.pt", weights_only=True)
    straight = torch.load(tmp_path / "straight/ckpt.pt", weights_only=True)
    assert resumed["step"] == 3
    assert resumed["tokens_seen"] == 3 * 2 * 2 * 8
    for name in resumed["model"]:
        torch.testing.assert_close(
            resumed["model"][name], straight["model"][name], rtol=0, atol=0
        )
    assert resumed["optimizer"]["param_groups"] == straight["optimizer"]["param_groups"]
    for key, state in resumed["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, straight["optimizer"]["state"][key][name], rtol=0, atol=0
            )
    torch.testing.assert_close(resumed["rng"], straight["rng"], rtol=0, atol=0)
    assert resumed["environment"]["packages"]["torch"]
    assert (tmp_path / "resumed/metadata.json").exists()
    assert list((tmp_path / "resumed/events").iterdir())
    with pytest.raises(ValueError, match="grad_accum"):
        train(cfg, replace(training, steps=4, grad_accum=1), resume=True)
    capsys.readouterr()
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate",
            "--ckpt",
            str(tmp_path / "resumed/ckpt.pt"),
            "--prompt",
            "Once",
            "--tokens",
            "2",
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    generate()
    assert "Once" in capsys.readouterr().out
    assert (
        Tokenizer.from_str(resumed["tokenizer_json"]).get_vocab_size()
        == resumed["config"]["vocab_size"]
    )
