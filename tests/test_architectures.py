from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from learn_llm.config import GPTConfig, load_config
from learn_llm.model import GPT


@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("position", ["learned", "rope"])
@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
@pytest.mark.parametrize("kv_heads", [1, 4])
@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_component_combinations(norm, position, mlp, kv_heads, backend):
    cfg = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        norm_type=norm,
        position_type=position,
        mlp_type=mlp,
        n_kv_heads=kv_heads,
        attention_backend=backend,
        ffn_hidden_size=48,
        bias=False,
    )
    model = GPT(cfg)
    x = torch.randint(32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)
    assert (model.wpe is None) == (position == "rope")
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    torch.testing.assert_close(logits[:, :4], model(changed, x)[0][:, :4])
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert not any(name.endswith("bias") for name, _ in model.named_parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_kv_heads": 3},
        {"n_kv_heads": 16},
        {"n_kv_heads": 0},
        {"n_kv_heads": True},
        {"ffn_hidden_size": 0},
        {"ffn_hidden_size": 2.5},
        {"norm_type": "unknown"},
        {"position_type": "unknown"},
        {"mlp_type": "unknown"},
        {"norm_eps": 0},
        {"norm_eps": float("nan")},
        {"norm_eps": True},
        {"norm_eps": "1e-5"},
        {"rope_theta": -1},
        {"rope_theta": float("inf")},
        {"rope_theta": None},
        {"rope_theta": False},
        {"bias": "false"},
        {"tie_embeddings": 1},
        {"n_embd": 24, "n_head": 8, "position_type": "rope"},
    ],
)
def test_invalid_architecture_config(kwargs):
    with pytest.raises(ValueError):
        GPTConfig(**kwargs)


def test_weight_tying_option():
    cfg = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
    tied, untied = GPT(cfg), GPT(replace(cfg, tie_embeddings=False))
    assert tied.lm_head.weight is tied.wte.weight
    assert untied.lm_head.weight is not untied.wte.weight
    assert (
        sum(p.numel() for p in untied.parameters())
        - sum(p.numel() for p in tied.parameters())
        == 32 * 16
    )


def test_ablation_configs_and_parameters():
    baseline = load_config("configs/gpt2_42m.yaml")
    expected_changes = {
        "rmsnorm": {"norm_type"},
        "rope": {"position_type"},
        "swiglu": {"mlp_type", "ffn_hidden_size"},
        "gqa": {"n_kv_heads"},
    }
    base = asdict(baseline.model)
    with torch.device("meta"):
        baseline_params = sum(p.numel() for p in GPT(baseline.model).parameters())
    for component, fields in expected_changes.items():
        cfg = load_config(f"configs/gpt2_{component}_42m.yaml")
        changed = {
            key for key, value in asdict(cfg.model).items() if value != base[key]
        }
        assert changed == fields
        train, expected_train = asdict(cfg.train), asdict(baseline.train)
        train.pop("out_dir")
        expected_train.pop("out_dir")
        assert train == expected_train
    for path in Path("configs").glob("*42m.yaml"):
        cfg = load_config(path)
        with torch.device("meta"):
            model = GPT(cfg.model)
        params = sum(p.numel() for p in model.parameters())
        assert 38_000_000 < params < 43_000_000
        if path.name == "llama_42m.yaml":
            assert abs(params / baseline_params - 1) < 0.01


def test_llama_tiny_batch_overfit():
    torch.manual_seed(7)
    cfg = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        norm_type="rmsnorm",
        position_type="rope",
        mlp_type="swiglu",
        n_kv_heads=1,
        ffn_hidden_size=48,
        bias=False,
    )
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.arange(8).repeat(2, 1)
    initial = model(x, x + 1)[1].item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, x + 1)
        loss.backward()
        optimizer.step()
    assert model(x, x + 1)[1].item() < initial * 0.3
