from dataclasses import replace

import pytest
import torch

from learn_llm import GPT, GPTConfig
from learn_llm.config import load_config


def tiny_config(**kwargs):
    return GPTConfig(
        vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, **kwargs
    )


def test_shape_causality_and_backward():
    model = GPT(tiny_config()).eval()
    x = torch.randint(32, (2, 8))
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    logits, loss = model(x, x)
    other, _ = model(changed, x)
    assert logits.shape == (2, 8, 32)
    torch.testing.assert_close(logits[:, :4], other[:, :4])
    assert model.wte.weight is model.lm_head.weight
    assert model(x)[0].shape == (2, 1, 32)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_manual_sdpa_outputs_and_gradients():
    cfg = tiny_config()
    manual = GPT(cfg)
    sdpa = GPT(replace(cfg, attention_backend="sdpa"))
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randint(32, (2, 8))
    outputs = []
    for model in (manual, sdpa):
        logits, loss = model(x, x)
        outputs.append(logits)
        loss.backward()
    torch.testing.assert_close(*outputs, rtol=1e-4, atol=1e-6)
    for left, right in zip(manual.parameters(), sdpa.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
    # eval() 必须关闭 SDPA 内部 dropout，而不只是外围 Dropout 模块。
    model = GPT(tiny_config(dropout=0.5, attention_backend="sdpa")).eval()
    torch.testing.assert_close(model(x)[0], model(x)[0])


def test_default_size_and_config_validation():
    cfg = load_config("configs/gpt2_42m.yaml")
    with torch.device("meta"):
        model = GPT(cfg.model)
    assert 42_000_000 < sum(p.numel() for p in model.parameters()) < 43_000_000
    with pytest.raises(ValueError):
        GPTConfig(n_embd=17, n_head=2)
    with pytest.raises(ValueError):
        GPTConfig(attention_backend="unknown")


def test_tiny_batch_overfit():
    torch.manual_seed(7)
    model = GPT(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.arange(8).repeat(2, 1)
    y = x + 1
    initial = model(x, y)[1].item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.3
