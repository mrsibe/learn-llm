import math
from dataclasses import replace

import pytest
import torch

from learn_llm.attention import CausalSelfAttention
from learn_llm.config import GPTConfig
from learn_llm.model import GPT


@pytest.mark.parametrize("length", [1, 3, 8])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("position", ["learned", "rope"])
def test_manual_sdpa_at_different_lengths(length, kv_heads, position):
    torch.manual_seed(12)
    cfg = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_kv_heads=kv_heads,
        position_type=position,
    )
    manual, sdpa = GPT(cfg), GPT(replace(cfg, attention_backend="sdpa"))
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randint(32, (2, length))
    for model in (manual, sdpa):
        logits, loss = model(x, x)
        loss.backward()
        if length > 1:
            changed = x.clone()
            changed[:, -1] = (changed[:, -1] + 1) % cfg.vocab_size
            torch.testing.assert_close(logits[:, :-1], model(changed, x)[0][:, :-1])
    torch.testing.assert_close(manual(x, x)[0], sdpa(x, x)[0], rtol=1e-4, atol=1e-6)
    for left, right in zip(manual.parameters(), sdpa.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_gqa_matches_per_query_head_reference(backend, kv_heads):
    cfg = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        n_kv_heads=kv_heads,
        attention_backend=backend,
    )
    attention = CausalSelfAttention(cfg)
    x = torch.randn(2, 5, 16, requires_grad=True)
    q, k, v = attention.qkv(x).split((16, kv_heads * 4, kv_heads * 4), dim=-1)
    q = q.view(2, 5, 4, 4).transpose(1, 2)
    k, v = [t.view(2, 5, kv_heads, 4).transpose(1, 2) for t in (k, v)]
    heads = []
    # 独立参考：逐 Query head 显式选择所属 KV head，而不是 repeat_interleave。
    for head in range(4):
        kv = head // (4 // kv_heads)
        scores = q[:, head] @ k[:, kv].transpose(-2, -1) / math.sqrt(4)
        scores = scores.masked_fill(~attention.mask[:5, :5], float("-inf"))
        heads.append(scores.softmax(-1) @ v[:, kv])
    expected = attention.proj(torch.stack(heads, dim=2).reshape(2, 5, 16))
    actual = attention(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)
