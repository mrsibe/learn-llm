from dataclasses import replace

import pytest
import torch

from learn_llm.config import GPTConfig
from learn_llm.model import GPT


@pytest.mark.parametrize("length", [1, 3, 8])
def test_manual_sdpa_at_different_lengths(length):
    torch.manual_seed(12)
    cfg = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
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
