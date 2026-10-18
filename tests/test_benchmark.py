import math
from dataclasses import replace

import pytest

from learn_llm.benchmark import MODES, benchmark
from learn_llm.config import GPTConfig, TrainConfig


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("llama_style", [False, True])
def test_cpu_benchmark_scenarios(mode, llama_style):
    model = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
    if llama_style:
        model = replace(
            model,
            norm_type="rmsnorm",
            position_type="rope",
            mlp_type="swiglu",
            n_kv_heads=1,
            ffn_hidden_size=48,
            bias=False,
            attention_backend="sdpa",
        )
    cfg = TrainConfig(batch_size=2, grad_accum=2, device="cpu")
    result = benchmark(model, cfg, mode=mode, warmup=1, steps=2)
    accum = 2 if mode == "train_step" else 1
    assert result["tokens_per_iteration"] == 2 * 8 * accum
    assert result["effective_grad_accum"] == accum
    assert result["dtype"] == "fp32"
    assert result["peak_allocated_bytes"] is None
    assert math.isfinite(result["tokens_per_s"]) and result["tokens_per_s"] > 0
    assert result["step_ms_mean"] >= result["forward_ms"] > 0
    assert (result["backward_ms"] == 0) == (mode == "forward")
