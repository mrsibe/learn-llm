from dataclasses import replace

import pytest
import torch

from learn_llm.config import GPTConfig, TrainConfig
from learn_llm.model import GPT
from learn_llm.optim import Precision, build_optimizer, build_scheduler, optimizer_step
from learn_llm.train import train

CUDA_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pytestmark = pytest.mark.skipif(not CUDA_BF16, reason="需要支持 BF16 的 CUDA GPU")


@pytest.mark.parametrize("llama_style", [False, True])
def test_cuda_bf16_training_and_resume(token_data, tmp_path, llama_style):
    model = GPTConfig(
        block_size=16, n_layer=1, n_head=2, n_embd=32, attention_backend="sdpa"
    )
    if llama_style:
        model = replace(
            model,
            norm_type="rmsnorm",
            position_type="rope",
            mlp_type="swiglu",
            n_kv_heads=1,
            ffn_hidden_size=88,
            bias=False,
        )
    cfg = TrainConfig(
        data_dir=str(token_data),
        out_dir=str(tmp_path / "gpu"),
        steps=1,
        batch_size=2,
        grad_accum=2,
        eval_every=1,
        eval_batches=2,
        device="cuda",
    )
    train(model, cfg, verify_data=True)
    train(model, replace(cfg, steps=2), resume=True, verify_data=True)
    checkpoint = torch.load(tmp_path / "gpu/ckpt.pt", weights_only=True)
    assert checkpoint["step"] == 2
    assert checkpoint["data_verified"] is True
    assert checkpoint["environment"]["dtype"] == "bf16"
    assert "cuda_rng" in checkpoint
    assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())


@pytest.mark.parametrize("llama_style", [False, True])
def test_cuda_fp32_bf16_manual_sdpa_numerics(monkeypatch, llama_style):
    # 关闭 TF32，避免把 TF32/FP32 的差异混入 BF16 对照。
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(12)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    if llama_style:
        cfg = replace(
            cfg,
            norm_type="rmsnorm",
            position_type="rope",
            mlp_type="swiglu",
            n_kv_heads=1,
            ffn_hidden_size=88,
            bias=False,
        )
    reference = GPT(cfg)
    state = reference.state_dict()
    x = torch.randint(64, (2, 16), device="cuda")
    y = torch.randint(64, (2, 16), device="cuda")
    results = {}
    for backend in ("manual", "sdpa"):
        for bf16 in (False, True):
            model = GPT(replace(cfg, attention_backend=backend)).cuda()
            model.load_state_dict(state)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                logits, loss = model(x, y)
            loss.backward()
            grads = torch.cat([p.grad.flatten() for p in model.parameters()])
            assert torch.isfinite(grads).all()
            results[backend, bf16] = logits.detach().float(), grads.detach()
    ref_logits, ref_grads = results["manual", False]
    for (backend, bf16), (logits, grads) in results.items():
        if bf16:
            # BF16 只有 7 位尾数；以 FP32 参考检查输出和整体梯度相对范数。
            torch.testing.assert_close(logits, ref_logits, rtol=0.05, atol=0.003)
            relative_error = (grads - ref_grads).norm() / ref_grads.norm()
            assert relative_error < 0.05, (backend, relative_error.item())
        else:
            torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(grads, ref_grads, rtol=1e-4, atol=1e-6)


def test_fp16_scaler_overflow_skips_optimizer_and_scheduler():
    model = torch.nn.Linear(4, 4).cuda()
    cfg = TrainConfig(scheduler="cosine", warmup_steps=2, decay_steps=8)
    precision = Precision("cuda", "fp16")
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    x = torch.randn(2, 4, device="cuda")
    with precision.autocast():
        loss = model(x).square().mean()
    precision.scaler.scale(loss).backward()
    original = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad.fill_(float("inf"))
    scale = precision.scaler.get_scale()
    updated, norm = optimizer_step(model, optimizer, precision, cfg.grad_clip)
    if updated:
        scheduler.step()
    assert not updated and not torch.isfinite(torch.tensor(norm))
    assert precision.scaler.get_scale() < scale
    assert scheduler.last_epoch == 0
    for p, expected in zip(model.parameters(), original):
        torch.testing.assert_close(p, expected, rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    with precision.autocast():
        loss = model(x).square().mean()
    precision.scaler.scale(loss).backward()
    updated, _ = optimizer_step(model, optimizer, precision, cfg.grad_clip)
    if updated:
        scheduler.step()
    assert updated and scheduler.last_epoch == 1


def test_cuda_fp16_training_resume(token_data, tmp_path):
    model = GPTConfig(
        block_size=16, n_layer=1, n_head=2, n_embd=32, attention_backend="sdpa"
    )
    cfg = TrainConfig(
        data_dir=str(token_data),
        out_dir=str(tmp_path / "fp16"),
        steps=1,
        batch_size=2,
        grad_accum=2,
        eval_every=1,
        eval_batches=1,
        device="cuda",
        dtype="fp16",
        scheduler="cosine",
        warmup_steps=1,
        decay_steps=8,
    )
    train(model, cfg)
    train(model, replace(cfg, steps=2), resume=True)
    checkpoint = torch.load(tmp_path / "fp16/ckpt.pt", weights_only=True)
    assert checkpoint["step"] == 2
    assert checkpoint["precision"] == "fp16" and checkpoint["scaler"]
    assert checkpoint["scheduler"]["last_epoch"] == checkpoint["optimizer_updates"]
    assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
