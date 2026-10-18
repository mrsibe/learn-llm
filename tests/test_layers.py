from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from learn_llm.config import GPTConfig
from learn_llm.layers import GELUMLP, RMSNorm, RotaryEmbedding, SwiGLU


def test_rmsnorm_reference_and_gradient():
    torch.manual_seed(2)
    norm = RMSNorm(8).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    expected = (
        x / torch.sqrt(x.square().mean(-1, keepdim=True) + norm.eps) * norm.weight
    )
    torch.testing.assert_close(norm(x), expected)
    assert torch.autograd.gradcheck(norm, (x,))
    norm(x).sum().backward()
    assert norm.weight.grad is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_low_precision(dtype):
    norm = RMSNorm(8)
    x = torch.full((2, 8), 1000.0, dtype=dtype, requires_grad=True)
    out = norm(x)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones_like(out))
    out.float().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(norm.weight.grad).all()


def test_rope_formula_norm_and_relative_positions():
    torch.manual_seed(2)
    rope = RotaryEmbedding(8)
    q, k = torch.randn(2, 4, 5, 8), torch.randn(2, 2, 5, 8)
    rotated_q, rotated_k = rope(q, k)
    torch.testing.assert_close(rotated_q[..., 0, :], q[..., 0, :])
    torch.testing.assert_close(rotated_q.square().sum(-1), q.square().sum(-1))
    torch.testing.assert_close(rotated_k.square().sum(-1), k.square().sum(-1))
    # 第一个二维旋转平面的频率为 1，直接检查解析公式。
    position = torch.arange(5).view(1, 1, 5)
    expected = q[..., 0] * position.cos() - q[..., 1] * position.sin()
    torch.testing.assert_close(rotated_q[..., 0], expected)
    # 同时平移所有 Q/K 位置，内积矩阵只依赖相对位置。
    shift = 3
    padded_q = F.pad(q, (0, 0, shift, 0))
    padded_k = F.pad(k, (0, 0, shift, 0))
    shifted_q, shifted_k = rope(padded_q, padded_k)
    before = rotated_q[:, :2] @ rotated_k.transpose(-2, -1)
    after = shifted_q[:, :2, shift:] @ shifted_k[:, :, shift:].transpose(-2, -1)
    torch.testing.assert_close(before, after, rtol=1e-5, atol=2e-6)


def test_rope_dtype_and_gradient():
    # module.to 不应将频率计算永久降精度。
    rope = RotaryEmbedding(8).to(dtype=torch.bfloat16)
    q = torch.randn(1, 2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 1, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    out_q, out_k = rope(q, k)
    assert out_q.dtype == out_k.dtype == torch.bfloat16
    ref_q, ref_k = rope(q.float(), k.float())
    torch.testing.assert_close(out_q, ref_q.to(q.dtype), rtol=0, atol=0)
    torch.testing.assert_close(out_k, ref_k.to(k.dtype), rtol=0, atol=0)
    (out_q.float().sum() + out_k.float().sum()).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    q = torch.randn(1, 2, 3, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 3, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(rope, (q, k))


def test_swiglu_reference_and_ffn_budget():
    cfg = GPTConfig(
        n_embd=16, n_head=2, mlp_type="swiglu", ffn_hidden_size=48, bias=False
    )
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 3, 16, requires_grad=True)
    expected = F.linear(
        F.silu(F.linear(x, mlp.gate.weight)) * F.linear(x, mlp.up.weight),
        mlp.proj.weight,
    )
    torch.testing.assert_close(mlp(x), expected)
    mlp(x).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in mlp.parameters()
    )
    # 默认 SwiGLU 为约 8/3*d、向上对齐 64，而非直接使用 GELU 的 4*d。
    cfg = GPTConfig(bias=False)
    gelu = sum(p.numel() for p in GELUMLP(cfg).parameters())
    swiglu = sum(
        p.numel() for p in SwiGLU(replace(cfg, mlp_type="swiglu")).parameters()
    )
    assert abs(swiglu / gelu - 1) < 0.05
