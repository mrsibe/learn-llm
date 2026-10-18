# learn-llm

手写 GPT-2 → Llama 风格模型，学习 AI Infra。模型、训练循环和采样自己实现，
底层复用 PyTorch 与 Hugging Face Datasets/Tokenizers，不使用 Transformers Trainer。

当前 V2 支持架构消融、AMP、固定计划的 Warmup/Cosine、latest/best 断点恢复、
固定验证窗口、FineWeb-Edu 数据准备和 PyTorch Profiler。暂不实现 KV Cache、分布式或自定义 CUDA Kernel。

## 使用

```bash
uv sync

# 少量数据调试
uv run python scripts/prepare_data.py --streaming --out data/debug --max-docs 2000
uv run train --data data/debug --out out/debug --steps 10 --verify-data
uv run generate --ckpt out/debug/best.pt --prompt "Once upon a time"

# TinyStories 正式配置：先 1K，再 5K，最后按原计划训练到 25K
uv run python scripts/prepare_data.py
uv run train --config configs/gpt2_42m_tinystories.yaml --steps 1000
uv run train --config configs/gpt2_42m_tinystories.yaml --resume --steps 5000
uv run train --config configs/gpt2_42m_tinystories.yaml --resume
uv run evaluate --ckpt out/gpt2-tinystories/best.pt --verify-data

# Llama 风格组合使用同一数据和训练预算
uv run train --config configs/llama_42m_tinystories.yaml
uv run benchmark --config configs/llama_42m.yaml --mode all --backend both
uv run tensorboard --logdir out
uv run pytest -q
uv run ruff check src scripts tests
```

默认 `uv run train` 仍为 1000 步调试配置，不会自动开始正式长训。
数据准备目录必须不存在；`--steps` 为累计训练步数，不会改变 `decay_steps`。
显存不足时首次运行可用 `--batch-size 1 --grad-accum 16`，保持每步 token 数。
`--profile-steps N` 先 warmup 一步，再记录 N 步，trace 写到输出目录的 `profile/`。

## 代码

- `model.py` / `attention.py` / `layers.py`：共享 Decoder 与架构组件。
- `train.py` / `optim.py` / `checkpoint.py` / `evaluate.py`：训练生命周期。
- `data.py` / `scripts/prepare_data.py`：分词、token 缓存和批处理。
- `generate.py` / `benchmark.py`：生成与独立性能测量。
- `configs/` / `tests/`：实验配置与 CPU/CUDA 检查。

详见 [V2 训练指南](docs/training.md)、[V1 架构实验](docs/experiments.md)、
[Benchmark 边界](benchmarks/README.md)。尚未完成正式预训练，不提供官方权重导入或旧版本权重迁移。
