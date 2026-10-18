# learn-llm

从手写 GPT-2 出发，学习模型架构与 AI Infra。当前支持 GPT-2 和 Llama 风格的
RMSNorm、RoPE、SwiGLU、GQA，可通过 YAML 做单组件消融。

模型、训练循环和采样自己实现；底层复用 PyTorch、Hugging Face Datasets/Tokenizers。
不使用 Transformers Trainer，暂不实现 KV Cache、分布式或自定义 CUDA Kernel。

## 使用

```bash
uv sync

# 少量数据调试：训练 BPE 并生成 train.bin / val.bin
uv run python scripts/prepare_data.py --streaming --out data/debug --max-docs 2000
uv run train --data data/debug --out out/debug --steps 10 --verify-data
uv run generate --ckpt out/debug/ckpt.pt --prompt "Once upon a time"

# 正式准备 TinyStories，使用默认 GPT-2 配置
uv run python scripts/prepare_data.py
uv run train --config configs/gpt2_42m.yaml
uv run train --resume --steps 2000

# Llama 风格组合：同样从随机初始化训练
uv run train --config configs/llama_42m.yaml
uv run generate --ckpt out/llama/ckpt.pt

uv run benchmark --config configs/llama_42m.yaml --mode all --backend both
uv run tensorboard --logdir out
uv run pytest -q
uv run ruff check src scripts tests
```

数据准备输出目录必须不存在；`--verify-data` 可选，启动时核验 token 文件 SHA256。
`--steps` 是总优化步数，续训需使用相同架构配置和优化超参。

## 结构

- `configs/`：GPT-2 基线、单组件消融和约 42M 的 Llama 风格组合。
- `src/learn_llm/model.py`：共享的 Decoder 骨架。
- `attention.py` / `layers.py`：MHA/GQA、归一化、RoPE、FFN。
- `data.py` / `train.py` / `generate.py`：数据、训练、生成。
- `benchmark.py`：独立吞吐、耗时和显存测量。
- `scripts/prepare_data.py`：TinyStories 数据准备。
- `tests/`：CPU/CUDA 正确性与训练检查。

架构参数与实验说明见 [docs/experiments.md](docs/experiments.md)，
Benchmark 边界见 [benchmarks/README.md](benchmarks/README.md)。
这些是学习用模型，不提供 OpenAI/Meta 官方权重导入或旧版本权重迁移；尚未完成正式预训练。
