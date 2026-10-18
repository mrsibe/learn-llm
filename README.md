# learn-llm

GPT-2 / AI Infra 学习基线。

当前只实现 **V0：单卡 GPT-2 风格预训练与生成**，不实现 Llama、GQA、KV Cache、DDP 或自定义 CUDA Kernel。
默认 8 层、512 hidden、8 heads、1024 context、32K byte-level BPE，约 42.5M 参数。
这是从随机初始化训练的缩小版本，不支持直接加载 OpenAI GPT-2 权重。

## 代码边界

- **自己写**：Transformer Block、因果 MHA、GELU MLP、模型组合、token packing/batching、训练循环和采样。
- **PyTorch**：Tensor 运算、Autograd、CrossEntropy、AdamW、AMP、可选 SDPA。
- **Hugging Face**：`datasets` 下载/读取，`tokenizers` 训练 BPE。
- **工程**：uv、YAML、TensorBoard、pytest、ruff；不依赖 Transformers/Trainer。

```text
configs/gpt2_42m.yaml       模型与训练配置
src/learn_llm/
  config.py                Dataclass 与配置校验
  model.py                 GPT-2 模型，包含 MHA、MLP 和 Block
  data.py                  token packing、memmap 与随机窗口
  train.py                 训练、验证、保存和恢复
  generate.py              无 KV Cache 的自回归生成
scripts/prepare_data.py    TinyStories + tokenizer 数据准备
tests/                     最小正确性与训练检查
```

目前不为了未来架构提前拆 `attention.py`、`layers.py`，也不引入注册表或通用框架。
数据准备脚本同时训练 tokenizer，无需额外运行第二个脚本。

## 运行

在仓库根目录执行：

```bash
uv sync
# 先少量文本跑通；注意 Datasets 会先下载/构建整个数据集的本地缓存。
uv run python scripts/prepare_data.py --out data/debug --max-docs 2000
uv run train --data data/debug --out out/debug --steps 10
uv run generate --ckpt out/debug/ckpt.pt --prompt "Once upon a time"

# 正式准备 TinyStories，再训练；不会自动启动长时间预训练。
uv run python scripts/prepare_data.py
uv run train --config configs/gpt2_42m.yaml
# steps 是总步数，不是额外步数；其他优化超参保持不变。
uv run train --resume --steps 2000
uv run generate

# loss 与累计训练 token 数
uv run tensorboard --logdir out

uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

也可使用 `uv run python -m learn_llm.train` / `uv run python -m learn_llm.generate`。
项目发行名为 `learn-llm`，Python 包名为 `learn_llm`（Python 导入名不能包含连字符）。
`uv sync --no-dev` 可跳过 pytest、ruff。配置路径与数据路径均相对于当前工作目录。

## 数据：为什么继续用 .bin

**Datasets 负责原始语料，.bin 负责训练 token 缓存，两者不冲突。**

1. Datasets 将原始语料下载并缓存为 Arrow，默认从磁盘映射，不整库装进 RAM。
2. 使用 TinyStories 官方 train/validation 切分；BPE 仅在训练集前 20000 篇文档上训练。
3. 每篇故事编码后添加 `<|endoftext|>`，连续写入 `train.bin` / `val.bin`。
4. token 为小端 uint16，每个占 2 字节。`meta.json` 记录 dtype、词表、token 数及内容 SHA256。
5. 训练使用 `np.memmap` 按需取窗口：输入 `tokens[i:i+T]`，目标右移一位。

窗口允许跨故事，EOT 作为分隔；不 padding，不对文档之间做 Attention 隔离。
预处理只运行一次，以后更换模型或反复实验不需要重新分词。
输出目录必须不存在，以免覆盖 tokenizer 后造成 token id 语义错配。
`--max-docs` 限制每个 split 编码的文档数，**不限制下载量**。
少量语料可能训不满 32K 词表，训练按实际大小对齐；此时模型参数量会减少。
不要直接比较不同 tokenizer 下的原始 loss/PPL。

## 模型、训练与恢复

`configs/gpt2_42m.yaml` 是唯一实验配置：

- Pre-LN、可学习位置编码、MHA、4× hidden GELU FFN、embedding/head 权重绑定。
- `attention_backend: manual` 默认保留可读的参考实现；设为 `sdpa` 使用 PyTorch 优化算子。
- 两种后端是同一个 GPT-2 架构；SDPA 不保证一定选择 FlashAttention，由设备/输入条件决定。
- CUDA 支持时用 AMP bf16，否则 fp32；不启用 fp16，所以不需要 GradScaler。
- 默认 batch=2、累积=8，每优化步处理 16384 tokens；常量学习率，梯度裁剪 1.0。
- 手写 Attention 保存完整注意力矩阵，显存不足时减小 batch；GPU 性能尚需实测。

验证时原子保存最新 `out/gpt2/ckpt.pt`，包含权重、优化器、随机数状态、累计 tokens、
实际模型/训练配置和 tokenizer；生成无需提供外部分词器文件。
同时写出实际 `config.yaml` 与 TensorBoard 日志。
续训校验模型、tokenizer、数据 metadata 与关键优化超参；允许增加总步数和改变验证频率。
恢复随机数流在同一设备与运行环境下用于复现实验，不承诺跨硬件逐位一致。
旧版根目录脚本的 checkpoint 缺少训练配置等字段，不支持直接续训。

模型权重和训练状态暂用 PyTorch `.pt` 一起保存；safetensors 导出、学习率调度、
Profiler、分布式、FineWeb-Edu 和架构改造都留待后续，不放进 V0。
