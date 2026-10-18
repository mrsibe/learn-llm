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
  benchmark.py             独立合成输入性能测量
  metadata.py              Git、运行环境与设备信息
scripts/prepare_data.py    TinyStories + tokenizer 数据准备
tests/                     最小正确性与训练检查
```

目前不为了未来架构提前拆 `attention.py`、`layers.py`，也不引入注册表或通用框架。
数据准备脚本同时训练 tokenizer，无需额外运行第二个脚本。

## 运行

在仓库根目录执行：

```bash
uv sync
# 按需读取远程分片，先少量文本跑通。
uv run python scripts/prepare_data.py --streaming --out data/debug --max-docs 2000
uv run train --data data/debug --out out/debug --steps 10
uv run generate --ckpt out/debug/ckpt.pt --prompt "Once upon a time"

# 正式准备 TinyStories，再训练；不会自动启动长时间预训练。
uv run python scripts/prepare_data.py
uv run train --config configs/gpt2_42m.yaml --verify-data
# steps 是总步数，不是额外步数；其他优化超参保持不变。
uv run train --resume --steps 2000
uv run generate

# loss、累计 tokens、实际训练步耗时/吞吐/峰值显存
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

1. 默认用 Datasets 下载并缓存 Arrow；`--streaming` 则按需读取远程分片，适合配合 `--max-docs` 调试。两种方式都不整库装进 RAM。
2. 使用 TinyStories 官方 train/validation 切分；BPE 仅在训练集前 20000 篇文档上训练。
3. 每篇故事编码后添加 `<|endoftext|>`，连续写入 `train.bin` / `val.bin`。
4. token 为小端 uint16，每个占 2 字节。`meta.json` 记录 dtype、词表、token 数及内容 SHA256。
5. 训练使用 `np.memmap` 按需取窗口：输入 `tokens[i:i+T]`，目标右移一位。

窗口允许跨故事，EOT 作为分隔；不 padding，不对文档之间做 Attention 隔离。
预处理只运行一次，以后更换模型或反复实验不需要重新分词。
输出目录必须不存在，以免覆盖 tokenizer 后造成 token id 语义错配。
`--max-docs` 限制每个 split 编码的文档数；仅缓存模式会先准备完整数据集。
Streaming 仍会按分片/row group 请求数据，不保证下载量恰好等于这些文档的文本大小。
Tokenizer 训练、训练集编码和验证集编码各自创建迭代器，并显式关闭提前截断的流。
脚本在退出前回收 Arrow reader 的引用环，避免后台读取资源留到解释器退出时才释放。
少量语料可能训不满 32K 词表，训练按实际大小对齐；此时模型参数量会减少。
不要直接比较不同 tokenizer 下的原始 loss/PPL。

## 模型、训练与恢复

`configs/gpt2_42m.yaml` 是唯一实验配置：

- Pre-LN、可学习位置编码、MHA、4× hidden GELU FFN、embedding/head 权重绑定。
- 正式训练 YAML 默认 `attention_backend: sdpa`；`manual` 保留为数学参考与性能对照。
- 两种后端是同一个 GPT-2 架构；SDPA 不保证一定选择 FlashAttention，由设备/输入条件决定。
- CUDA 支持时用 AMP bf16，否则 fp32；不启用 fp16，所以不需要 GradScaler。
- 默认 batch=2、累积=8，每优化步处理 16384 tokens；常量学习率，梯度裁剪 1.0。
- 手写 Attention 保存完整注意力矩阵，显存不足时减小 batch。
- 梯度累积期间 loss 保留为设备 Tensor，每个优化步仅统一读回一次，在更新参数前检查有限性。
- 实际训练步的计时包含 batching、前后向、梯度检查与优化器；不包含 TensorBoard、验证和保存。
  CUDA 在步边界同步计时，首步包含冷启动开销，稳定吞吐请使用下方独立 Benchmark。

验证时原子保存最新 `out/gpt2/ckpt.pt`，包含权重、优化器、随机数状态、累计 tokens、
实际模型/训练配置和 tokenizer；生成无需提供外部分词器文件。
同时写出实际 `config.yaml`、环境 `metadata.json` 与 TensorBoard 日志。
Checkpoint 记录当前/初始运行环境（Git commit、dirty 标记、依赖版本、设备、精度等）。
续训校验模型、tokenizer、数据 metadata 与关键优化超参；允许增加总步数和改变验证频率。
`--verify-data` 会额外扫描 train/val.bin 并核验 SHA256，能发现文件长度不变的损坏；
默认只做格式/长度检查以避免每次启动扫描大型语料。Hash 与数据一起被篡改不在该检查的防护范围内。
恢复随机数流在同一设备与运行环境下用于复现实验，不承诺跨硬件逐位一致。
旧版根目录脚本的 checkpoint 缺少训练配置等字段，不支持直接续训。

模型权重和训练状态暂用 PyTorch `.pt` 一起保存；safetensors 导出、学习率调度、
Profiler、分布式、FineWeb-Edu 和架构改造都留待后续，不放进 V0。

## 独立 Benchmark

```bash
uv run benchmark --mode all --backend both --warmup 5 --steps 20 \
  --output benchmarks/results/attention.json
# 指定规模/有效 batch，所有参数与运行环境都会写入 JSON。
uv run benchmark --mode train_step --backend sdpa --batch-size 1 --grad-accum 8
# 仅用于诊断旧循环的逐 micro-batch 标量读回开销：
uv run benchmark --mode train_step --loss-readback per-microbatch
```

三种场景：`forward`（eval + no_grad，但计算全序列 logits 和 loss）、
`forward_backward`（一次前向+反向）、`train_step`（梯度累积+有限性检查+裁剪+AdamW）。
前两种按一个 micro-batch 计，完整训练步按配置的 `grad_accum` 计。
使用固定合成 token，排除数据下载、数据 IO、主机拷贝、日志与 checkpoint；
**合成吞吐不等于端到端训练吞吐**。先 warmup，CUDA 同步后用墙钟测量每轮耗时。

JSON 报告 mean/median/stdev step ms、tokens/s、forward/backward ms、峰值 allocated/reserved 显存，
以及模型、batch、精度、种子、环境信息。CUDA 阶段耗时用 Events，CPU 阶段耗时用墙钟；
它们不等同于完整 step 的耗时分解。显存为 warmup 后的总峰值，包含参数、梯度和优化器状态，
不是相对于空闲 GPU 的显存增量。CPU 显存指标为 null。
输出文件拒绝覆盖；`benchmarks/results/` 不纳入 Git。
目前不估算 MFU 或 GPU 利用率，避免用不确定的峰值算力冒充实测；这些留给 Profiler/Nsight。
本机短测和测量边界见 [benchmarks/README.md](benchmarks/README.md)。

## 检查与 CI

测试覆盖 CPU 前后向/因果性、不同长度的 Manual/SDPA 梯度一致性、微小 batch 过拟合、
完整优化器/RNG 状态恢复、等长数据损坏、配置不匹配、失败保存与非有限 loss。
支持 CUDA BF16 时额外执行完整训练/续训及 FP32/BF16 数值对照，无 GPU 时自动跳过。
普通测试不下载远程数据。

`.github/workflows/ci.yml` 在 push/PR 上运行 Ruff 和 CPU 测试，使用 CPU 版 PyTorch；
其余依赖从锁文件导出，避免在 CPU runner 下载 CUDA 库。GitHub 托管运行需在推送后确认。
正式消融仍应增加固定验证窗口数量，不要凭默认 20480 个验证 tokens 上的微小 loss 差异下结论。
