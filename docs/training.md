# V2：训练工程化

V2 保留 V1 的模型与数据 `.bin` 格式，完善训练生命周期。不引入 Trainer、DDP、
KV Cache、Triton 或复杂的注册系统。实现了正式训练所需的入口和配置，
**并不表示已经完成 TinyStories/FineWeb-Edu 预训练或获得架构效果结论**。

## 预算与配置

默认 `configs/gpt2_42m.yaml` / `llama_42m.yaml` 仍是 1000 步、常量 LR 的调试配置。
新增四个正式配置：

- `gpt2_42m_tinystories.yaml` / `llama_42m_tinystories.yaml`
- `gpt2_42m_fineweb_edu.yaml` / `llama_42m_fineweb_edu.yaml`

模型保持 V1 的 42.52M GPT-2 与 42.48M Llama 风格组合；同一数据集上的两种模型使用相同训练参数：

| 参数                                 |                      值 |
| ------------------------------------ | ----------------------: |
| Context / micro batch / accumulation |            1024 / 2 / 8 |
| 总训练步数                           |                   25000 |
| 每步 tokens / 总处理 tokens          |       16384 / 409600000 |
| Peak LR / Min LR                     |             3e-4 / 3e-5 |
| Warmup / decay horizon               |             500 / 25000 |
| AdamW betas / weight decay / clip    | (0.9, 0.95) / 0.1 / 1.0 |
| 验证频率 / batch 数 / 固定窗口数     |          500 / 50 / 100 |
| 保存间隔                             |                     500 |

一次正式验证处理 102400 tokens。先进行 1K → 5K → 25K 的阶段检查，
验证曲线决定是否继续，不把达到预算等同于收敛。
`tokens_seen / meta.train_tokens` 是有放回随机窗口采样的 **token 等效 epochs**，不代表完整遍历。

显存不足可在首次运行前改成 batch=1、accum=16，保持有效 tokens/step。
为保证恢复语义，续训不允许修改 batch/accum；独立实验须同时记录吞吐差异。

## LR、AMP 与累计更新

- 使用 PyTorch AdamW、LambdaLR、autocast、GradScaler。矩阵参数参与 weight decay，bias/norm 不参与。
- 第一个成功更新使用 `lr / warmup_steps`；warmup 最后一次达到 peak LR。
- Cosine 在成功更新索引 `decay_steps - 1` 达到 min LR，之后维持 min LR。
- `decay_steps` 独立于运行的 `steps`，保存在配置和 checkpoint 中，续训不得修改。
  因此 `--steps 1000` 只是暂停 25K 计划，不会将整个 cosine 压缩到 1000 步。
- 从 25K 延长到 50K 会在 min LR 继续，并不会悄悄重算前半段曲线。
  如希望从头按 50K 衰减，应在开始实验前设置 `decay_steps: 50000`。
- `dtype: auto`：CUDA 优先 BF16；不支持时用 FP16；CPU 用 FP32。
  也可显式选择 `fp32/bf16/fp16`，CPU 不接受低精度训练模式。
- FP16 先 unscale，再裁剪；溢出由 GradScaler 跳过更新并降低 scale。
  **只有真正更新参数时才推进 scheduler**。FP32/BF16 的非有限梯度直接报错。
- 前向非有限 loss 在 optimizer 更新之前报错；loss 在累计结束后统一读回，不逐 micro-batch 同步。

`step` 是处理数据的训练尝试次数，`optimizer_updates` 是成功参数更新次数；
FP16 跳过更新仍计入处理 tokens，但不推进 LR。TensorBoard 同时记录 skipped_updates、
scale、实际 LR、梯度范数、loss、tokens、等效 epochs、step ms、tokens/s 和峰值显存。
计时不包含验证、日志及 checkpoint；冷启动/Profiler 开销仍会影响实测，不能拿短测直接估算长训耗时。

## 固定验证与 Checkpoint

启动时按 `eval_seed` 固定一组 token 窗口，保存在 checkpoint 中；评估不消耗训练 RNG。
`evaluate.py` 按窗口数加权平均 loss，支持不足一整 batch 的尾部，恢复调用前的 train/eval 模式。
续训可以改变验证频率，但不能改变验证窗口数/种子。不同 tokenizer 的原始 PPL 不可直接排名。

- `ckpt.pt`：按保存间隔、验证改善或训练结束保存最新完整状态。
- `best.pt`：只在 validation loss 严格改善时保存，相同 loss 保留更早最佳点。
- 两者都包含模型、优化器、scheduler、scaler、成功更新数、最佳记录、固定验证窗口、
  tokenizer、数据 metadata、Torch/CUDA/Python/NumPy RNG 和运行环境。
- 临时文件写入、fsync 后原子替换，失败清理临时文件，不破坏先前文件。
- latest 与 best 是两个独立原子文件，而不是跨文件事务。续训前会核对并恢复历史 best：
  latest 本身是最佳点时可以重建 best；迁移输出目录时复制原 best。
  历史权重无法恢复或属于其他实验时拒绝继续，避免只保留最佳 loss 而丢失权重。

```bash
uv run train --config configs/gpt2_42m_tinystories.yaml --resume --steps 5000
# 显式从最佳点恢复；模型与优化设置仍需与原实验匹配。
uv run train --config configs/gpt2_42m_tinystories.yaml --resume out/gpt2-tinystories/best.pt
uv run evaluate --ckpt out/gpt2-tinystories/best.pt --verify-data
uv run generate --ckpt out/gpt2-tinystories/best.pt
```

V2 不迁移旧版训练状态。CUDA 算子、硬件或依赖变化不保证逐位复现；固定环境下的 CPU
恢复测试同时检查权重、优化器、LR 状态、所有 RNG 与验证窗口。`perplexity` 超出浮点范围时为 null。

## FineWeb-Edu 数据准备

`--dataset fineweb-edu` 接入 `HuggingFaceFW/fineweb-edu` 的 `sample-10BT`。
其训练源没有官方验证 split，因此先按 `SHA256(seed + 文本)` 做内容级固定划分，默认 1% 留作验证。
相同文本始终属于同一集合；这不是近似重复检测，也不进行通用网页清洗或多语种过滤。
Tokenizer 只读取训练分区，或者用 `--tokenizer` 复用已有词表。

```bash
uv run python scripts/prepare_data.py --dataset fineweb-edu --streaming \
  --tokenizer data/tinystories/tokenizer.json \
  --train-tokens 500000000 --val-tokens 1000000 --out data/fineweb-edu
uv run train --config configs/gpt2_42m_fineweb_edu.yaml --verify-data
uv run train --config configs/llama_42m_fineweb_edu.yaml --verify-data
```

可省略 `--tokenizer`，在 FineWeb 训练分区训练新的 32K BPE；两种架构仍需共用该数据产物。
若需从原始远程数据复现，使用 `--revision <dataset-commit>` 固定数据版本；未指定时使用远程默认版本。
Metadata 记录来源、subset、revision、划分策略、预算、实际 token 数和内容 SHA256。

`--max-docs` 为每个输出分区的文档上限，`--train-tokens/--val-tokens` 是各自 token 上限，0 表示不限。
若足够数据，token 上限精确生效，可能截断最后一篇文档（末尾不保证保留 EOT）。
不保证数据源足够长；以实际 metadata 为准。空分区报错，失败不会发布半成品目录。

为保持实现简单，FineWeb 编码的 train/val 使用独立扫描；streaming 也会扫描源两遍，
默认 1% 验证比例可能需要扫描较多文档才能积累足够验证 tokens。
远程读取按分片/row group/编码 batch 发生，token 预算不等于精确的网络下载字节上限。
缓存模式会先下载/构建完整子集，不要用 `--max-docs` 误当下载限制。

## Profiling 与验证记录

```bash
uv run train --data data/debug --out out/profile --steps 4 --profile-steps 2
```

先 warmup 一步，再记录 N 步，只记录一次；完整窗口需要本次至少运行 N+1 步，
续训时也会重新 warmup。输出 `profile/*.pt.trace.json`，
可以用 Perfetto/Chrome Trace 打开。支持设备上同时记录 CPU/CUDA、shape 和内存，
阶段标签包括 data、forward、backward、optimizer。记录窗口期间可能包含发生的验证或保存，
因此它是生命周期诊断，不等同于排除这些干扰的 `uv run benchmark`。
trace 的可用性受 PyTorch/CUPTI/驱动支持影响，Profiler 会带来额外耗时和显存，不应在正式吞吐比较时开启。

本轮短程验证（不是正式预训练）：

- CPU 数学/状态测试，包括调度边界、固定计划续训、完整 RNG/优化器恢复、固定验证加权平均、
  latest/best 部分写入恢复与目录迁移、hash 分区、tokenizer 复用及 token 上限。
- CUDA BF16/FP16 短训及恢复、强制 FP16 梯度溢出跳过更新、真实 CUDA Profiler trace。
- 远程 FineWeb-Edu streaming：复用 300-token 调试词表，写出 256/128 个 train/val tokens，SHA256 校验通过。
- RTX 4060 Laptop 8GB 上，正式配置的 batch=2、accum=8、T=1024、BF16 完整训练步短测成功：
  GPT-2 峰值 allocated 约 1.66 GiB，Llama 风格约 1.85 GiB（warmup=1、采样=2）。
  此结果只证明短测未 OOM，不保证完整长训稳定，也不用于预测数小时训练耗时。
