# V0 Benchmark

## 边界

`uv run benchmark` 使用预先放到设备上的固定合成 token，不测数据 IO、下载、
TensorBoard、验证或 checkpoint。完整训练步包含梯度累积、loss 有限性检查、
梯度裁剪和 AdamW。CUDA 在样本边界同步；forward/backward 阶段使用 CUDA Events。

常用命令：

```bash
uv run benchmark --mode all --backend both --warmup 5 --steps 20 \
  --output benchmarks/results/attention.json
uv run benchmark --mode train_step --backend sdpa --batch-size 1 --grad-accum 8 \
  --loss-readback per-microbatch --output benchmarks/results/micro-readback.json
uv run benchmark --mode train_step --backend sdpa --batch-size 1 --grad-accum 8 \
  --loss-readback per-step --output benchmarks/results/step-readback.json
```

`per-microbatch` 重现旧循环每次累积的 isfinite Python 判断和 `.item()`，
`per-step` 每个优化步统一读取 loss，再在更新前检查。此开关只用于 Benchmark 诊断；
正式训练只保留 per-step 实现。

## 本机短测（2026-10-18）

- RTX 4060 Laptop 8GB / WSL2；Python 3.12.14，PyTorch 2.14.1，CUDA runtime 13.0。
- 42.52M GPT-2，32K vocab，1024 context，BF16 autocast，FP32 参数。
- Git 基点 `162303c`，**工作区有本次未提交改动**，不代表已发布版本。
- 完整训练步：batch=1，grad_accum=8，每步 8192 tokens。
- 每种后端按 micro → step → step → micro 顺序测量；每次重新初始化模型/优化器与种子，
  warmup 3 步，采样 8 步。表格以同类两轮的平均 step ms 换算 tokens/s。

| Backend | Loss 读回 | Step ms | Tokens/s | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: |
| Manual | 每 micro-batch | 363.35 | 22546 | 1.589 |
| Manual | 每优化步 | 354.96 | 23079 | 1.589 |
| SDPA | 每 micro-batch | 210.76 | 38869 | 1.200 |
| SDPA | 每优化步 | 190.67 | 42965 | 1.200 |

这些是短测观察，不是稳定的加速承诺。Manual 两种方式各轮耗时区间重叠，
SDPA 的 per-step 在这两轮里更快；不足以据此认定标量同步是最大瓶颈。
首次冷测曾明显慢于后续复测，因此**没有将首次/后续的巨大差异归因于代码优化**。
笔记本 GPU 的频率、温度、功耗与主机调度均未固定，正式比较需更多重复样本。

同机三场景检查（SDPA，batch=1，BF16，warmup=2，采样=5；完整步累积=2）：

| 场景 | 每轮 tokens | Wall ms | Tokens/s |
| --- | ---: | ---: | ---: |
| 全序列 forward + loss（无梯度） | 1024 | 8.46 | 121089 |
| Forward + backward | 1024 | 24.57 | 41675 |
| 完整训练步 | 2048 | 64.58 | 31711 |

不同场景的工作量不同，不要直接用表格推算训练/推理加速倍数。
原始本机 JSON 位于被 Git 忽略的 `benchmarks/results/`，记录实际模型、配置、
环境、各阶段耗时、step 分布和 allocated/reserved 显存。要分享结果请显式选择相应文件。

当前没有 MFU/GPU utilization 估计，也未执行完整 TinyStories 预训练。
