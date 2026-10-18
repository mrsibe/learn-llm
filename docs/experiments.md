# V1：GPT-2 → Llama 风格组件实验

## 范围

使用同一个 `GPT` / Pre-Norm Block，通过配置选择具体组件。只拆分
`attention.py`、`layers.py`，不引入模型注册表，不复制训练或生成系统。

V1 包含 RMSNorm、RoPE、SwiGLU、GQA；KV Cache、Triton/CUDA、量化和分布式属于后续工作。
所有模型从随机初始化开始，不提供旧 checkpoint 迁移或官方预训练权重导入。
目前只有正确性检查与 GPU 短测，**没有完成正式预训练，也没有架构效果结论**。

## 配置与实际参数量

所有配置均为 8 层、512 hidden、8 Query heads、1024 context、32768 vocab，
默认使用 SDPA 和 embedding/head 权重绑定。

| 文件（`configs/`） | 相对 GPT-2 的改动 | FFN hidden | KV heads | 参数量 |
| --- | --- | ---: | ---: | ---: |
| `gpt2_42m.yaml` | 基线 | 2048 | 8 | 42,521,600 |
| `gpt2_rmsnorm_42m.yaml` | 只替换归一化 | 2048 | 8 | 42,512,896 |
| `gpt2_rope_42m.yaml` | 只替换位置编码 | 2048 | 8 | 41,997,312 |
| `gpt2_swiglu_42m.yaml` | 只替换 FFN | 1360 | 8 | 42,461,440 |
| `gpt2_gqa_42m.yaml` | 只替换 Attention | 2048 | 2 | 39,369,728 |
| `llama_42m.yaml` | RMSNorm+RoPE+SwiGLU+GQA、无 bias | 1664 | 2 | 42,476,032 |

四个单组件实验都独立相对 GPT-2，不是累计改造。训练默认超参、数据路径和种子相同，
输出目录分别独立。文件名 `42m` 表示所属基线规模组，不表示参数量严格相等。
实际词表若小于 32K，模型参数量也会相应减少。

SwiGLU 有三组投影，不能直接沿用 GELU 的 `4*d`：

- GELU 的矩阵参数约为 `2*d*h`，SwiGLU 约为 `3*d*h`。
- 缺省 SwiGLU hidden 为约 `8/3*d`，向上对齐 64。
- 单组件实验显式设 1360，即接近 `8/3*512` 的 16 对齐值，参数量接近基线。
- 完整组合改用 1664，补偿 GQA 减少的参数。这是预算匹配的组合实验，**不是单变量消融**。

## 模块约定

| 配置字段 | 默认值 | 可选值 / 说明 |
| --- | --- | --- |
| `norm_type` | `layernorm` | `rmsnorm` |
| `position_type` | `learned` | `rope` |
| `mlp_type` | `gelu` | `swiglu` |
| `n_kv_heads` | `null` | null 表示等于 `n_head`；1 为 MQA；必须整除 Query heads |
| `ffn_hidden_size` | `null` | 缺省依 FFN 类型计算，也可显式指定 |
| `norm_eps` | `1e-5` | 两种 norm 的 epsilon |
| `rope_theta` | `10000` | RoPE 频率基数 |
| `bias` | `true` | 控制线性层及 LayerNorm；RMSNorm 和输出头始终无 bias |
| `tie_embeddings` | `true` | 可关闭输入/输出 embedding 绑定 |
| `attention_backend` | `manual` | `sdpa`；正式实验 YAML 使用 SDPA |

- RMSNorm 的低精度均方/归一化在 FP32 中计算，再返回输入 dtype。
- RoPE 对 Q/K 的相邻偶/奇维度旋转，不旋转 V，不创建绝对位置表。
  频率和低精度旋转均使用 FP32，最后转回输入 dtype；不维护低精度频率缓存。
- RoPE 是完整序列路径，位置从 0 开始。context 仍受 `block_size` 限制，不承诺长度外推。
- GQA 使用较小的 K/V 投影；Manual 后端按组显式展开 K/V，SDPA 使用 `enable_gqa`。
  由 PyTorch 根据设备/形状选择具体内核；不能保证 GQA 在小模型上更快。
- 没有实现 KV Cache，不把 KV head 数量的理论下降写成已经实现的缓存节省。
- 所有残差输出投影沿用按深度缩放的初始化。完整组合保留权重绑定，因此只是
  **Llama 风格**，不是 Meta 官方 Llama 配方或权重布局。

## 数据与可复现性

使用 TinyStories 官方 train/validation 切分。Tokenizer 只读取训练集，32K byte-level BPE。
每篇文档编码后加 EOT，连续写入小端 uint16；训练通过 memmap 取随机连续窗口，
目标右移一位。允许跨文档窗口，不 padding，不额外隔离文档间 Attention。

默认预处理缓存整个 Arrow 数据集；`--streaming --max-docs N` 按需读取，
但远程下载仍以分片/row group 为单位，不保证只传输 N 篇文档的精确字节数。
输出目录拒绝覆盖。`meta.json` 记录 token 数和 SHA256；`train --verify-data` 扫描核验。

Checkpoint 包含模型、优化器、配置、tokenizer、RNG、累计 tokens 与运行环境，
采用临时文件原子替换。续训必须使用相同架构/优化设置，可增加总步数和改变验证频率。
保存的 `metadata.json` 包含 Git commit/dirty 标记、依赖版本、设备和精度。

消融时固定 tokenizer、语料/顺序、优化器、有效 batch、种子和 token 预算。
同时记录参数量和吞吐，区分固定 tokens 与固定计算预算；不要直接比较不同 tokenizer 的原始 PPL。
正式比较应增加固定验证窗口数量，默认 20480 验证 tokens 只适合初步调试。

## 验证与短测

测试包括 RMSNorm/RoPE 的解析对照与 gradcheck、低精度 RoPE 对 FP32 旋转结果的精确 cast 对照、
SwiGLU 公式、32 种组件组合、GQA/MQA 的独立逐头参考、Manual/SDPA 输出与梯度、
两种完整架构的 tiny-batch 过拟合、训练/续训/生成和 Benchmark。
CUDA BF16 支持时额外检查完整训练/续训及 FP32/BF16 数值；无 GPU 时跳过。

V1 完整规模本机短测：RTX 4060 Laptop 8GB，42.476M 组合，BF16，batch=1、T=1024，
每场景 warmup=2、采样=5；完整训练步 grad_accum=2。

| 场景 | Manual tok/s | SDPA tok/s |
| --- | ---: | ---: |
| 全序列 Forward + loss（无梯度） | 54440 | 69644 |
| Forward + backward | 19918 | 31966 |
| 完整训练步 | 16961 | 26418 |

这是合成输入短测，不含数据 IO、验证或 checkpoint；未固定 GPU 频率/功耗，
不是稳定性能承诺，更不是训练后质量结果。实际配置和环境记录在本机被忽略的
`benchmarks/results/v1-llama42m.json`，本次未提交代码的 Git dirty 标记为 true。

```bash
uv run benchmark --config configs/llama_42m.yaml --mode all --backend both \
  --batch-size 1 --grad-accum 2 --warmup 2 --steps 5
```
