# 第二篇实验运行说明

实验代码已完成本地检查，并在 RTX 4090 上完成首轮 CUDA 实测，见 [实验记录](chapter02-rtx4090-20261025.md)。下文说明运行方法；FA3/FA4 的不可用项和 BF16 MoE 的路由敏感性在记录中单独列出，不把实验计划或未通过的检查当成成功结果。

## 要回答的问题

1. 第一篇教学实现里，逐 head 的线性投影和 Attention 调用、缓存拼接、Softmax 等操作，在时间线上是什么样的？
2. 合并多头计算和接入 FlashAttention 各自带来多少变化？不能把两步优化的收益都算到 FlashAttention 头上。
3. MLA 把历史状态压缩后，缓存确实小了，但重建 K/V 和矩阵吸收分别增加了什么工作？
4. MoE 每个 token 只走少数专家，路由、分发、专家计算与聚合又花了多少时间？

不把不同模型的短程 loss 或延迟当成同等模型能力下的比较，也不从单卡结果推断跨卡专家通信性能。

## 代码布局

- `tools/chapter02/config.py`：小规模数学检查配置，以及约 1.197 亿参数的 Dense 教学配置。
- `tools/chapter02/attention.py`：显式选择普通 Attention、SDPA 或 FA2/FA3/FA4，不允许自动回退。
- `tools/chapter02/model.py`：Dense、三条 MLA 路径、Top-k MoE，以及 MLA+MoE 组合。
- `tools/chapter02/cache.py`：`torch.cat` 和预分配缓存，分别记录逻辑容量与分配容量。
- `tools/chapter02/run.py`：环境检查、正确性检查、整模型计时、Attention 微基准、Profiler、文本训练。
- `tools/chapter02/suite.py`：按顺序执行检查和一组有界实验，保存每个命令、退出码与日志。
- `tests/test_chapter02*.py`：数学单元测试和 CLI 集成测试。

代码不依赖本机原 Notebook 才能运行。第一篇的 Notebook 不随仓库分发，也没有修改它。需要时可以额外传 `--notebook` 做权重映射检查。

## 模型与实验范围

`teaching` 配置的 Dense 模型保留第一篇的 6 层、768 隐藏维、12 个 head、2048 FFN 和 50257 词表，参数量仍是 119,690,496。第二篇将多头参数整理成完整线性层，并按后端选择逐 head 或整体执行。数学检查已覆盖与第一篇修正 RoPE 后的 Notebook 的同权重输出对齐。

这不是第一篇代码原封不动的重跑：缓存已统一为多头张量，RoPE 的构造方式也不同。要解释第一篇原始代码的性能，可以另外运行原有 `tools/validate_cuda_notebook.py`；不能直接把两台机器、两份实现的耗时差归因于 FlashAttention。

默认推理只对最后一个位置计算 LM Head；`--all-logits` 可恢复 Prefill 的全位置词表输出。训练始终计算全位置 logits。所有路径都应在相同口径下比较。

| 变体 | Attention | FFN | 缓存内容 |
| --- | --- | --- | --- |
| `dense` | 普通 MHA | Dense SwiGLU | 每层展开的 K/V |
| `mla-expanded` | MLA 数学结构 | Dense SwiGLU | 每个 head 展开的 K/V，包含复制的 RoPE key 分量 |
| `mla-compressed` | 同一套 MLA 参数 | Dense SwiGLU | 共享潜在状态与旋转后的 key 分量；每次重建历史 K/V |
| `mla-absorbed` | 同一套 MLA 参数 | Dense SwiGLU | 共享潜在状态与旋转后的 key 分量；将 key 投影吸收到 query 侧，将 value 投影移到聚合后 |
| `moe` | MHA | Top-k 路由专家 | 展开的 K/V |
| `mla-moe` | 吸收后的 MLA | Top-k 路由专家 | 压缩缓存 |

MLA 三条路径共用完全相同的参数名，可直接加载同一份权重；测试比较输出、输入梯度和参数梯度。实现保留解耦 RoPE 和低秩投影，但不是 DeepSeek 模型的完整复刻。

后续新增 `mla-reference-naive` 与 `mla-reference-absorb`，对齐固定版本 DeepSeek-V3 的合并 KV 上投影、FP32 Softmax 和共享压缩缓存路径；它们是独立于原三路径的新参数布局，只允许显式 `math` 后端，不会静默调用 FlashAttention。官方源码核对、真实语料各 300 步训练与推理计时见 [MLA 补充记录](mla-reference-20261101.md)。

MoE 默认 8 个路由专家、Top-2，每个专家中间维度 512，无共享专家、不丢 token、不设容量截断。接口支持一个共享专家，已在单元测试中验证。路由权重对选中专家重新归一化。训练可附加负载均衡项，默认系数 0.01。当前用逐专家 `where/gather/index_add`，不是生产级 grouped GEMM；其中 `where` 产生动态索引可能触发 CUDA 同步，这本身也是待观察的工程代价。专家负载统计在推理计时外另做一次，训练计时包含路由辅助损失。Dense 与 MoE 默认没有做总参数、激活参数或模型质量对齐，报告会分别列出参数口径。

本轮不实现专门的 FlashMLA kernel，不做专家并行或多卡 All-to-All，也不改第一篇已发布的数据。

## 准备远程环境

建议一张完整独占卡，避免 MIG 切片和其他任务干扰；确认 CUDA trace/CUPTI 权限。GPU 型号、驱动、PyTorch/CUDA 版本、可选库版本和源码 SHA256 会写入每次运行的 `environment.json`。

先安装与驱动相容的 CUDA 版 PyTorch，再在隔离环境里安装基础依赖：

```bash
python -m pip install -r requirements-chapter02.txt
python -m tools.chapter02.run doctor --output runs/ch02-environment
python -m unittest discover -s tests -v
```

`requirements-chapter02.txt` 是宽版本范围，不是远程实验的最终锁定环境。拿到服务器后再选定和验证具体版本，保存 `pip freeze`。不能用 CPU 版 PyTorch 跑 CUDA 实验。先不要为了安装某个后端覆盖服务器现有的工作环境。

按需要安装可选后端，不要求三者全部安装：

```bash
# FA2；需要可用的 nvcc、编译工具和相容的 CUDA/PyTorch。
MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation

# FA3；按官方 Hopper 子项目构建，实际运行时固定具体源码提交。
# git clone https://github.com/Dao-AILab/flash-attention.git /path/to/flash-attention
# cd /path/to/flash-attention/hopper
# MAX_JOBS=4 python -m pip install . --no-build-isolation

# FA4；CuTe DSL 版本，实际安装版本需与环境匹配。
python -m pip install --pre flash-attn-4
```

FA3 只走 Hopper 路径。FA4 有架构相关的实现和形状限制，不把“安装成功”当成“已经正确调用”。`sdpa-flash` 只允许 PyTorch Flash 后端；它的版本不应直接写成独立 `flash-attn` 包的版本。

为防止误用，本实验的 FA2/SDPA Flash 只接入 QK 与 V head dim 相同的输入；默认吸收后 MLA 的维度不满足这一限制，先用 `math` 做结构对照。FA3/FA4 在本实验中只启用 attention dropout=0 的路径。SDPA Flash 的缓存多 token chunk 掩码路径暂不接入；单 token Decode 和方形 Prefill 分别测试。被限制的组合会标记 `unavailable`，而不是换后端后继续计时。

## 第一步 做正确性检查

```bash
python -m tools.chapter02.run check \
  --device cuda --dtype bfloat16 \
  --backends math head-loop sdpa-math sdpa-flash fa2 fa3 fa4 \
  --output runs/ch02-cuda-check
```

检查包括方形 Prefill、单 token Decode、带历史前缀的 chunk、MHA/GQA 输入下的输出与 Q/K/V 梯度，以及完整模型的因果性、缓存一致性、MLA 三路径等价性。算子参考使用同一组低精度输入提升到 FP64 后的显式计算，容差写入结果。容差不是模型质量证明。

可选的第一篇 Notebook 对齐：

```bash
python -m tools.chapter02.run check \
  --device cpu --dtype float32 --backends math \
  --notebook /path/to/build_gpt.ipynb \
  --output runs/ch02-notebook-check --strict
```

没有通过正确性检查的后端不用于文章性能结论。`failed` 表示错误，需要排查；`unavailable` 表示没安装或明确不支持该组合，二者不能混为一谈。`--strict` 会将不可用组合也视为非零退出；包含上述有意限制的组合时，不要用它要求整张支持矩阵全部通过。

## 第二步 跑性能对照

先试小配置：

```bash
python -m tools.chapter02.suite --device cuda --quick \
  --backends head-loop math sdpa-flash fa2 fa3 fa4 \
  --require-backends math fa2 \
  --output runs/ch02-gpu-smoke
```

确认环境后，运行正式的有界矩阵：

```bash
python -m tools.chapter02.suite --device cuda \
  --backends head-loop math sdpa-flash fa2 fa3 fa4 \
  --require-backends math fa2 \
  --output runs/ch02-gpu-main
```

这条命令先检查后端，再执行长度 512/2048、batch 1/8 的 Prefill/Decode，对照 cat/static 缓存；另外做短的训练步骤计时和 Attention 微基准，最后采集几段 Profiler 时间线。训练计时用固定随机 token 验证执行与测量，不代表模型学到了语言规律。默认不启动真实语料训练、不下载数据。

H20/H100 环境想强制要求 FA3 时，把 `fa3` 加到 `--require-backends`。A800 不要要求 FA3；不安装的后端可以直接从 `--backends` 中去掉。

单独运行整模型对照：

```bash
python -m tools.chapter02.run bench \
  --device cuda --dtype bfloat16 --preset teaching \
  --context 512 --batch 1 \
  --variants dense --backends head-loop math sdpa-flash fa2 \
  --phases prefill decode --cache-modes cat static \
  --warmups 5 --repeats 20 --output runs/ch02-dense

python -m tools.chapter02.run bench \
  --device cuda --dtype bfloat16 --preset teaching \
  --context 2048 --batch 1 \
  --variants mla-expanded mla-compressed mla-absorbed moe mla-moe \
  --backends math --phases prefill decode \
  --output runs/ch02-structure
```

`context` 对 Prefill 表示输入长度，对 Decode 表示包含当前 token 的可见 KV 总长度。Decode 先在计时外建立 `context-1` 的前缀，测量时反复恢复到该逻辑长度，只计算一个新位置；不把不断增长的上下文混成一组样本。Prefill 包含新缓存的分配与写入。预分配缓存按容量分配，cat 缓存会复制历史，逻辑字节数与峰值显存不是一回事。

每个 case 从相同随机种子初始化。Dense 的各后端共用相同参数布局；MLA 的三种执行路径也如此。固定输入、精度、LM Head 输出范围和缓存策略后再比较。计时使用显式 CUDA 同步，保存全部样本、中位数、P10/P90、CUDA Event 跨度和峰值显存。Event 跨度仍可能包含 kernel 间空隙，不能用墙钟减 Event 就宣称得到了精确 CPU 开销。

## 第三步 采集细粒度 Profiling

```bash
python -m tools.chapter02.run profile \
  --device cuda --dtype bfloat16 --preset teaching \
  --variant dense --backend head-loop --phase decode \
  --cache-mode cat --context 512 --batch 1 \
  --warmups 3 --steps 3 --output runs/ch02-profile-dense
```

输出 `trace.json`、按输入形状聚合的 `operators.json` 和便于浏览的 `operators.txt`。标记覆盖 Q/K/V 投影、RoPE、缓存追加、QK/Softmax/PV、MLA 重建/吸收、专家分发和 LM Head；fused kernel 内部没有被重新拆开计时。

Profiler 与正式计时分开运行。开启 shape/memory 记录会影响时间，不拿 trace 中的整体耗时替代 benchmark；嵌套事件的 total time 不能直接相加成百分比。训练 profile 会预热优化器状态，再记录零梯度、前向、反向和更新。

需要进一步区分 host launch 与 GPU 空闲时，可以额外用 Nsight Systems 包裹一个短实验：

```bash
nsys profile --trace=cuda,osrt --output=/path/to/ch02-nsys \
  python -m tools.chapter02.run bench \
  --device cuda --dtype bfloat16 --preset teaching \
  --variants dense --backends math --phases decode --cache-modes static \
  --context 512 --batch 1 --warmups 2 --repeats 3 \
  --output runs/ch02-nsys-wrapped
```

这段同样不是正式延迟数据。Nsight Compute 的计数器访问需平台授权，本轮代码没有假定已获得权限，也不自动修改宿主机驱动设置。

## 可选 用同一份真实语料做短程训练

复用第一篇准备好的目录，包含 `train.bin`、`val.bin` 和 `manifest.json`，不从不同模型各自重新采一份语料。若换服务器，可迁移该数据目录；原文档级划分、分词器及来源说明见 [第一篇训练记录](seq-monkey-300.md)。训练前核对 manifest 中的 token 文件哈希与文档集合是否重叠。

```bash
python -m tools.chapter02.run train \
  --device cuda --dtype bfloat16 --preset teaching \
  --variant mla-moe --backend math \
  --data /path/to/seq-monkey-subset \
  --context 512 --batch 4 --steps 300 \
  --eval-every 25 --eval-blocks 192 \
  --output runs/ch02-mla-moe-text
```

分别换成 `dense`、`mla-absorbed`、`moe` 可检查每种结构的学习过程。训练从随机权重开始，不继承第一篇训练结果；默认 dropout=0，和第一篇 dropout=0.1 的训练不是严格复现。训练日志区分交叉熵与路由辅助损失，验证只报告交叉熵，在固定验证块上关闭梯度评估。支持 BF16 AMP，以及带 GradScaler 的 FP16；权重与 AdamW 状态保持 FP32。

`--save-checkpoint` 只在验证 loss 改善时保存最佳权重，不包含精确恢复训练所需的全部状态，不宣称可以无缝续训。模型、语料原文、token 文件、trace 和大体积日志都留在忽略的 `runs/` 或数据目录，不自动上传 GitHub。

## 本地已验证与尚未验证

在 Mac 的 CPU 上已跑通 31 项单元/集成测试，包含 22 项新增测试；用人工 token 夹具检查了训练入口，用 CPU Profiler 检查 trace 导出，用 quick suite 检查命令串联。独立检查入口跑通 39 条记录，其中包含第一篇 Notebook 的权重映射检查。CPU 环境 PyTorch 2.14.1、Python 3.12；完整本地环境见每个运行目录的记录。

首轮远程测试已完成 169 个整模型计时 case、15 个 Attention 微基准，以及 GPU Profiler 采集。FA2 和 PyTorch Flash 后端通过了当前算子输出/梯度检查；FA3 架构不支持，FA4 在现有 PyTorch 2.7 环境无法导入。完整参数规模下的 FP32 缓存检查通过，BF16 MoE 仍存在近边界 Top-k 敏感性，不能宣称低精度端到端检查全部通过。未来文章只引用注明范围的结果，不把缺失或失败的 case 填成零耗时。

补充诊断可用 `python -m tools.chapter02.check_precision --device cuda --dtype bfloat16 --output /path/to/new.json` 和 `python -m tools.chapter02.diagnose_router --output /path/to/diagnosis.json`。前者保留不满足容差的结果并返回非零退出码；后者会使用完整前向的专家 ID 做一次诊断重放，这不是可以用于正式计时或实际推理的优化。`router_float32` 默认为 False，以保留首轮测量的基线语义。

## 原始资料

- [FlashAttention 官方实现与安装说明](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4](https://arxiv.org/abs/2603.05451)
- [DeepSeek-V2 技术报告，MLA 与 DeepSeekMoE](https://arxiv.org/abs/2405.04434)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [第一篇代码起点与授权范围](../SOURCES.md)
