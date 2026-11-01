# 从大模型到 AI 芯片

从大模型推理工作负载理解 AI 芯片架构。

这是一个配套知乎专栏的技术文章系列，先用三篇理解模型，后续再补充硬件分析。从应用场景出发，搭建小模型、跑通短程训练，再分析推理的计算形状、数据生命周期和存储流量，讨论它们如何影响 NPU 的架构选择。文章配图保留可编辑的 Draw.io 源文件，所有数值区分解析估算与实测。

## 系列目录

1. [从大模型到 AI 芯片（一）：搭建小模型，理解训练与推理](articles/llm-inference-from-chip-architecture.md)（初稿与实验已完成）：以 Dense Decoder、MHA 和 SwiGLU 建立基线，用中文语料跑通 300 步训练并记录训练与验证损失，讲解 KV Cache、Prefill/Decode、计算量与存储需求，附 CUDA 推理实测。
2. [从大模型到 AI 芯片 二 从性能分析到 MLA 与 MoE](articles/llm-profiling-mla-moe.md)（初稿与限定范围实验完成）：包含 Dense Profiling、FA2/SDPA Flash、官方对齐 MLA 和 MoE 的短程文本训练；训练后 BF16 MoE 的分块路由检查仍有未通过项。参见[首轮记录](experiments/chapter02-rtx4090-20261025.md)、[MLA 补充实验](experiments/mla-reference-20261101.md)、[MoE 补测](experiments/moe-text-20261101.md)与[运行说明](experiments/chapter02-runbook.md)。
3. [从大模型到 AI 芯片 三 Hybrid Attention 如何改变计算与访存](articles/llm-hybrid-attention.md)（初稿与限定范围实验完成）：固定 V4-Flash 衍生 SWA/CSA/HCA，含两组短训练、状态/计时、连续 Decode、消融和相同目标后缀的历史长度对照。训练后 Hybrid 的一组 BF16 长序列对照未全部满足容差；没有验证长上下文能力或完整 V4 模型。参见[首轮记录](experiments/chapter03-rtx4090-20261101.md)、[补测](experiments/chapter03-supplement-20261101.md)与[运行说明](experiments/chapter03-runbook.md)。

三篇采用同一条分析路径：应用场景 → 模型执行过程 → 算子形状 → 数据生命周期 → 芯片资源需求。MoE 与 Attention 设计是不同维度，可以组合；本系列的篇章顺序不代表它们彼此替代的模型演进顺序。

## 配套内容

- [工作负载估算脚本](tools/estimate_workload.py)：只使用 Python 标准库，不依赖 PyTorch 或 GPU。
- [估算数据](data/llm-inference/estimates.json)：包含模型配置、FLOPs 口径、容量假设和示例计算。
- [可编辑配图](articles/images/llm-inference/)：`.drawio` 和供 Markdown 展示的 PNG。
- [来源与范围](SOURCES.md)：开源学习起点、原始论文以及当前验证边界。
- [CUDA 测试记录](experiments/cuda-20261025.md)：RTX 4090 D 上的数值检查、推理基准与训练冒烟测试。
- [中文语料训练实验](experiments/seq-monkey-300.md)：Seq-Monkey 子集上的 300 步训练与验证 Loss 曲线，保留逐步日志及数据划分摘要。
- [FA2 与 SDPA Flash 训练曲线](experiments/seq-monkey-attention-backends-300.md)：同一初始模型和真实语料，各训练 300 步，附逐步日志、后端对照和本地归档说明。
- [第二篇实验代码与运行步骤](experiments/chapter02-runbook.md)：Dense/MLA/MoE、FA2/FA3/FA4 显式后端、缓存对照、Profiler 与短程文本训练入口。
- [第二篇 RTX 4090 实验记录](experiments/chapter02-rtx4090-20261025.md)：首轮计时、GPU 时间线、MLA 容量与速度对照，以及 BF16 Top-k 路由敏感性的边界。
- [MLA 官方对照与文本训练](experiments/mla-reference-20261101.md)：固定 DeepSeek 源码，核对 naive/absorb 路径，附两组真实语料 Loss 曲线和推理补充实测。
- [第三篇 Hybrid Attention 实验](experiments/chapter03-rtx4090-20261101.md)：SWA/Hybrid 训练曲线、缓存状态、Prefill/Decode 延迟与压缩块闭合开销，附[代码运行步骤](experiments/chapter03-runbook.md)。
- [第三篇补充实验](experiments/chapter03-supplement-20261101.md)：训练后长序列检查、3840 条连续 Decode 样本、28 项消融及有限长历史观察，保留未通过项。
- [MoE 文本训练](experiments/moe-text-20261101.md)：300 步学习曲线、专家分配热图和训练后真实文本的低精度路由诊断。

在仓库根目录运行：

```bash
python3 tools/estimate_workload.py
python3 -m unittest discover -s tests -v
```

第一篇中的 FLOPs、参数量和缓存容量为解析估算；CUDA 小节与后续实验记录另给出教学执行路径的实测，不代表 GPU 优化后的性能上限。第三篇实现的是 V4 衍生的小模型注意力组件，不是完整 DeepSeek-V4 的训练、质量或生产内核复现。知乎 Word 保留在本地 `zhihu/` 目录，导出工具保留在 `.build/`，不上传仓库。

作者：Ch'in。
