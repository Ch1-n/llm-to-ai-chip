# 从大模型到 AI 芯片

从大模型推理工作负载理解 AI 芯片架构。

这是一个配套知乎专栏的技术文章系列，先用三篇理解模型，后续再补充硬件分析。从应用场景出发，搭建小模型、跑通短程训练，再分析推理的计算形状、数据生命周期和存储流量，讨论它们如何影响 NPU 的架构选择。文章配图保留可编辑的 Draw.io 源文件，所有数值区分解析估算与实测。

## 系列目录

1. [从大模型到 AI 芯片（一）：搭建小模型，理解训练与推理](articles/llm-inference-from-chip-architecture.md)（初稿与实验已完成）：以 Dense Decoder、MHA 和 SwiGLU 建立基线，用中文语料跑通 300 步训练并记录训练与验证损失，讲解 KV Cache、Prefill/Decode、计算量与存储需求，附 CUDA 推理实测。
2. **MoE 与 MLA 如何改变大模型工作负载**（计划中）：分别分析专家路由与 FFN 计算、KV 状态压缩与 Decode 路径，讨论激活参数、专家权重访问、负载均衡和矩阵吸收的取舍。标题暂定，正文与实验尚未完成。
3. **Hybrid Attention 下的计算与数据访问**（计划中）：固定具体模型版本与官方配置，分析不同注意力路径、压缩与选择操作，以及稀疏计算和不规则访存如何映射到芯片。标题暂定，正文与实验尚未完成。

三篇采用同一条分析路径：应用场景 → 模型执行过程 → 算子形状 → 数据生命周期 → 芯片资源需求。MoE 与 Attention 设计是不同维度，可以组合；本系列的篇章顺序不代表它们彼此替代的模型演进顺序。

## 配套内容

- [工作负载估算脚本](tools/estimate_workload.py)：只使用 Python 标准库，不依赖 PyTorch 或 GPU。
- [估算数据](data/llm-inference/estimates.json)：包含模型配置、FLOPs 口径、容量假设和示例计算。
- [可编辑配图](articles/images/llm-inference/)：`.drawio` 和供 Markdown 展示的 PNG。
- [来源与范围](SOURCES.md)：开源学习起点、原始论文以及当前验证边界。
- [CUDA 测试记录](experiments/cuda-20261025.md)：RTX 4090 D 上的数值检查、推理基准与训练冒烟测试。
- [中文语料训练实验](experiments/seq-monkey-300.md)：Seq-Monkey 子集上的 300 步训练与验证 Loss 曲线，保留逐步日志及数据划分摘要。

在仓库根目录运行：

```bash
python3 tools/estimate_workload.py
python3 -m unittest discover -s tests -v
```

当前文章中的 FLOPs、参数量和缓存容量为解析估算；CUDA 小节另外给出教学执行路径的实测，不代表 GPU 优化后的性能上限。尚未提供 DeepSeek-V4 Hybrid Attention 的复现。知乎 Word 保留在本地 `zhihu/` 目录，导出工具保留在 `.build/`，不上传仓库。

作者：Ch'in。
