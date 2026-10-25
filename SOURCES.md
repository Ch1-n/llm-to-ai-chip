# 来源与验证范围

## 学习起点

- 原项目：[bbruceyuan/LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero)。
- 原作者：chaofa / bbruceyuan。
- 原项目许可证：Apache License 2.0。
- 本地学习副本的上游基线提交：`9eb4cb11e99a7925ba9ecd5a7c44d8c37092b49f`。
- 本地 `build_gpt.ipynb` 含尚未提交的改造；不能用上述基线提交代表改造后的实现。

本仓库不是原项目的镜像或完整衍生发行版，没有复制 Notebook、数据集、模型权重或作者的配图。文章说明学习来源，估算脚本与插图为本篇单独编写。后续若发布改造后的原项目代码，需要一并保留适用的原始许可证和通知，标注改动。

## 数值口径

正文中的参数量、FLOPs 与容量由 `tools/estimate_workload.py` 生成，数据保存在 `data/llm-inference/`。六个标准库测试检查矩阵形状计数、参数量、LM Head、缓存容量、batch 缩放与非法配置。

主要矩阵 FLOPs 按一次乘法和一次加法为 2 FLOPs，不包含归一化、激活、Softmax、RoPE、bias、残差和采样。Prefill 的 Attention 按完整 dense 矩阵乘法计数，不声称等于跳过因果上三角后的实际执行 FLOPs。

权重存储以每元素 2 字节作假设，不代表已经执行 FP16/BF16 测试。容量不包含 allocator padding、分页元数据、临时工作区和量化元数据。32 层、head dim 128 的 MHA/GQA 缓存图是独立容量算例，不是 512-token 教学模型的长上下文运行结果。

## 当前代码审查状态

2026-10-25 对本地教学 Notebook 的 CPU/FP32 小配置做过完整前向与缓存推理、分块 Prefill 及部分边界检查。审查发现的 RoPE 频率排列、配置派生和生成边界问题尚未在原文件中自动修改；修正 RoPE 的验证仅在临时测试的内存副本中进行。

因此，本仓库文章按正确的数学定义解释结构与工作负载，不将当前 Notebook 声明为完整验证的参考实现。没有上传那份尚在修改的 Notebook。

## CUDA 验证与后续实验

2026-10-25 已在 RTX 4090 D 上完成 CUDA 检查：在测试进程内修正 RoPE、显式设置小配置的 head_size，检查因果性、逐 token/分块缓存一致性与上下文边界。完整 1.197 亿参数配置运行了 FP32/BF16/FP16 的缓存检查、24 组推理基准，以及每种精度 3 步的训练冒烟测试。低精度训练使用 FP32 权重和 AMP，FP16 使用 GradScaler。原始 Notebook 未被修改，结果见 `experiments/cuda-20261025.md`。

推理采用 eager、逐 head 循环、显式 score 和动态 cat 缓存，未进行 kernel 优化。测试使用随机权重和 token，没有下载训练集，不代表生成质量、收敛性或 GPU 优化性能上限。原有 generate 的额外前向和边界行为已记录，未宣称完整生成接口通过。尚未完成 MPS、完整训练或 DeepSeek-V4 复现。

后续应先整理原实现，再用 profiler 分析小算子和调度开销，并在数学等价的前提下比较融合多头、预分配缓存、优化 Attention kernel 等执行路径。每次保留模型指纹、环境、shape、精度、预热和同步方法，分别报告 Prefill、Decode 和显存。
