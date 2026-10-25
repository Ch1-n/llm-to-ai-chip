# LLM Workload Notes

从大模型推理工作负载理解 AI 芯片架构。

这里整理我学习模型结构、分析算子与数据访问的过程。先用简化的 Dense Decoder 建立工作负载基线，再逐步分析其他 Attention 结构。文章配图保留可编辑的 Draw.io 源文件，所有数值标明是理论估算还是实测。

## 文章

- [从芯片架构看大模型推理](articles/llm-inference-from-chip-architecture.md)：从 Token、Decoder 和 KV Cache，看到 Prefill/Decode 的矩阵形状、计算量和存储需求。

## 配套内容

- [工作负载估算脚本](tools/estimate_workload.py)：只使用 Python 标准库，不依赖 PyTorch 或 GPU。
- [估算数据](data/llm-inference/estimates.json)：包含模型配置、FLOPs 口径、容量假设和示例计算。
- [可编辑配图](articles/images/llm-inference/)：`.drawio` 和供 Markdown 展示的 PNG。
- [来源与范围](SOURCES.md)：开源学习起点、原始论文以及当前验证边界。
- [CUDA 测试记录](experiments/cuda-20261025.md)：RTX 4090 D 上的数值检查、推理基准与训练冒烟测试。

在仓库根目录运行：

```bash
python3 tools/estimate_workload.py
python3 -m unittest discover -s tests -v
```

当前文章中的 FLOPs、参数量和缓存容量为解析估算；CUDA 小节另外给出教学执行路径的实测，不代表 GPU 优化后的性能上限。尚未提供 DeepSeek-V4 Hybrid Attention 的复现。知乎 Word 保留在本地 `zhihu/` 目录，导出工具保留在 `.build/`，不上传仓库。

作者：Ch'in。
