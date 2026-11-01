# 第三篇参考代码与边界

主要来源为 DeepSeek-V4-Flash 官方参考实现：

- 仓库：[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)。
- 固定 revision：`60d8d70770c6776ff598c94bb586a859a38244f1`。
- [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py)，本地 `model.py` SHA256：`ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71`。
- [inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py)，本地 `kernel.py` SHA256：`59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`。
- 官方源码保持原样，随附 MIT `LICENSE`。
- [DeepSeek-V4 技术报告第 2.3 节](https://arxiv.org/html/2606.19348)说明 CSA、HCA、共享 K=V、滑动窗口、输出逆 RoPE 和 attention sink。

第二个独立参考是 [Transformers DeepSeek-V4 实现](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)，固定 commit `f90e93c4f83974f634860e246cdfd2c68666d605`。本地 `transformers-modeling.py` 保留 Apache 2.0 头及随附 `LICENSE-transformers`；仅规范化文件末尾换行，SHA256 为 `0375649d727a9fbea1756a500336af7d9ca66469e63c16bfabded25c85e5cebd`。

`tools/chapter03/reference.py` 只执行经过 SHA256 检查的指定 AST 定义，不导入整个模型，不下载权重。验证基础设施有以下明确适配，不把它称作完整官方精度/内核复现：

1. 关闭 FP8/FP4 量化模拟；省略同时作用于 Indexer Q/K 的正交 Hadamard 变换（实数点积不变）。
2. 用 Transformers 的 eager Attention 和显式选择掩码，替代 TileLang 稀疏 kernel，检查我们 gather 核心的输出及梯度。
3. 原始 Indexer 状态更新保持执行，但同分时按压缩块 ID 稳定排序。普通 top-k 不保证同分顺序；这一调整保证不同前缀长度下的同分选择一致，纯 PyTorch 参考使用 stable argsort，不代表生产级选择器性能。
4. 关闭 YaRN；单设备、未量化参数。仅核对 Attention、Compressor、Indexer，不验证 MoE、mHC、MTP 或 Muon。

项目实现是缩小维度的、可训练的 V4 衍生工作负载。CSA/HCA 压缩比分别为 4/128，其他维度和 Indexer top-k 缩小；使用普通 residual、Dense SwiGLU 和 AdamW。Indexer 添加了教学用 KL 蒸馏辅助目标，不声称复制官方完整训练策略。V4.1 不包含在本轮范围。
