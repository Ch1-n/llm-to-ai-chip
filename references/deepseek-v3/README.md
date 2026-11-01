# DeepSeek-V3 MLA 参考来源

- 上游：[deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)。
- 固定 commit：`b15f0dbbbe6a4bc403306175698439ef380f5fb5`（2025-08-27）。
- 原始文件：[inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/b15f0dbbbe6a4bc403306175698439ef380f5fb5/inference/model.py)。
- `model.py` SHA256：`fdb7995bc234badbf0011b5af8cf391be94343abc51305428cb9a04452116f02`。
- 保留上游 `LICENSE-CODE`（MIT，Copyright (c) 2023 DeepSeek）；本地副本仅规范化了许可证文件末尾换行。

核对工具按 AST 提取所需的原始类与函数，不执行上游文件的顶层导入或整个模型。以单进程 `world_size=1`、未量化权重运行，沿用上游的 Linear、RMSNorm、RoPE 和 MLA；不使用 FP8 kernel，不加载 DeepSeek 权重，不验证张量并行或 YaRN 长上下文缩放。

`tools/chapter02/mla_reference.py` 是基于这些公式与实现改写的可训练小模型组件。改动包括：接入本项目配置和缓存接口、缓存关闭时保留完整梯度、动态生成普通 RoPE、显式因果 mask、可选输出 dropout。它不是 FlashMLA，也不代表完整 DeepSeek-V3 模型。
