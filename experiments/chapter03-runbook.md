# 第三篇实验：V4 衍生 Hybrid Attention

目标不是在 4090 上加载完整 DeepSeek-V4，而是把 SWA、CSA、HCA 的计算与状态变化做成能核对、能训练的小模型，再观察它对 NPU 工作负载意味着什么。

## 固定来源

- [DeepSeek-V4 官方报告，第 2.3 节](https://arxiv.org/html/2606.19348)。
- [DeepSeek-V4-Flash 官方代码](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py)，revision `60d8d70770c6776ff598c94bb586a859a38244f1`。
- [Transformers 的 eager Attention](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)，commit `f90e93c4f83974f634860e246cdfd2c68666d605`。
- 原始代码、许可证与验证适配说明保留在 [references/deepseek-v4](../references/deepseek-v4/)。不引用名称相似、实际上仍是 V3 MLA 的非官方仓库。

## 实现范围

缩小模型的主干沿用前两篇的宽度、层数、词表和 Dense SwiGLU，使用普通 residual 和 AdamW。注意力则实现共享 K=V、低秩 Q 与分组低秩 O 投影、逐 head Query 归一化、末尾通道 RoPE 与输出逆 RoPE、attention sink。

Hybrid 的 6 层压缩比为 `0,0,4,128,4,128`。`0` 是纯 SWA；`4` 是重叠压缩和 Indexer 选择；`128` 是非重叠重压缩、不做稀疏选择。滑动窗口为 128，教学版 Indexer top-k 为 8，不是官方生产配置的 512。隐藏维 768、12 heads、head dim=64、RoPE dim=16、Q rank=128、O groups=4、组内 rank=64、Indexer 4 heads×32。

本轮不实现 FP8/FP4、YaRN、正交旋转量化、MoE、mHC、MTP 或 Muon，不编译生产级 FlashMLA/TileLang 算子，也不涉及 V4.1。实验可支持至 8192 token 的缩小配置，不宣称复现百万上下文能力。

### 对照变体

| 变体 | 用途 |
| --- | --- |
| `mha` | 前两篇的普通多头 Attention 数学实现，随机权重 |
| `mla` | 第二篇官方结构对齐的吸收路径，随机权重 |
| `shared-dense` | 新投影与 K=V 结构，但读取完整历史；不进行窗口截断或压缩 |
| `swa` | 所有层只保留滑动窗口 |
| `hybrid` | SWA、CSA、HCA 按层混合 |

`shared-dense` 能帮助区分 MQA/K=V 自身的状态缩减，与窗口/压缩产生的额外缩减。跨结构没有做参数量或质量匹配，不能拿延迟或短程 Loss 排模型优劣。

## 正确性

```bash
python -m tools.chapter03.run check --device cuda --preset teaching \
  --output runs/ch03-check-new
python -m unittest discover -s tests -v
```

验证包括官方类的未量化 Attention 输出和压缩缓存、独立 Transformers 密集核心的输出与 Q/KV/sink 梯度、未来 token 扰动、压缩块未完成时的可见性、跨 4/128 边界的分块 Prefill、整段与增量推理。

原始 Indexer 的普通 top-k 不保证同分顺序。项目以稳定排序选较小的块 ID，避免补上更多被掩码的未来槽位后，同分选择发生变化；官方类对照也明确适配这一条。它使用的是纯 PyTorch stable argsort，不能代表优化后的 DeepSelect/top-k 性能。

硬件计时仅在验证完成后运行。不同注意力结构的输出本来就不一样，检查各自的因果、缓存和参考公式，不要求 Hybrid 与 MHA 数值相同。

## 真实文本训练

```bash
python -m tools.chapter03.run train --device cuda --dtype bfloat16 \
  --preset teaching --variant hybrid --data /path/to/seq-monkey-subset \
  --batch 12 --context 512 --steps 300 --eval-every 25 --eval-blocks 192 \
  --output runs/ch03-train-hybrid-new
```

另将 variant 换成 `swa`，在新目录训练一组局部注意力对照。数据划分、验证块和训练顺序与前两篇一致。FP32 权重与 AdamW 状态、BF16 autocast，压缩投影、门控池化和 Softmax 使用 FP32；无 dropout。曲线展示每步交叉熵与固定验证集交叉熵。

硬 top-k 不可微，不能指望交叉熵自然训练好 Indexer。教学模型用一个额外 KL 目标：以主注意力对所有已完成压缩块的概率为教师（detach），监督 Indexer 分布，权重 0.01。训练日志分别记录交叉熵、辅助损失和 Indexer 梯度；验证只报告交叉熵。这是明确的实验训练策略，不是对官方完整训练配方的复现。

最佳 checkpoint 只含权重、配置、step 和验证值，不含优化器/RNG 状态，不能精确断点续训。

## 性能、状态与时间线

```bash
python -m tools.chapter03.run bench --device cuda --dtype bfloat16 \
  --variants mha mla shared-dense swa hybrid --contexts 512 2048 8192 \
  --batches 1 4 --warmups 2 --repeats 5 --output runs/ch03-bench-new
python -m tools.chapter03.run bench --device cuda --dtype bfloat16 \
  --variants swa hybrid --phases decode --contexts 511 512 513 \
  --batches 1 --warmups 3 --repeats 10 --output runs/ch03-boundary-new
python -m tools.chapter03.run profile --device cuda --dtype bfloat16 \
  --variant hybrid --context 512 --steps 3 --output runs/ch03-profile-new
```

主实验显式跳过 B>1 且上下文>2048 的组合，以限定显存和运行时间。每项保存所有延迟样本、环境、参数、状态统计和失败信息，不静默回退到其他后端。Prefill 计算最后一位置的 LM Head，计时包含缓存建立；Decode 的前缀建立和位置恢复对所有结构均放在计时外。

主扫描的长度都是 128 的倍数，单 token Decode 恰逢 CSA/HCA 的压缩块闭合，因此额外测相邻长度。不能把边界位置的开销当成每个 token 的平均开销。

Hybrid 的持久状态包含局部 K=V、主压缩历史、Indexer 压缩历史，以及两个压缩器的 FP32 未完成块和重叠状态。分别报告有效逻辑字节、去重后的实际 tensor storage 分配，以及整次计算的 peak allocated；不将 KV 载荷、缓存预分配和临时显存混为一谈。

Profiler 保留 CPU/CUDA 时间线与阶段标记，覆盖投影、压缩、Indexer score/排序、gather、核心 Attention 和输出投影。它不提供 Tensor Core/SFU 利用率等硬件计数器，正式延迟表不使用 Profiler 时间。

## 本地保存

租赁服务器只运行实验；源码快照、配置、日志、权重和 trace 均回收到 `runs/chapter03-20261101/`。该目录由 Git 忽略。可公开的数值记录进入 `data/chapter03/20261101/`，不上传语料正文或权重。

```bash
python -m tools.chapter03.plot --runs /path/to/hybrid /path/to/swa --output /path/to/loss.png
python -m tools.chapter03.verify --runs /path/to/hybrid /path/to/swa \
  --data /path/to/subset --source-archive /path/to/source.tar.gz --output /path/to/verification.json
```

所有命令使用新的空运行目录。保留失败记录，也不覆盖前两篇实验。

## 后续补测和正文

长序列训练后检查、连续 Decode、SWA/CSA/HCA 与 top-k 消融、固定目标后缀评估的命令及边界见[补充记录](chapter03-supplement-20261101.md)。其中一组 BF16 缓存检查未全部满足原有容差；不能把首轮短序列通过概括为长序列全面对齐。

对应正文见[第三篇](../articles/llm-hybrid-attention.md)。绘制新增图：

```bash
python -m tools.chapter03.plot_followup --runs runs/chapter03-20261101 \
  --moe runs/rtx4090-20261025/train-moe-300-extra --images articles/images/llm-hybrid
```
