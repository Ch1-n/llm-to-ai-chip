# 第三篇补充实验记录

2026-11-01，沿用首轮 RTX 4090、PyTorch 2.8.0+cu128 独立环境。新增命令进入 `tools/chapter03/supplement.py`，不覆盖首轮结果。完整原始输出位于本地 `runs/chapter03-20261101/supplement-*/`，可公开副本位于 `data/chapter03/20261101/supplement/`。

## 训练后长序列检查

两份 300 步 checkpoint，分别检查 2048/8192 token，FP32 与 FP32 权重+BF16 autocast，共 8 组。比较完整前向与跨边界分块末位置、最后 130 个位置逐 token 输出，并扰动未来 128 个 token 检查之前两个位置不变。

7 组全部满足原有容差。Hybrid、2048、BF16 autocast 的 130-token 增量对照未全部满足逐元素条件：6,533,410 个元素里 3 个不满足，最大绝对差 0.21875、相对 L2=0.0036096。该组因果和分块末位置通过，FP32 通过。容差仍为 atol=0.08、rtol=0.08、relative L2≤0.1，没有调宽。

独立诊断使用同一输入和权重，不强制选择；两个 CSA 层在最后 130 个位置中分别有 34/45 个位置的压缩块集合变化。这不是将全部误差归因于 Indexer 的因果隔离试验。保留未通过记录，不宣称 BF16 长序列缓存推理已经全面对齐。

该检查覆盖特定长度、一个随机输入种子与最后 130 个位置，不覆盖所有 token、所有分块和训练样本。与首轮原生 BF16 随机权重检查的精度设置不同。

## 连续 Decode

B=1，原生 BF16、随机权重，先建立 512 token 前缀，随后连续推进 256 token，重复 3 次，五种结构共 3840 条单 token 计时。输入预先生成，不包含 tokenizer/采样；每步同步，前缀建立在计时外。记录是连续缓存推进，不是恢复同一位置。

| 结构 | 平均ms | 样本P95 ms | 样本P99 ms | token/s |
| --- | ---: | ---: | ---: | ---: |
| MHA | 7.331 | 7.502 | 8.033 | 136.42 |
| MLA | 8.896 | 9.273 | 10.755 | 112.41 |
| shared-dense | 10.538 | 11.223 | 17.124 | 94.89 |
| SWA | 11.039 | 12.651 | 14.361 | 90.59 |
| Hybrid | 14.091 | 15.999 | 18.781 | 70.97 |

这些分位数来自有限的同步调用样本，不是服务 SLA；包含主机发起与同步开销。跨结构未做参数/质量匹配，也不是生产内核性能排名。

## 路径和 top k 消融

shared-dense、SWA、CSA、HCA 各一组，Hybrid top-k=1/8/32 三组，共 7 组。B=1、上下文 2048/8192、Prefill/Decode，合计 28 项，每项预热 3 次、测量 10 次。所有结果有限、命令完成。

8192 token Decode 中位数：shared-dense 10.365 ms、SWA 11.013 ms、CSA 22.360 ms、HCA 15.032 ms；Hybrid k=1/8/32 分别为 16.046/16.176/16.228 ms。都是压缩块闭合位置，不能代替连续平均值。

相应持久逻辑状态为 6.000/0.09375/2.361328/0.140625 MiB；三个 Hybrid 都为 0.865234 MiB。top-k 不删除历史，且参考实现仍然完整评分和稳定排序，因此状态不随 k 减少，延迟差异也不大。

同一层结构的 top-k 变体沿用相同初始化，不改变参数形状；改变全部层的路径会改变参数集合，因此不是严格的参数/质量控制试验。没有为所有 k 重训，也不声称 k 变化不会影响质量。

## 相同目标后缀的历史长度对照

从验证 token 流中按种子 2026 固定选 8 个互不重叠的 8192-token 区间，每个固定最后 128 个目标 token，输入长度取 512/2048/8192。两份已训练 checkpoint，共 48 条后缀记录、6144 个目标 token 的计算；每个模型每种长度实际上是同一组 1024 个目标 token，不能当作 6144 个独立样本。

| 长度 | SWA交叉熵 | Hybrid交叉熵 |
| --- | ---: | ---: |
| 512 | 3.036479 | 3.054206 |
| 2048 | 3.036575 | 3.052915 |
| 8192 | 3.036161 | 3.053559 |

没有观察到明显持续的长历史收益。它不是长文检索评测：权重只在长度 512 训练 300 步，样本少、packed 文本可能跨 EOT，裁剪后的 RoPE 从零开始。评估经过缓存路径，BF16 缓存的限定检查仍有上述未通过项，因此不能把这些小差异当作质量优劣证据。

## 复跑

```bash
python -m tools.chapter03.supplement check --checkpoints /path/to/hybrid.pt /path/to/swa.pt \
  --contexts 2048 8192 --output runs/check-new
python -m tools.chapter03.supplement continuous --prefix 512 --steps 256 --repeats 3 \
  --output runs/continuous-new
python -m tools.chapter03.supplement ablation --contexts 2048 8192 --output runs/ablation-new
python -m tools.chapter03.supplement history --checkpoints /path/to/hybrid.pt /path/to/swa.pt \
  --data /path/to/subset --contexts 512 2048 8192 --samples 8 --suffix 128 \
  --output runs/history-new
python -m tools.chapter03.diagnose_selection --checkpoint /path/to/hybrid.pt \
  --context 2048 --output runs/selection-new
```

检查命令当前在任一对照未通过时返回非零。首次运行保存了失败状态，但当时 CLI 尚未传递非零退出码；后续只补上退出码处理，不改原始数值。首个上传包和最终源码分别归档，用环境中的源码哈希区分，不覆盖首轮 `source-measured.tar.gz`。
