# 接触头与距离推导接触的对照诊断

## 结论

在同一个 step1600 checkpoint 上，距离推导的接触评分没有在固定 PPI val 上表现出明确优势；独立接触头在 DDI 训练池诊断样本上反而略好。本次不支持“距离头已经学好接触，而独立接触头明显失效”的解释。

更明显的现象是 DDI 训练池与 PPI val 的表现差距。它提示需要检查训练数据记忆、DDI→PPI 迁移、序列长度和数据来源差异，不能仅凭本次诊断确定根因。

**step1600 是切换 Focal 前的权重。** 当前 Focal run 的 `resume_initial` 与原 BCE step1600 权重完全相同；本次不能证明 Focal 优于或劣于 BCE。

## 评估协议

- checkpoint：`runs/ddi_esmc300_focal_resume1600_20261013/resume_initial`。
- 权重 SHA256：`fd89ef7df1cd48ad894e9b95106ffca4c454699fc2c53ad86d62a8f0602124e6`。
- 固定结构 val：原有完整 500 对，文件 SHA256 `fc22be650bfd89607c1e68510032d13113230c1ca4fd49ed2a92c7e8c957ad54`。
- 完整序列、无裁剪、无 MASK、原始链方向；CUDA BF16 推理，评分使用 FP32。
- 独立接触头使用原始 logits 排序；距离评分为所有距离严格小于 12 Å 的分箱概率之和。
- 分箱边界为 2–32 Å；接触对应前 11 个 bin。距离恰好为 12 Å 不计为接触。
- 两种评分使用相同坐标有效掩码、相同 K。K 为每个结构的真实接触位置数；最终逐结构等权平均。
- DDI：使用训练的无序序列簇对采样器，独立固定随机种子 `20261013`，从训练池抽取 256 个不同结构；不进行训练，不保证这些样本此前未被训练访问。
- 全部 DDI 诊断样本链长 27–368，中位数 118；250 对来自 teddymer，6 对来自 conglab。不是独立 DDI val/test。

## 结果

| 数据 | 数量 | 独立接触头 P@K | 距离推导 P@K | 随机排序期望 P@K |
|---|---:|---:|---:|---:|
| 固定 PPI val | 500 | 4.0720% | 4.1392% | 2.0551% |
| PPI val：两条链均 ≤512 | 410 | 4.7217% | 4.7874% | 2.4243% |
| PPI val：至少一条链 >512 | 90 | 1.1121% | 1.1864% | 0.3734% |
| DDI 训练池诊断样本 | 256 | 19.1006% | 18.4564% | 1.2593% |

随机排序期望按每个结构 `真实接触数/有效位置数` 计算，再逐结构平均。长度较长的结构接触更稀疏，不能把其较低的原始 P@K 全部归因于长度外推。

固定 PPI val 中，距离推导减去独立接触头的平均差为 **+0.0672 个百分点**。逐结构配对 bootstrap（10,000 次、种子 20261013）95% 区间为 **[-0.1334, +0.2629] 个百分点**。距离评分胜 115 对、负 110 对、平 275 对；两者 top-K 集合平均重合率 77.04%。其中 249 对的两个评分均未在 top-K 命中真实接触。

DDI 诊断样本上的对应差为 **-0.6442 个百分点**，区间 **[-0.9858, -0.2943] 个百分点**。距离评分胜 55 对、负 118 对、平 83 对；top-K 平均重合率 83.19%。这些区间只描述当前抽样，不处理剩余生物学相关性。

残基对 AP（逐结构平均，**不是互作 CLIP AP**）：PPI val 独立头 0.038195、距离推导 0.038386；DDI 诊断独立头 0.169670、距离推导 0.158707。

## 正确性与资源核验

- 原接触头 **500/500 对精确复现**原 step1600 验证结果，最大绝对误差 0。
- 所有标签重新检查：坐标距离、原始接触索引、距离分箱推导的接触标签完全一致。
- 3 个 CPU 回归检查通过：严格 12 Å 边界、概率求和与 log-odds 的等价性、两种评分的掩码和 K 一致性。
- 两次独立评估均限制 CUDA 分配不超过 4 GiB；val 最大分配 1.8754 GiB，DDI 诊断 1.4015 GiB；GPU 驱动/context 占用另计。
- val 用时约 92 秒，DDI 诊断约 40 秒。后台训练源码、配置、优化器和进程未修改；评估进程已正常退出。
- 没有访问 RF2 或其他 test，也没有根据本次结果更新模型或选模规则。

## 下一步依据

暂不根据本次结果替换接触头，或仅为放大梯度而恢复旧归一化。优先对 Focal 续训后的固定 checkpoint 使用同一诊断，并将训练前后的变化与真正的 BCE/Focal 同起点对照区分。DDI 训练池表现不能替代独立 DDI 泛化评估，也不能与 PPI val 的差值直接解释为损失函数问题。

## 复现与产物

从仓库根目录运行：

```bash
.venv/server/bin/python tests/test_ddi_contact_readouts.py

.venv/server/bin/python -u tools/compare_ddi_contact_readouts.py \
  --source-root experiments/ddi_focal \
  --checkpoint runs/ddi_esmc300_focal_resume1600_20261013/resume_initial \
  --output experiments/ddi_focal/checks/contact_readouts_step1600_reproduction \
  --device cuda --gpu-memory-gib 4 \
  --reference-contact-metrics runs/ddi_esmc300_no_interaction_resume1100_20261013/val_step_0001600/contact_metrics.json

.venv/server/bin/python -u tools/diagnose_ddi_readouts.py \
  --source-root experiments/ddi_focal \
  --checkpoint runs/ddi_esmc300_focal_resume1600_20261013/resume_initial \
  --output experiments/ddi_focal/checks/ddi_readouts_step1600_reproduction \
  --pairs 256 --seed 20261013
```

原始结果目录：

- `experiments/ddi_focal/checks/contact_readouts_step1600_20261013/`
- `experiments/ddi_focal/checks/ddi_readouts_step1600_20261013/`

各目录保存 `protocol.json`、`per_structure.jsonl`、`metrics.json`，包含代码/数据/权重哈希和逐结构结果；DDI 协议还记录抽样索引与 pair ID。
