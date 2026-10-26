# Step 1800 在固定 DDI val 上的表现

2026-10-13，评估当前 Focal 续训实验最近已保存的 checkpoint：`runs/ddi_esmc300_focal_resume1600_20261013/last`，step **1800**；评估启动时 `best` 也是相同权重。后台训练已超过此步数，因此本报告指保存的 step 1800，不代表内存中仍在更新的模型。

## 结果

全部 **677/677 对 DDI** 均成功评估。所有指标按结构等权平均，使用完整域序列、原始方向、无随机 mask／裁剪，CUDA BF16 推理、FP32 指标计算。

| 指标 | 接触头 | 距离头推导接触 |
| --- | ---: | ---: |
| 接触 P@K | **10.5824%** | 9.8738% |
| 残基对 AP | **9.3013%** | 8.7136% |
| 随机排序的期望 P@K | 1.2609% | 1.2609% |

K 是每个结构的真实接触数。距离头分数为 `P(distance < 12 Å)`，两种读出使用完全相同的标签和有效位置。这里的 AP 在单个结构的残基对间排序，不是蛋白互作分类的 CLIP AP；本集合没有互作负例。

- 距离交叉熵：**1.719773**，先在每个结构的有效残基对上取均值，再在结构间取均值；越低越好。
- 接触头 P@K 高 **0.7085 个百分点**；按结构配对 bootstrap 的 95% 区间为 **+0.3709 至 +1.0465 个百分点**，种子 20261013，10,000 次重采样。
- 接触头在 253 对上更好，距离读出在 157 对上更好，267 对持平。
- 接触头有 **239/677（35.30%）** 的结构在前 K 个位置未恢复任何真实接触；两种读出都为零的有 219 对。
- 两种读出的前 K 个位置平均重叠 71.37%。

## 来源分层

| 来源 | 对数 | 接触头 P@K | 距离读出 P@K | 接触头残基对 AP |
| --- | ---: | ---: | ---: | ---: |
| TED / teddymer | 661 | 10.7060% | 9.9795% | 9.4102% |
| CongLab | 16 | 5.4741% | 5.5088% | 4.8031% |

总体结果主要反映 TED 子集，CongLab 仅 16 对，不宜据此作稳定的来源间优劣判断。

## 解读范围

接触头 P@K 约为随机排序期望的 **8.39 倍**，说明模型在这份监督训练隔离的 DDI 集合上具有接触排序能力。但绝对精度仍有较大改善空间，35.30% 的结构没有恢复前 K 内的真实接触。

本次结果支持该 checkpoint 的直接接触头优于距离读出，不能据此判断 Focal 相比 BCE 的因果效果；需要在同一固定集合上评估转换前的 step 1600，且训练前后比较仍同时混入额外训练步数的影响。也不应把此前训练池诊断或不同 PPI val 的数值作为直接同分布对照。

验证集来自 PPI-val 相关的原始排除集合，标签源于预测结构；尚有域家族相关性，结构 bootstrap 未对这种依赖作校正。完整数据筛选边界见 [DDI 验证集说明](DDI_VALIDATION_V1_20261013.md)。

## 复现与核验

脚本：`tools/evaluate_ddi_validation.py`。完整结果目录：`experiments/ddi_focal/checks/ddi_validation_step1800_20261013/`，包含 `metrics.json`、逐结构 `per_structure.jsonl`、`protocol.json`、checkpoint 配置与 manifest 快照、GPU 预检和独立 `audit.json`。

```bash
.venv/server/bin/python tools/evaluate_ddi_validation.py \
  --source-root experiments/ddi_focal \
  --checkpoint runs/ddi_esmc300_focal_resume1600_20261013/last \
  --validation data/ddi_validation_v1 \
  --output experiments/ddi_focal/checks/新的结果目录 \
  --gpu-memory-gib 4
```

`last` 会随训练更新，复现本报告必须提供权重哈希 `d3846f932e7a42a8c1b383ff70364c56aa8ece6e4f55ba0af286cdf015499600` 的 step 1800。验证 Parquet 哈希为 `f5bf41a953ffdaeaa0b8ad241c48805d84d6745ef1baadb75dd21cab51d67bbf`。评估器对实际加载的权重字节、模型源码、训练划分和验证数据执行哈希校验。

3 项现有接触指标回归检查通过；最长序列和最大矩阵 GPU 预检通过；677 对逐条验证名单、接触数、有效位置及有限数值，并从保存的逐结构结果独立复算所有宏平均。推理与汇总共约 59.7 秒，PyTorch 峰值分配显存约 1.43 GiB。此次为独立评估，后台训练仍继续。
