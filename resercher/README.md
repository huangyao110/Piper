# CLIP 残基门控研究

> Piper 源码发布说明（2026-10-26）：本目录保留独立分析代码与合成 CPU 测试。
> 下述报告、checkpoint、完整 vendor 快照和运行控制记录为本地历史研究产物，不随 GitHub 发布；
> 这些结果也不是当前 RF2-PPI v3 fresh 模型的结论。复现实验需要先取得 `study.json` 指向的历史资产。
> CPU 测试只使用 `tests/fixtures/` 中带来源哈希的小型门控快照，不依赖本地 vendor 目录。

按照约定使用目录名 **`resercher`**。本目录用于独立研究 CLIP 的残基加权池化，
所有新增脚本、代码快照、数据视图、测试、缓存和结果都在这里。原模型、训练脚本、
数据过滤规则和后台训练不作修改。checkpoint/大型预训练权重通过明确路径只读加载。

研究问题、对照和解释边界见 [PLAN.md](PLAN.md)。
**2026-10-23 已完成首次真实 GPU 实验**，结果见
[实验报告](results/comparison/REPORT.md)：CLIP 门控有预测作用及温和界面富集，
首次实验当时尚未检验保守性和表面暴露度。训练已从 step 2221 原样恢复。
**随后已完成保守性/暴露度分析**，见
[注释研究报告](results/annotation_study/REPORT.md)：step 2182 的 gate 倾向表面暴露残基；
控制氨基酸种类与暴露度后，没有检测到稳定的独立保守性关联。
同类氨基酸、相近暴露度的对照下仍有少量界面富集，但不足以把 gate 当作功能重要性。
准备阶段的 14 项 CPU 检查见 [VERIFICATION.md](VERIFICATION.md)，
真实实验与续训核验见 `results/comparison/experiment_audit.json`。

## 文件布局

| 路径 | 用途 |
|---|---|
| `study.json` | 固定 checkpoint、val 文件哈希、随机种子和分析参数 |
| `run.py` / `clip_gate/` | 独立入口、提取、界面关联、CLIP 干预、比较 |
| 本地 `control/` | 已完成实验的控制记录；一次性暂停/恢复脚本已退役，不作为发布入口 |
| `summarize_experiment.py` | 历史分数回放、配对变化、续训核验与比较图 |
| `annotate.py` / `clip_gate/annotations.py` | 原始坐标 SASA、ConSurfDB 真实 MSA/保守性、精确残基映射和质量控制 |
| `annotation_report.py` | 注释关联、氨基酸/RSA 调整、按结构依赖块汇总与图 |
| `vendor/before_1616/`、`vendor/after_2182/` | 对应模型源代码的本地快照；不导入工作区生产代码 |
| `data/` | val 小文件逐字节副本、唯一序列清单、标签分析视图 |
| `manifests/` | 来源、哈希、快照 fallback 说明、CPU 预检结果 |
| `results/` | 门控表、CLIP 向量、分析报告与图；不保存完整 residue embedding |
| `tests/` | 独立 CPU 回归；不需要加载完整大模型 |
| `cache/`、`tmp/` | 本研究专用的运行缓存与临时文件 |

只复制小型代码和 val 数据，不复制多 GB 模型权重。外部权重被删除或更改时检查会
失败，不会静默换用其他 checkpoint。每个 checkpoint 必须用独立进程加载，避免
不同历史版本共用 Python module cache。

## 使用

从仓库根目录运行，沿用已安装的 gated 训练环境（不安装或修改生产依赖）：

```bash
research_python="$PWD/artifacts/environments/esmc_prott5_gated_py312/bin/python"
"$research_python" -B resercher/run.py prepare
"$research_python" -B resercher/run.py preflight --checkpoint before_1616
"$research_python" -B resercher/run.py preflight --checkpoint after_2182
"$research_python" -B -m unittest discover -s resercher/tests -v
```

`prepare` 不改变 split；只校验并复制固定 val，建立解释分析视图，校验 checkpoint
和记录源代码快照。已有准备记录时仅校验，不覆盖。`preflight` 导入本地模型快照、
检查 gate 参数完整与有限，并保存环境版本；它不会加载大模型或启动 GPU 推理。

**GPU 空闲时**，依次执行以下命令；脚本不会暂停或自动续训生产任务：

```bash
"$research_python" -B resercher/run.py extract --checkpoint before_1616
"$research_python" -B resercher/run.py analyze --checkpoint before_1616
"$research_python" -B resercher/run.py extract --checkpoint after_2182
"$research_python" -B resercher/run.py analyze --checkpoint after_2182
"$research_python" -B resercher/run.py compare
```

GPU 0 被占用或空闲显存不足时，`extract` 在模型加载前退出。沿用正式评估全长输入、
batch 1 和 FP16 autocast；不为了节省显存裁剪序列。结果目录已存在时拒绝覆盖。
失败记录会保留，修复后将失败目录改名归档于 `results/` 内再重跑。

首先查看 `results/<checkpoint>/analysis/REPORT.md` 与 `overview.png`；完整数值在
`summary.json`、`interface_per_structure_side.parquet`、`gate_distributions.parquet`、
`intervention_pair_scores.parquet`。置信区间针对结构依赖块；20 次打乱的离散程度不是
样本置信区间。当前 val 上的发现只作为探索，不用于声称确认性泛化结论。

保守性/表面暴露度需要额外注释，格式见 [ANNOTATIONS.md](ANNOTATIONS.md)。准备后可运行：

```bash
"$research_python" -B resercher/run.py analyze --checkpoint after_2182 --annotations data/annotations.csv
```

这将生成独立的 `analysis_annotated/`。无注释时仍可做界面与 CLIP 干预分析，但报告
明确标出缺失的生物学控制。

已经完成的注释分析可按以下顺序复现，全部使用 CPU；已有最终输出拒绝覆盖，
原始下载和逐链注释会复用缓存。源坐标及 CongLab component catalog 只读访问，
ConSurfDB 只获取公开既有结果；这不涉及重新生成训练集或修改同源过滤条件。

```bash
OPENBLAS_NUM_THREADS=1 "$research_python" -B resercher/annotate.py sasa --workers 2
OPENBLAS_NUM_THREADS=1 "$research_python" -B resercher/annotate.py conservation --workers 4
"$research_python" -B resercher/annotate.py assemble
OPENBLAS_NUM_THREADS=1 "$research_python" -B resercher/run.py analyze --checkpoint before_1616 --annotations data/annotations.csv
OPENBLAS_NUM_THREADS=1 "$research_python" -B resercher/run.py analyze --checkpoint after_2182 --annotations data/annotations.csv
OPENBLAS_NUM_THREADS=1 "$research_python" -B resercher/annotation_report.py
```

固定面板 200 条链中，RSA 关联合格 196 条，保守性关联合格 155 条，
联合调整合格 145 条；缺失位置没有用零填充。全部数值及来源见上述注释报告。
