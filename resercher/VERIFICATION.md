# Preparation and verification — 2026-10-23

## 后续：保守性和表面暴露度已完成

复用 step 1616/2182 已提取的 gate，新增注释与分析全部使用 CPU，未暂停训练。
完整结果见 [注释研究报告](results/annotation_study/REPORT.md)，机器可读审计见
`results/annotation_study/verification.json`。

- 独立 CPU 回归 **24 / 24 通过**（此前 14 项 + 新增 10 项注释检查）。
- 200 条链的坐标序列逐字母匹配；通过原始 component catalog 识别对应链，
  正确处理原始名称顺序与整理后 A/B 不同的 49 个结构。
- 新检查覆盖缺失内部/末端残基的对齐、多解和错误参考拒绝、重原子完整性、
  ConSurf 方向及低可信标记、MSA 重复重加权、低覆盖片段、参考序列核验、
  纯协变量效应剔除和已知残差关联保留。
- 163 条链通过唯一 ConSurf 对齐；所有这些 MSA 的实际序列数与分数表完全一致。
  764 个已下载来源文件通过 SHA256 复核。24 条多解对齐及 1 条 MSA 参考不一致
  被拒绝，另 12 条未取得数据库结果，均没有填零或静默替换。
- 43,759 个残基中，42,585 个可用 RSA；34,007 个有合格保守性分数。
  应用按链最低覆盖和位点数要求后，RSA / 保守性 / 联合调整分别分析 196 / 155 / 145 条链。
- 固定面板哈希未变；添加注释前后的 CLIP 干预分数与未调整的界面统计逐项完全一致。
- Biopython 1.86，沿用已有 Python 环境，没有安装生产依赖。
- 结束核验时原训练 PID 755747 仍存活，已到 step 2247；本次未改模型或训练配置。

最终测试日志：`control/annotation_tests.log`。图已作视觉检查，PNG/PDF 均位于注释结果目录。

## 首次实验和准备阶段记录

**后续更新：用户要求实际运行后，两个 checkpoint 的真实 GPU 提取与分析均已完成。**
原训练从完整 step 2221 恢复，已核验新 step 2223。
结果与额外检查见 [实验报告](results/comparison/REPORT.md) 及
`results/comparison/experiment_audit.json`。下文保留最初准备阶段的检查记录，
其中“尚未运行”描述的是准备结束时的状态。

状态：研究目录、固定分析面板、源代码快照和 CPU 检查已完成。
真实 GPU 门控提取尚未运行，也没有启动自动等待/停训任务。

| 检查 | 结果 |
|---|---|
| 固定 val 哈希与两个 checkpoint 的记录一致 | 通过 |
| 两个 trainable checkpoint 的 SHA256 | 准备时通过 |
| 历史记录覆盖的源文件 SHA256 | 通过 |
| 本目录代码快照导入，无生产 `flashppi` 交叉导入 | 两个版本通过 |
| q/k 各 6 个门控 tensor，完整性与有限值检查（另记录形状） | 两个版本通过 |
| `run.py --help` | 通过 |
| 重复 `prepare` 校验，不覆盖既有面板 | 通过 |
| 独立 CPU 回归 | **14 / 14 通过** |
| 合成数据贯通分析表、JSON、Markdown、PNG | 通过；临时数据已自动移除 |
| 真实 GPU 大模型加载与门控提取 | 尚未运行 |
| 保守性 / relative SASA 注释 | 尚未提供 |

CPU 检验命令：

```bash
artifacts/environments/esmc_prott5_gated_py312/bin/python -B -m unittest discover -s resercher/tests -v
```

测试覆盖原生门控公式逐值回放、NaN padding、门控缩放不变性、置换分布保持、
uniform 对照、缺失坐标处理、标签坐标边界、精确序列去重、并列 Top-K、
已有负例限定、q/k 分别干预、注释对齐、目录逃逸、GPU 忙碌保护和报告生成。

准备出的面板：100 个结构、200 条结构链 / 唯一序列、100 个结构依赖块；
228 条互作记录（25 正、203 负），覆盖其中 50 条唯一序列。最长序列 976 残基。
所有原始验证记录都保留；去重仅用于避免同一序列重复编码。

各 checkpoint 的环境版本和门控参数检查见
`manifests/preflight_before_1616.json`、`manifests/preflight_after_2182.json`。
代码来源与未被历史记录覆盖的 fallback 依赖见 `manifests/prepared.json`。

此次仅在 `resercher/` 内创建研究文件。生产训练继续运行，原训练代码、配置、
数据过滤脚本与原始数据未改动。真实 GPU 推理的数值一致性和显存占用，仍需在
首次正式提取时验证；CPU 通过不能代替这项验证。
