# 五物种固定 PPI test：构建与质量记录

日期：2026-10-12。数据目录：`data/multispecies_test_v1_20261012/`。本轮仅执行 CPU 数据处理、序列搜索及核验，没有启动训练或模型推理。RF2-PPI 与 FlashPPI 大肠杆菌官方文件和标签保持原样，并校验原始 SHA-256。

## 固定规模

| 物种 | taxid | 正例 | 负例 | 不同蛋白序列 | 正例 A / B / C |
|---|---:|---:|---:|---:|---:|
| 小鼠 | 10090 | 1,000 | 100,000 | 1,391 | 124 / 654 / 222 |
| 拟南芥 | 3702 | 1,000 | 100,000 | 1,473 | 668 / 332 / 0 |
| 玉米 | 4577 | 1,000 | 100,000 | 1,998 | 9 / 991 / 0 |
| 粳稻 | 39947 | 1,000 | 100,000 | 1,596 | 9 / 8 / 983 |
| 秀丽隐杆线虫 | 6239 | 1,000 | 100,000 | 1,301 | 407 / 421 / 172 |

合计 **5,000 正例、500,000 负例，505,000 对**。每个物种独立使用 1:100 比例。相同蛋白的正负样本出现次数严格为 1:100，避免单凭蛋白出现频率区分标签；每个蛋白最多参与 6 对正例，每个序列簇最多贡献 12 次正例端点。

证据层级：

- **A**：至少两个不同 PubMed 文献提供二元互作证据，或有结构实验记录，或在本轮读到的原始补充表中有明确的 BiFC 阳性复验记录。
- **B**：单文献低通量二元实验支持，或玉米在两个及以上实验文库中出现。
- **C**：其余满足筛选条件的单研究高通量二元实验支持。

这些是证据来源层级，不是统一校准的可信度概率。同一篇论文被多个数据库收录、同一实验使用不同术语，均不自动算作独立验证。**粳稻的 983 对 C 层正例不能称为逐对独立复验的金标准**；本版本保留这些标记，评估时应同时报告证据层级组成。

## 已执行的质量规则

### 正例

1. 使用可追溯到实验方法、来源记录和文献的二元互作候选；不以模型预测、共表达、遗传互作或 STRING 总分直接产生正例。BioGRID 与 IntAct 的共复合物等广义 physical 记录仍用于负例排除。[BioGRID 方法](https://wiki.thebiogrid.org/doku.php/experimental_systems)
2. 限定目标物种；粳稻不混入泛水稻 4530。序列与固定 STRING v12 序列逐字一致，且唯一映射到同物种 STRING ID。只接受 20 种标准氨基酸；保留完整序列，不裁剪。
3. 排除明确的突变、截短等构建体标记，以及已读原始补充表中的实验结果冲突。保留原始证据，数据库缺少构建体注释时不声称已经复核全部实验细节。
4. 对 [粳稻 RiPPID](https://pmc.ncbi.nlm.nih.gov/articles/PMC11923860/)读取 Table S1、S11、S12：保守排除自激活表出现的所有可映射参与蛋白，并排除存在复验阴性或不一致结果的配对。对[玉米研究](https://www.nature.com/articles/s41588-022-01262-1)读取 Supplementary Table 3、4，优先跨文库支持，排除读到的 BiFC 阴性配对。表中支持计数只作为原始实验支持量，不解释为统一的验证概率。
5. 按 MMseqs2 的 **30% identity、80% 双向 coverage** 序列簇，确保每个无序 cluster pair 只选一个正例；按证据层级和蛋白多样性确定性选择，不使用任何模型打分。
6. 新 test 的任意蛋白均排除与现有 500 对结构 val 及 4,200 对互作 val 达到 **30% identity、50% 双向 coverage** 的 MMseqs2 搜索命中。两个 val 文件合计 994 条不同序列。
7. 扫描现有 1,120,675 行 PPI train 的完整序列列，并排除精确重复的训练正例；同时排除与现有官方 test 完全相同的序列对。复核时发现的 1 对拟南芥旧训练重复和 1 对小鼠/RF2 官方重复已剔除并补齐。

### 负例

负例仅使用各物种所选 1,000 个正例的端点蛋白，不引入跨物种配对，不从全蛋白组另取简单负例，不重复抽样。每条负例都标记为 `filtered_presumed_negative`。

排除以下组合的并集：

- BioGRID、IntAct、原始实验补充表中的已知或结果不确定配对，包括未被选入正例的记录；未映射记录通过同物种可靠 ID 别名扩大排除范围，歧义映射按所有可能端点保守排除。
- STRING v12 physical score ≥700，**或** physical 网络具有任何 experimental 通道证据的配对。后一项比 FlashPPI 论文的基本分数阈值更保守。[STRING 分数说明](https://string-db.org/help/scores/)
- 已知 physical 序列簇对、相同蛋白簇内配对、自互作，以及跨来源完全相同序列的已知正例。
- 本轮读取到的旧训练集、val 和官方 test 中可精确匹配的正例。

以正例蛋白度数的 100 倍构造简单负例图，删除禁用边后执行 1,000,000 次成功的保度数交换。固定随机种子 **20261012**，禁止重复边；不宣称该算法是在所有可行图上的严格均匀采样。

## 核验结果与适用边界

`validation.json` 记录逐物种核验结果：样本数正确、序列哈希与物种/STRING 映射一致、正例簇对唯一、负例排除规则零违规、正例冲突零违规、蛋白出现次数比例准确；新物种间无完全重复序列对；官方输入哈希不变。现有 val 同源命中为零，现有训练集的精确序列对重叠为零。

**这些检查不构成“已与全部 DDI/PPI 训练数据同源隔离”的声明。** 本轮只对现有 val 做了同源排除，对旧 train 做了精确重复扫描。下一次 DDI 预训练/PPI finetune 前，必须用冻结 test/val 的序列对训练候选执行完整同源过滤，包括 DDI 的局部片段匹配；旧 checkpoint 不能据此直接获得“无泄漏”评估资格。

随机筛选负例仍可能包含未知真实互作；高通量正例也可能有实验噪声。本版本提高了可追溯性、排除范围和抽样控制，不能保证每个生物学标签绝对正确。

本数据提供 **PPI 分类标签**，不提供残基接触图或距离矩阵。不能把互作负例直接伪造成实验接触/距离真值。序列未截断；各物种最长蛋白分别为 5,035、2,193、3,811、1,820、8,081 aa，后续推理需提前验证长序列资源需求。

## 文件与复现

每物种提供：

- `{species}_test.parquet`：完整序列、STRING ID、物种、二元标签、证据层级、长度、簇和稳定 pair ID。
- `{species}_pairs.tsv`：不重复存储序列的轻量表。
- `{species}_positive_evidence.parquet`：逐正例来源记录、PubMed、实验方法、复验/文库支持信息。
- `{species}_proteins.fasta`：该 test 使用的完整序列。

另有 `manifest.json`（输入/输出 SHA-256、脚本哈希、软件版本）、`config.json`、`validation.json`、`build_statistics.json`、val 搜索命令与全局排除记录。`new_test_holdout_sequences.fasta` 及 `new_and_available_official_test_holdout_sequences.fasta` 用于后续训练排除，后者包含当前已取得的两个官方 test；不包含尚未取得的铜绿假单胞菌官方清单，也不替代 val 的保护名单。

参数固定在 `configs_data/multispecies_test_v1.json`，构建脚本为 `tools/build_multispecies_test.py`。使用已保存来源快照复现：

```bash
/home/zlab/miniconda3/bin/python tools/build_multispecies_test.py prepare
/home/zlab/miniconda3/bin/python tools/build_multispecies_test.py global-exclusions
/home/zlab/miniconda3/bin/python tools/build_multispecies_test.py build
/home/zlab/miniconda3/bin/python tools/build_multispecies_test.py release
/home/zlab/miniconda3/bin/python tests/test_multispecies_test_builder.py
```

`release` 会拒绝覆盖已冻结的输出目录；复现实验应使用新的输出位置。来源下载脚本是 `tools/download_multispecies_test_sources.py`，但重新查询在线数据库可能得到不同版本，应以 manifest 固定的本地快照为准。

2026-10-12 按用户要求取消番茄并清理 `data/`。可重建的 `multispecies_test_v1_20261012_work/`、未采用的候选版本、聚类临时文件及番茄专用下载已删除；上述 `prepare` 和 `global-exclusions` 会重建工作文件。正式 test 和原始 manifest 均未修改；历史 manifest 中被清理的工作输入及番茄附件见 [清理明细](DATA_CLEANUP_20261012.json)。下载清单将番茄标记为 `removed_after_scope_change`，新构建的来源清单与历史 manifest 因此会有差异。目录用途见 [data/README.md](../data/README.md)。

## 番茄与铜绿假单胞菌补充进展

补充记录目录：`data/multispecies_test_v1_20261012_sources/supplementation/`。

- **番茄：已按用户要求取消**。此前从 [YFT1/SlTFT1 原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12576456/)整理的 6 对候选涉及 M82 品种及 YFT1 C 端片段，未完成精确映射、从未并入正式 test。番茄专用候选表及下载文件现已清理，不再继续补充。
- **铜绿假单胞菌**：重新检索官方 GitHub 主分支归档与 Hugging Face 模型/数据集目录，仍未取得论文 179 正例基准的原始清单。另从 IntAct 收集到 152 条同 taxid 蛋白记录，其中非自互作阳性记录对应 37 个不同 ID 配对，涉及多个菌株；仍需构建体、展开记录和序列复核。这些补充候选不等于官方 test。查询记录、来源哈希和候选表均已保存。
