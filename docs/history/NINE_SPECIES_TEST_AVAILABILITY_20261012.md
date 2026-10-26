# 九物种 PPI test 数据可用性盘点

> 后续状态（2026-10-12）：本页保留最初九物种盘点结果。五物种 test 已冻结，见[构建报告](MULTISPECIES_TEST_V1_20261012.md)；用户已取消番茄，番茄专用数据和容量盘点临时导出已清理。铜绿假单胞菌仍待官方清单。下载与盘点脚本现已排除番茄；不要在现存快照上重跑下方历史盘点命令，以免覆盖冻结 test 的构建输入。目录现状见 [data/README.md](../data/README.md)。

盘点日期：2026-10-12。已读取本地官方基准，并实际下载、解析实验记录、补齐部分序列、运行 MMseqs2 聚类和计算负例容量。本轮没有启动训练或模型推理，没有修改原始 PPI/DDI 数据库，也没有冻结新的 test。

**结论：人和大肠杆菌已有可沿用的官方基准；五个新增物种具备构建千级正例 test 的数据基础；番茄需要补充证据与映射；铜绿假单胞菌需要取得官方原始清单。** 以下是本轮来源范围内的实测可用量，不是各物种所有已发表 PPI 的总量。

## 九物种总表

新增物种的“正例候选”指：同物种二元互作实验候选 → 序列映射 → 无序序列对去重 → 30% identity / 80% 双向 coverage 聚类对去重。“配套规模”进一步限定正例两端都能按完全相同序列唯一映射到 STRING v12，再计算负例池。

| 物种 / 基准 | 已核验正例或去冗余候选 | 目前可配套的规模：正例 / 负例 | 状态与主要来源 |
|---|---:|---:|---|
| 人，RF2-PPI | 3,000（官方固定） | 3,000 / 30,000 | 本地文件及序列已核验，保留官方 1:10 |
| 大肠杆菌，FlashPPI | 650（官方固定） | 650 / 65,000 | 本地文件已核验，保留官方 1:100 |
| 铜绿假单胞菌，FlashPPI | 179（论文报告） | 原始比例约 1:48.8；精确负例数待核验 | 尚未取得官方原始样本清单，不能标记为就绪 |
| 小鼠，10090 | 3,339 | 2,377 / 237,700 | BioGRID；可支持 1:100 |
| 拟南芥，3702 | 25,828 | 25,100 / 2,510,000 | BioGRID；可支持 1:100 |
| 玉米，4577 | 48,851 | 48,104 / 4,810,400 | 主要为 IntAct 收录的实验 Y2H；可支持 1:100 |
| 粳稻，39947 | 18,291 | 6,367 / 636,700 | RiPPID 补充表、BioGRID、IntAct；可支持 1:100 |
| 番茄，4081 | 54 | 12 / 63 | BioGRID、IntAct；严格映射子集仅约 1:5.25 |
| 秀丽隐杆线虫，6239 | 8,184 | 6,921 / 692,100 | BioGRID；可支持 1:100 |

**新增物种的负例数量是容量核算，不是已经生成或实验验证的负样本。** 所有新增候选还需要构建体、标签冲突与来源复核。尚未检查它们与拟用 DDI/PPI train、val 的同源重叠，不能据此宣称无泄漏。原有两个可用官方 test 不重采样、不改标签。

官方微生物基准的数量、去冗余和 STRING 排除规则见 [FlashPPI 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13291599/)。人类基准来源为 [CongLab 官方下载页](https://conglab.swmed.edu/humanPPI/humanPPI_download.html)；本地实测与 SHA-256 保存在 `official_local_verification.json`。

## 新增六物种：各阶段实测数量

| 物种 | 映射后广义 physical 对 | 映射后二元实验对 | 30%/80% 聚类对 | STRING 可用正例覆盖率 | 排除后负例池容量 |
|---|---:|---:|---:|---:|---:|
| 小鼠 | 66,831 | 3,940 | 3,339 | 71.2% | 2,121,430 |
| 拟南芥 | 64,695 | 38,931 | 25,828 | 97.2% | 27,444,793 |
| 玉米 | 49,377 | 49,364 | 48,851 | 98.5% | 106,075,472 |
| 粳稻 | 18,793 | 18,792 | 18,291 | 34.8% | 8,276,505 |
| 番茄 | 98 | 98 | 54 | 22.2% | 63 |
| 线虫 | 31,496 | 8,650 | 8,184 | 84.6% | 5,649,898 |

“广义 physical”包含共复合物等实验关联，不全部充当直接二元互作正例；已映射的这些记录全部进入负例排除表。覆盖率分母为聚类后正例候选数，低覆盖率主要表示版本、基因模型或序列映射尚未解决，不表示其余实验结果无效。

## 数据版本与处理口径

- BioGRID：2026-10-12 下载的 **5.0.262** organism tab3；只保留双方精确属于目标物种的 physical 记录，排除 RNA 类实验、基因自互作和相同序列配对。官方已有 test 不应用这些新增筛选。[BioGRID 下载](https://downloads.thebiogrid.org/BioGRID/Latest-Release/)
- 二元候选层：BioGRID 的 Two-hybrid、PCA、FRET、Co-crystal Structure、Reconstituted Complex、Far Western、Surface Display；IntAct 使用脚本固定的 PSI-MI 方法白名单，排除复合物展开和明确 mutation 构建体。该层仍是待复核的实验候选，不保证每条都有直接结合界面证据。[BioGRID 方法说明](https://wiki.thebiogrid.org/doku.php/experimental_systems)
- IntAct：使用已有 **2026-09-16 快照**的玉米、粳稻、番茄记录，不声称已经重新获取所有最新记录。序列要求源 ROGID 校验一致；玉米的许多失效 accession 通过精确序列校验值恢复。
- UniProt：本轮成功完成 60 个公开 accession 查询批次，返回版本 **2026_03**，保留查询清单和响应哈希；旧序列缓存也与原始 manifest 校验。没有要求旧工程的额外 embedding 特征齐全。
- STRING：六物种均已下载 **v12.0** protein sequences 与 physical links。仅用于序列映射及排除 physical score ≥700 的负例候选；不把 STRING 预测或同源转移边直接当作实验正例。[STRING 分数说明](https://string-db.org/help/scores/)
- 去冗余：MMseqs2 **18.8cc5c**，`--min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 0 --alignment-mode 3 -s 7.5 --threads 4`。按无序 cluster pair 取一个代表；优先保留两端可精确映射 STRING 的成员，再按结构实验标记和稳定序列键排序，不使用模型预测分数。
- 负例容量：仅在可映射正例的同物种端点池内计算不同蛋白的无序组合，排除所有已映射的已知 physical 正例和 STRING ≥700 边。`容量 = n(n−1)/2 − |实验正例 ∪ STRING 排除边|`。未映射的实验记录尚不能完全纳入排除，因此这仍是指定排除表下的候选容量；未来采样后还要检查假负例风险。
- 本轮没有施加长度上限或裁剪。长链与少数非标准残基的统计见 `sequence_and_evidence_quality.json`，后续需固定推理和残基处理规则。

30%/80% 用于本轮 **test 内部去冗余**。FlashPPI 论文另有训练排除的 >30% identity / >50% coverage 描述；不能把两种用途混为一谈。对 DDI 的局部片段同源过滤也尚未在本轮执行。

## 需要特别注意的来源与缺口

### 玉米：实验数据充足，证据集中

BioGRID 单库只给出极少候选，但 IntAct 收录了 [2023 年玉米多组学研究](https://pubmed.ncbi.nlm.nih.gov/36581701/)的 Y2H 实验。映射后二元候选中 49,353 / 49,364 对来自这一研究，不能把整个多组学网络或预测关联全部作为 PPI。本次纳入的是 IntAct 记录为相应实验方法的边。后续应保留文献、实验批次和构建体信息，并单列低通量或独立验证子集。

### 粳稻：取得了原始补充表，正文与文件数量存在差异

[RiPPID / BIP-seq 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11923860/)报告 23,032 条边、12,741 个节点。本次实际下载的 Table S2 为 **23,029 行、11,784 个不同 locus**；移除 103 条自互作后为 **22,926 个不同 locus 对**。保留此差异，不补造缺失记录。

按该研究使用的 Nipponbare RGAP7 体系，采用官方非 TE representative 蛋白（每个 locus 的最长 CDS 模型），可映射为 18,562 个不同序列对，再与数据库记录合并。该选择不是对实际实验片段或剪接异构体的完全重建，需要后续构建体复核。[RGAP7 官方下载与代表模型定义](https://rice.uga.edu/download_osa1r7.shtml)

目前 6,367 对聚类代表可通过严格 STRING 序列映射；不能把 18,291 全部视为负例过滤已经就绪。约 98.8% 的映射后二元候选来自 RiPPID，存在明显来源集中性。泛水稻 taxid 4530 的记录没有自动混入粳稻。

### 番茄：正例和负例池都不足

本轮 98 个映射序列对去冗余后为 54 对，代表仅涉及 41 条不同序列。即使全部完成 STRING 映射，在排除本轮已知 physical 对后、尚未排除 STRING 边前，也只有 **731** 个剩余组合，远少于 54×100=5,400。因此补齐映射本身也不能解决 1:100 容量问题。

当前严格映射子集为 12 对正例、13 条序列，已知实验边 15 条；`13×12/2−15=63`。不应通过重复负例、跨物种配对或改为全蛋白组抽样来暗中补足比例。

这不是“番茄全世界只有这些 PPI”。还可补充文献中的实测对，例如 [2025 年 YFT1 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12576456/)；本轮没有把尚未逐条整理的文献算入实测总数。[PTIR](https://www.nature.com/articles/srep25047)主体是预测网络，不能整体用作实验 test 真值。

### 铜绿假单胞菌：论文数字不等于已拿到基准

论文确认 179 个正例，实际负例比例约 1:48.8，评估时加权到 1:100。本轮未在取得的作者 Hugging Face 数据集目录或本地文件中定位该原始清单；部分仓库/补充材料请求失败。因此精确负例数、样本顺序和标签文件仍待核验，不能由四舍五入比例反推并声称已复现。

### 互作评估与接触评估分开

上述新增六物种主要支持互作 AP、加权 AP 及基于 contact score 的互作排序评估。没有实验结构和残基映射时，不能计算残基接触 P@K 或距离预测误差。BioGRID 中标记 Co-crystal Structure 也不等于已经有可用的接触标签。

现有大肠杆菌文件中 649 个正例有非空 contacts，另 1 个正例的 contacts 为空；保持官方互作标签不变，接触评估时另外核查该条。RF2-PPI 当前分类基准文件不提供残基接触标签。

## 建议的首版规模与复现

保留 RF2-PPI 和大肠杆菌全量官方 test。小鼠、拟南芥、玉米、粳稻、线虫可先规划每物种 **1,000 个正例 + 100,000 个负例**，按方法、来源和蛋白家族控制组成；实际挑选 1,000 个正例后，需要重新计算其端点池容量，不能直接照搬全候选池容量。番茄暂列补充探索集，铜绿假单胞菌等待原始清单。表中的全候选规模用于判断可用性，不要求首版全部评估。

本轮产物目录：`data/benchmark_inventory_20261012/`。

- `nine_species_availability.tsv`：九物种总表；`cluster_and_capacity_inventory.tsv`：六物种详细计数。
- `evidence_inventory.parquet`：实验方法、来源文献、序列映射状态；`candidate_pairs.parquet`：去重候选。
- `*_representative_candidates.csv`：用于容量盘点的确定性代表，**不是冻结的 test**。
- `clusters/*_cluster.tsv`、`*_command.json`、`*_input.sha256`：聚类成员、命令与输入哈希。
- `input_manifest.json`、`audit_manifest.json`、`validation.json`：输入/输出哈希、软件版本及核验结果；`raw/` 保留下载记录和版本。

在保留原始快照与映射缓存的本机环境中复现计数：

```bash
/home/zlab/miniconda3/bin/python tools/inventory_nine_species.py prepare
/home/zlab/miniconda3/bin/python tools/inventory_nine_species.py cluster
/home/zlab/miniconda3/bin/python tools/inventory_nine_species.py summarize
```

聚类缓存核对输入 FASTA 的 SHA-256；输入变化会报错，不能静默复用旧聚类。下载脚本为 `tools/download_inventory_sources.py`，缺失 accession 恢复脚本为 `tools/recover_inventory_sequences.py`。复现实测数字应使用已保存快照，重新查询在线数据库可能产生不同结果。
