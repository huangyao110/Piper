# RF2-PPI 数据隔离对照与 v2 过筛风险复核

日期：2026-10-24。范围：查阅原始论文、作者公开代码及本地作者数据说明，并对当前 v2 新增 DDI 命中做固定抽样诊断。**没有修改过滤脚本、协议锁、train/test 成员，也没有启动训练。**

后续进展：用户在核实以下规则后，明确授权按 RF2-PPI 的命中条件和双端逻辑重新构建，并要求将现有入口改名为 `tools/train_test.py`。实施版本为 `conglab_distilled_ecoli_rf2ppi_v3`，详细固定标准见 [DATA_FILTERING_POLICY.md](DATA_FILTERING_POLICY.md)。本文件以下诊断保留原始 v2 复核语境，不能误读为继续沿用 v2 的建议。用户随后明确授权删除过时目录；v2 输出与其草稿现已删除，历史参数锁、诊断记录与数量对照仍保留，见 [清理记录](DATA_CLEANUP_20261024.json)。

## 判断

当前 v2 的程序验收通过，表示其成员符合已锁定的计算规则；这不能证明规则可靠地识别了生物学同源，也不能证明复现了 RF2-PPI。

现已取得并核实 RF2-PPI 补充方法。其 M3.1（第 20–21 页）和 M3.3（第 27 页）明确规定：单条命中同时满足 identity >30%、MMseqs E-value <1e-5，以及 query 或 hit 任一侧 coverage >50%；一对中的**两端都**有合格的人类同源命中才排除整对。这适用于文中明确描述的 PPI 正例、PPI 负例和 DDI 正例。v2 的任一端排除、PPI 双侧 coverage、DDI 训练域 coverage 和不再筛显著性的规则均不等同于 RF2-PPI。

固定抽样显示，v2 新增 DDI 命中绝大多数在所复核方向没有达到论文的显著性门槛。**不建议直接用当前 v2 开始新的正式训练。**参考 RF2-PPI 不等于自动获得修改现有协议的授权；特别是改为双端命中才排除，会放宽用户原先希望的任一序列隔离要求。

之前将“直接比对满足阈值”表述为“已确认的同源”过于肯定。全量比对解决了预筛选可能漏掉比对的问题，但不能自动把弱比对变成可靠的同源证据。

## 原始来源中已确认和未确认的内容

| 项目 | RF2-PPI 可确认内容 | 当前 v2 |
|---|---|---|
| identity 与显著性 | 补充方法 M3.1/M3.3：identity >30%，MMseqs E-value <1e-5 | identity >30%，搜索允许 E-value 到 1e6，最终不再筛显著性 |
| 排除对象 | 两条训练链/域分别在人类蛋白集合中有合格命中；作者提供 `*_nohomo.csv` 名单 | 排除固定 E. coli test 正负样本的全部端点序列命中 |
| 覆盖度 | query coverage >50% **或** hit coverage >50%；原文为严格大于，没有给出更底层的 gap/coverage 计数实现 | PPI 双方 ≥50%；DDI 仅训练域 ≥50% |
| 一端或两端命中 | **两端均命中**才排除整对；仅一端命中不触发这条规则 | 任一端命中就排除整对 |
| 完整搜索参数 | 未取得用于生成原始去同源名单的脚本/完整命令；不能推定版本、k、方向或预筛选设置 | MMseqs 18.8cc5c，双向 k=6 + 双向 prefilter-mode=2 |
| 聚类 | 单链 `--min-seq-id 0.3 -c 0.8 --cov-mode 0`，再按两个链簇组合构建互作簇；DDI 此前另有 50% identity 去冗余步骤 | 来源 70/70 TSV 的单条序列簇，用于采样 |

论文依据：[Science 补充方法 PDF](https://pmc.ncbi.nlm.nih.gov/articles/instance/13281779/bin/NIHMS2182373-supplement-supplemental_text_and_figures.pdf)，M3.1 第 20–21 页、M3.3 第 27 页；另核对了[正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13281779/)和[作者提供的正式论文 PDF](https://www.bakerlab.org/wp-content/uploads/2025/10/science.adt1630.pdf)。补充方法已归档为 `experiments/rf2ppi_isolation_review_20261024/supplementary_methods.pdf`，对应文本与 SHA256 同时归档。

设 H(x) 表示训练序列 x 在人类蛋白集合中至少有一个满足上述三个条件的命中，则 RF2-PPI 此条排除规则为 `H(A) AND H(B)`，不是 `H(A) OR H(B)`。补充方法没有要求两个目标人类蛋白必须构成某条已知 benchmark pair。它也不保证测试蛋白逐条与训练端点完全无同源，因此不能称为严格的 C3 隔离。PPI 与 DDI 正例的这套排除条件相同，不是双侧覆盖度与仅训练域覆盖度的分工。

源码依据：[作者仓库 data_loader.py，固定提交 ffc86527838cc1b75b75002c0831941d58d8dfe5](https://github.com/CongLabCode/RoseTTAFold2-PPI/blob/ffc86527838cc1b75b75002c0831941d58d8dfe5/src/data_loader.py)。加载器读取预先生成的 `list.nohomo.csv` 等文件；其中排除 PDB validation 子链的代码不等于生成 human test 隔离名单的实现。本次检查了公开归档的全部 Python/脚本文件，未找到生成该去同源名单的 MMseqs 命令。

此前 Science 下载入口返回访问错误、PMC 附件要求浏览器验证，导致本复核最初未取得补充方法；现已完成公开下载页面的 JavaScript 验证并取得 PDF，上述覆盖度、端点逻辑和 DDI 正例条件已据此更新。仍未确认用于生成原始去同源名单的完整 MMseqs 搜索命令/版本。DDI 负例段落没有像 PPI 负例那样单独明确重述去人类同源的步骤，且末尾重复了正例计数句，不能据此声称其具体实现完全核实。不得把其他用途（MSA 构建、序列映射或聚类）的参数冒充隔离搜索参数。

## 作者原始名单与我们的统计口径

本地来源：`/media/zlab/lexar4t/ppidb/sources/conglab/metadata/{ppi,ddi}/`。作者 README 明确将 `posi_nohomo.csv` 描述为排除人类近同源的名单；数据入口为[作者下载页面](https://conglab.swmed.edu/humanPPI/humanPPI_download.html)。

| 作者发布文件 | 行数 | 文件中的互作 cluster 数 |
|---|---:|---:|
| PPI `posi_all.csv` | 391,897 | 21,661 |
| PPI `posi_nohomo.csv` | 190,389 | 13,231 |
| DDI `posi_all.csv` | 585,078 | 237,919 |
| DDI `posi_nohomo.csv` | 535,115 | 223,440 |

这些数是本地作者发布文件的实际统计，并非推测；文件哈希见实验目录的 `author_release_counts.json`。论文正文与发布文件存在个别计数差异，不能以正文数量反推发布文件的成员。作者针对人类的 `nohomo` 名单不能直接替代我们针对 E. coli 的过滤。这里的互作簇也不能与我们之前报告的 1,043,395 → 404,405 个 DDI 单条序列簇比较。

## 固定抽样诊断

实验目录：[experiments/rf2ppi_isolation_review_20261024](../experiments/rf2ppi_isolation_review_20261024)。没有新增 `tools` 入口；诊断通过一次性命令执行，完整 MMseqs 参数已归档。

抽样总体为当前 v2 中符合以下条件的 **298,206 条 DDI 域序列**：在 DDI→test 的无预筛选比对中有合格命中，但两个方向的 k-mer 搜索和 exact 匹配均未产生合格命中。先按 `md5(seq_id || ':rf2-review-20261024')` 排序，再以 seq_id 打破并列，取前 1,000 条；选择过程没有使用 E-value。

沿用原 `ddi_train_to_test_nofilter.search.json` 的所有比对参数，只把 query 缩为这些抽样序列、将输出放入实验目录，并在输出列追加 `evalue,bits`。**target 保持原有全部 772 条 test 序列及原哈希**，没有将 target 缩为一个命中蛋白。MMseqs 完成后重新执行原有严格 identity 和域 coverage 整数判断。

| 检验 | 结果 |
|---|---:|
| 抽样域序列数 | 1,000 |
| 复现原方向命中的序列数 | 1,000 |
| 复现原合格比对记录数 | 6,838 |
| 相对原始证据缺少/新增的比对记录 | 0 / 0 |
| 至少一个命中 E-value <1e-5 的序列 | **4** |
| 无命中达到 E-value <1e-5 的序列 | **996** |
| 最好命中的 E-value 仍 ≥1 的序列 | 810 |
| 每条序列最好 E-value 的中位数 | 8.782 |

原始证据坐标、identity 分子分母及覆盖度全部一致，所以这里观察到的差异是额外显著性判定造成的，并非换了序列或改变了局部比对条件。MMseqs 的 E-value、coverage、identity 是不同的筛选维度，见[18.8cc5c 版本用户指南](https://github.com/soedinglab/MMseqs2/wiki/Home/67ba9c6637b4b5121a73e5de034dd0c3414d2b81#clustering-criteria)。

限制：这是特定新增命中子集的固定抽样，仅复核 DDI→test 方向。996 条未达显著性门槛不等于证明它们在生物学上没有同源；也不是 996 条训练 pair，更不能用 99.6% 推算整库可恢复比例。另一方向的证据、同一训练 pair 的另一端和全库多重比较仍需单独核查。补充方法现在确认 DDI 正例也使用 E-value <1e-5，但这次抽样仍不是完整 RF2-PPI 协议的重建。

## 后续应如何修订

1. 先做全量显著性诊断，保存 E-value、bits、比对长度、identity、双侧 coverage 和方向；按序列及训练 pair 分别报告保留数量和簇数量，避免把序列比例当作训练样本比例。
2. 修订前区分两种科学目标：严格任一训练端点隔离，或采用 RF2-PPI 的双端命中排除。后者不保证 C3。借鉴 `E-value <1e-5` 但保留任一端排除可以作为另一个明确命名的协议，但不能叫完整复现 RF2-PPI。**加入 E-value 或改变 coverage/端点逻辑都改变成员判定**；应先确认科学目标并报告数量变化，不能直接重写活动规则。尚未公开的完整搜索参数也必须标明为我们固定的实现选择。
3. 把可核实的论文规定、用户要求和我们自行选定的实现参数分列。确定修订协议后，仍改现有 build/check 两个入口，建立新版本与锁；保留当前 v2 及历史 v1 证据，不刷新旧锁绕过检查。
4. 对照必须包含强显著命中、边界命中、短弱匹配及未命中样本。通过检查意味着实际规则执行正确；规则是否适合科学问题需要独立论证，不能由构建脚本的 `passed` 自证。

本次仅提出上述修订方向，没有把它写入活动过滤配置，也没有重建成员。

## 归档与未变化验证

- `sources.json`：论文、作者代码提交、作者 README 的来源和哈希。
- `diagnostic_setup.json`、`sample_ids.txt`、`ddi_direct_only_sample.fasta`：固定抽样与完整命令。
- `sample_mmseqs.log`、`sample_alignments.tsv`、`diagnostic_result.json`：原始比对及诊断结果。
- `unchanged_split_hashes.json`：正式产物未变化。

正式 train SHA256：`9f6813487c16a2a3b8d6da4cc4d3502b08d28c19b86dd8cc21e53dcea7d9d05f`。

正式 test SHA256：`be5440cfcb2281c7d9d89bb172c865154e2b4cffb36928262a4e28cbd52c99f9`。

v2 协议锁 SHA256：`b8a3a133238ac99c09a817241d77fade696d91266b67b4c0fab9b373b10b5a4f`。
