# 数据过滤的固定约束

## 当前：RF2-PPI 隔离规则 v3（2026-10-24 用户明确授权）

用户要求“严格采用 RF2PPI 的隔离方式，换入之前的脚本，重新生成 train.index”。本次采用 `conglab_distilled_ecoli_rf2ppi_v3`；历史规则与锁不覆盖。维护入口仍为 `tools/train_test.py`（build/check）和 `tools/audit_expanded_index_split.py`。

2026-10-24 用户随后要求删除 v2 和过时目录，已删除 v2 正式输出、未发布草稿及退役的 `expanded_conglab_pinder_teddymer_v1`，释放约 7.39 GiB。下文关于这些目录“保留”的表述属于历史记录，已被此次授权取代；配置、协议锁及数量对照等小文件保留。具体清单见 [清理记录](DATA_CLEANUP_20261024.json)。原始 v1、仍被历史配置引用的有效扩充目录与当前 v3 不在此次删除范围。

| 项目 | v3 固定标准 |
|---|---|
| 单序列命中 | identity >30% AND MMseqs E-value <1e-5 AND（query coverage >50% OR hit coverage >50%） |
| PPI / DDI | 使用相同命中条件；不再分别使用双侧覆盖和训练域覆盖 |
| 边界 | 均严格不等式：30%、50%、1e-5 恰好等于阈值时不满足该条件 |
| 配对排除 | 两个训练端点分别命中参考集合才排除整对；只有一端命中允许保留；不要求对应同一 test pair |
| 完全相同序列 | 必须通过同样的显著性条件，不使用自动纳入命中的旁路 |
| 参考 / 来源 | 既有 E. coli test 正负样本两侧全部序列；无 val；CongLab PPI/DDI + 8 个蒸馏 PPI 来源 |
| 序列与采样 | 来源 FASTA 条目裁剪前判定；继续核验与使用已有 70/70 序列聚类 TSV，不更改采样簇标准 |
| 输出 | `data/conglab_distilled_ecoli_rf2ppi_v3/train.index.parquet`，另有冻结 test 索引、完整证据、去重映射与 README |

论文依据为已归档的补充方法 M3.1 第20–21页和 M3.3 第27页，见 [RF2-PPI 复核](RF2PPI_ISOLATION_REVIEW_20261024.md)。这些段落明确了 identity、E-value、coverage OR 和 pair AND；参考集合按本项目要求使用 E. coli。原作者完整 MMseqs 搜索命令/版本未公开，不能宣称逐参数复现。

本地实现固定 MMseqs 18.8cc5c 与二进制 SHA256、8线程、双向 k=6 与双向 prefilter-mode=2、完整目标库、顺序和全部显式参数。`-c 0` 避免提前错误应用 coverage AND，输出后用整数跨度严格判断 OR；`-e 0.00001`，输出含 evalue/bits，最终再次严格筛 E-value。完全相同序列也走比对。每个方向的 E-value 使用对应完整目标库，不缩小数据库来增强显著性。上述参数均在独立 v3 锁中，不复用旧成员判定。

本地实际运行发现该 MMseqs 版本对部分全 A 序列输出零残基比对记录，其显著性数值未初始化。这类记录不构成比对，构建器和独立审计器均拒绝将其算作命中，并分别统计数量；其他无效坐标仍报错。修复未修改搜索参数、命中阈值或来源质量筛选。源码核对、最小复现与失败构建记录见 [空比对诊断](../experiments/rf2ppi_isolation_review_20261024/mmseqs_zero_alignment/README.md)。

独立验收重新读取全量原始比对并自行实现数值判定，核查全部排除成员和最终 train 的双端命中为零；允许的单端命中另行计数。build 构建后自动验收；check 不构建、不修改成员或锁。此标准不保证 C3，也不重新定义为盲测。

```bash
PYTHONPATH=artifacts/environments/data_index_deps \
  artifacts/environments/esmc_prott5_gated_py312/bin/python \
  tools/train_test.py --mode build \
  --config configs_data/conglab_distilled_ecoli_rf2ppi_v3.json
```

复验使用同一命令，将 `--mode build` 换为 `--mode check`。

## 以下为历史 v2 记录

2026-10-24 RF2-PPI 对照复核发现当前 v2 的显著性筛选有明显风险：固定抽样的 1,000 条新增 DDI 域命中中，仅 4 条在 DDI→test 方向达到正文提及的 E-value <1e-5。详情见 [RF2-PPI 隔离复核](RF2PPI_ISOLATION_REVIEW_20261024.md)。原文中“直接比对确认的同源”应理解为“符合当时 identity/coverage 的比对”，不能等同于确认生物学同源。v2 验收通过仅说明执行了锁定规则；不能据此宣称完整复现 RF2-PPI 或已排除过筛风险。本次复核不改变下列规则、现有成员和锁文件，修订仍须另行明确与版本化。

## 2026-10-24：用户明确授权的新 train–test 协议

新协议 `conglab_distilled_ecoli_strict_v2` 使用 `configs_data/conglab_distilled_ecoli_strict_v2.lock.json`，通过**原有** `tools/train_test.py` 构建、`tools/audit_expanded_index_split.py` 验收。没有新增构建脚本，也不覆盖下面记录的历史 v1。

| 项目 | 新 v2 固定规则 |
|---|---|
| Train | 2026.10.3 发行包的 CongLab PPI/DDI + 8 个预测 PPI 来源，按 `structure_source` 选择；不含 PINDER/Teddymer |
| Test / val | 固定 E. coli 65,650 条（650 正、65,000 负），内容、顺序、标签不变；不设 val |
| 参考集合 | test 正负样本两侧**全部**序列，排序后固定 FASTA 与 SHA256 |
| 过滤序列 | pairs CSV 对应的来源 FASTA 条目全长；逐条核对 Parquet ID/序列。先过滤后训练 crop |
| 序列含义 | FASTA 可能是来源结构片段/结构域，不擅自替换成推测的 UniProt 全长；本协议隔离实际来源输入序列 |
| Identity | `nident*10 > alnlen*3`，恰好 30% 不满足 |
| Coverage | PPI 双方比对区间各覆盖完整 FASTA 长度 ≥50%；DDI 仅要求训练域 ≥50%，恰好 50% 满足 |
| 端点规则 | 任意训练端点命中任意 test 序列，排除整对；全部来源重新检查，无旧成员豁免 |
| MMseqs | 18.8cc5c + 二进制 SHA256；双向 k=6 检索 + 双向关闭预筛选全量比对（prefilter-mode=2）并集 + 完全相同序列；k-score=96、s=7.5、mask=0、shuffle=0、8 线程、split=1、split-mode=0、16G |
| 搜索边界 | PPI cov-mode=0；DDI test→train 用 1、train→test 用 2。max-seqs=目标库全部序列数，其他参数见锁文件 |
| 聚类 | 核验现有 PPI/DDI 70% identity、70% coverage 序列 TSV，仅用于采样，不代替过滤；DDI 本轮也按序列簇均衡 |
| 去重 | 相同序列、contact、mask 才合并，支持链交换；不同监督分别保留。每序列对组内总权重=1 |

发布前用历史争议样本复查发现：双向 k=6 对 1,271 条已由直接比对确认的 test 同源域仍漏掉 391 条。因此最终锁定方案强制加入**全部入选序列的双向无预筛选比对**，不能只复查已有命中或争议样本。最初仅完成来源目录阶段的草稿未发布 train 索引，保留在 `/home/zlab/flashppi_data/conglab_distilled_ecoli_strict_v2_unpublished_draft_20261024`。

全量直接比对避免 k-mer 预筛选漏检；仍不宣称不存在所有远缘生物学同源、结构相似性或基础模型预训练暴露。多线程原始 TSV 行序可变，最终命中集合与索引排序固定。历史 test 已多次评估，不能重新称为盲测。更换输入版本、搜索或边界必须新版本，不能更新已有输出的锁绕过校验。新构建输出 `data/conglab_distilled_ecoli_strict_v2`；临时计算文件放 `/home/zlab` 盘。生成索引不启动训练。

```bash
PYTHONPATH=artifacts/environments/data_index_deps \
  artifacts/environments/esmc_prott5_gated_py312/bin/python \
  tools/train_test.py \
  --mode build \
  --config configs_data/conglab_distilled_ecoli_strict_v2.json
```

同一入口还提供独立检查模式：

```bash
PYTHONPATH=artifacts/environments/data_index_deps \
  artifacts/environments/esmc_prott5_gated_py312/bin/python \
  tools/train_test.py \
  --mode check \
  --config configs_data/conglab_distilled_ecoli_strict_v2.json
```

`build` 完成隔离并自动验收；`check` 要求已有完整划分，不构建、不改变 train/test 成员。检查器独立重算归档原始比对的 identity/coverage 和任一端点排除条件，核对全量搜索的参数、输入及日志、正式索引与源文件哈希、test 顺序/标签及去重权重。它不只读取既有的 passed 标记，也不重新运行 MMseqs。证据不完整、配置不同或发现违规均返回非零退出码；最新检查状态写入 `check_status.json`，通过后的详细结果写入 `independent_audit.json`。检查允许更新验收元数据，不允许自动修复成员或更新协议锁。

## 历史 v1（以下规则仅用于历史划分与其原协议扩充）

依据用户 2026-10-22 的明确要求：**必须使用旧的过滤条件。核心数据过滤脚本与条件不得未经用户同意改变。**

本轮唯一基准是 `data/conglab_ecoli_v1`，协议标识为 `conglab_ecoli_v1_locked`。`configs_data/conglab_ecoli_filter.lock.json` 固定原始脚本、配置、命中记录、划分文件及其 SHA256。增加数据、清理文件、加速、诊断出更多相似序列，都不是修改过滤协议的授权。

## 必须保持的规则

| 项目 | 旧条件，必须保持 |
|---|---|
| 序列一致性 | `nident / alnlen` **严格 > 0.30**；恰好 30% 不通过 |
| PPI 覆盖度 | 比对区间对 query 和 target 的覆盖度都**严格 > 0.50** |
| DDI 覆盖度 | 比对区间覆盖训练结构域长度 **≥ 0.80**；不要求覆盖完整参考蛋白 80% |
| test 排除 | 训练样本**两端分别命中** test 序列集合才排除；不要求对应同一 test pair |
| val 排除 | 训练样本**任意一端命中** val 序列集合即排除 |
| 参考序列 | 冻结的 test/val 文件中的原始序列集合；先过滤，后训练裁剪 |
| 已检索过的序列 | **复用原命中与未命中判定**；扩库结果不得重新判定这些序列 |
| 旧训练成员 | CongLab PPI **256,596** 条、DDI **570,709** 条全部保留 |
| 评估集合 | 原 val 接触 100 条、val 互作 228 条、test 65,650 条，文件、顺序、标签和哈希固定 |
| 额外条件 | 不擅自加入 RF2、PINDER 官方划分、pLDDT/PAE、全长链重建、聚类传播或更严格的端点排除条件 |

原判定函数仍是 `tools/audit_conglab_ecoli.py:classify_hit`，扩充构建直接调用该函数。保留以下四个原脚本及其字节级哈希，不改写其中的过滤与聚类算法：

- `tools/audit_conglab_ecoli.py`
- `tools/prepare_conglab_ecoli.py`
- `tools/prepare_conglab_mixture.py`：原流程依赖的辅助函数。
- `tools/prepare_ppi_finetune_split.py`：原流程依赖的 MMseqs/Policy 实现。

后两者自身的历史命令行不代表本轮允许切换到另一个划分协议。

## 实际搜索条件也固定

原 PPI/DDI、test/val 四次检索实际均使用：

- MMseqs **18.8cc5c**，参考序列为 query、候选蛋白/结构域为 target。
- `k=6`、k-mer score **96**、`-s 7.5`、`--mask 0`、`--shuffle 0`。
- `--min-seq-id 0.3 -c 0.5 --cov-mode 1` 产生候选比对，再由原整数判定函数执行上述严格边界。
- `--alignment-mode 3 --seq-id-mode 0 -a 1 -e 1000000`。
- `--max-accept 2147483647 --max-rejected 2147483647`；`--max-seqs` 等于实际候选序列数，避免固定 top-N 截断。
- 8 个线程、16G split-memory-limit；test、val 分开检索。

旧命令没有显式给出 `-k`，程序自动选出了 6。对新增数据显式固定 6 和 96，是落实本次用户要求，防止扩库后自动变为 7。每次检索必须检查实际日志，不允许仅凭配置名宣称沿用了旧条件。

旧序列的判定依据原缓存。构建时从只读来源重建旧候选序列范围，并验证 FASTA 哈希与历史完全一致；只在确认序列当时确实进入检索后，才可以把“未在命中列表中”解释为旧未命中。原 val 范围还原 test 排除和有效接触检查后的候选集，不能直接把所有旧序列都视为经过 val 检索。

新增序列使用旧命令和上述固定参数。MMseqs 仍是启发式搜索，目标库上下文也会影响结果；本协议保证旧判定不漂移，不宣称所有潜在同源都已排除。

## 诊断与正式划分

2026-10-22 的 k=7 / 关闭预筛选复核发现了旧流程未命中的样本。相关证据保留在 `experiments/homology_consistency_20261022/`。

**这些诊断不能自动替代旧过滤条件，不能自动增加黑名单，也不能删除旧训练样本。** 本次按照用户指令固定旧协议。保留旧协议不等于撤销已经观察到的相似性证据；论文或报告不能将本训练描述为穷尽比对确认的完全同源隔离。

原 `data/expanded_conglab_pinder_teddymer_v1` 使用自动 k 的扩库索引已退出当前流程，其配置从活动目录移除；历史数据和审计证据保留。当前重建入口是：

```bash
PYTHONPATH=artifacts/environments/data_index_deps \
  artifacts/environments/esmc_prott5_gated_py312/bin/python \
  tools/train_test.py \
  --config configs_data/expanded_conglab_pinder_teddymer_legacy_v1.json
```

输出为 `data/expanded_conglab_pinder_teddymer_legacy_v1`。仍只生成索引、去重映射、过滤证据和说明，训练数据本体通过索引读取。

## 修改纪律

改变上述任一条件，必须先向用户列明具体参数、原因、受影响样本和对照结果，取得明确同意后再实现，并使用新的协议版本与输出目录。禁止自动刷新锁文件、覆盖旧缓存、静默复用不同协议的缓存或删除历史证据来绕过校验。

常规实现修复和脚本清理可以在既定授权内执行，但必须通过原规则边界、旧成员保留、缓存完整性与实际搜索参数检查。训练启动不是数据构建的自动副作用。
