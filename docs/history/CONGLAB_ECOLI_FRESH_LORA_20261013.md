# CongLab / E. coli：从原始 gLM2 初始化的 patch 对照

> **2026-10-13 调度更新：** 用户取消对照训练，改为只训练 motif 版本 2 个 epoch。
> 原残基对照已在 step 11 保存并停止，原队列中的 patch 100-step 任务已取消。
> 当前方案、参数和启动入口见 [motif 两轮训练记录](CONGLAB_MOTIF_2EPOCHS_20261013.md)。
> 下文中的 100-step 两组计划保留为历史记录，不再执行；数据审计和 GPU 预检仍适用。

## 固定实验范围

- 只使用 `structure_source == conglab` 的结构，PPI/DDI 每个全局批次各 512 对。
- 官方 E. coli benchmark 不修改：650 正例、65,000 负例。此次不排除 RF2-PPI benchmark；不要把模型当作 RF2 隔离模型使用。
- 用户选择按论文正文的“两条链均命中才排除”处理 test；train/val 仍按任一链命中排除。
- 重新划分小 val，旧的 500 对不再整体作为排除集。
- 外部数据库保持只读，所有输出写入本仓库 `data/`、`artifacts/` 和 `experiments/`。

## 数据与论文能对应到哪一层

逐项比较本地发布目录的 `original_id` 和原始 CongLab `posi_all.csv`：PPI 391,897 条、DDI 585,078 条，均无缺失和额外原始结构 ID。它们是结构对记录数，不是无序序列对数。

| 阶段 | PPI | DDI |
|---|---:|---:|
| CongLab 原始正例 | 391,897 | 585,078 |
| 此次 test 过滤后 | 345,160 | 575,124 |
| 再剔除没有有效接触的记录 | 344,521 | 575,115 |
| 再隔离新 val 后用于训练 | 256,596 | 570,709 |
| 论文报告的训练数量 | 340,367 | 529,143 |

**只能确认原始来源完全对应，不能确认作者最终训练清单逐条对应。** 作者处理后的训练库目前不可公开下载；数量相近也不能替代样本 ID 比对。论文正文和方法对“both sequences”与“all sequences”的描述不一致。审计同时记录了两种排除方式及不同覆盖率定义的数量，没有通过调整阈值或随机删样本凑论文数量。

固定规则：PPI 序列一致性 >30%，比对覆盖双方完整序列均 >50%；DDI 一致性 >30%、覆盖候选结构域 ≥80%，避免要求短结构域覆盖完整测试蛋白。后者沿用此前明确的结构域过滤原则，**是与论文简单 30%/50% 描述的一个差异**。MMseqs2 18.8cc5c、敏感度 7.5、e-value 1e6；精确重复另行排除，阈值用比对整数而非四舍五入后的百分数判断。搜索为 held-out→candidate 的启发式搜索，不等于穷尽全部远缘同源关系。

来源：[FlashPPI 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13291599/)、[CongLab 原始下载说明](https://conglab.swmed.edu/humanPPI/humanPPI_download.html)。完整结果在 `data/conglab_ecoli_audit_v1/audit.json`。

## 小 val 与采样

- 100 对 CongLab 结构，按 30% identity / 80% 双侧 coverage 聚类，对无序 cluster pair 去冗余；选择可复用可靠注释的样本及较低冗余家族。它不是训练分布的随机无偏样本。
- 全部 100 对评估接触；互作 AP 只用可可靠映射子集 25 正例、203 同物种负例，覆盖率 25%。复用已固定的 STRING 12 physical ≥700 排除结果，负例端点必须来自选中的正例；不会加入跨物种容易负例。
- 这些负例是未观察到的互作，不是实验确证阴性。AP 子集小，变化需结合接触指标判断，不宜过度解读。
- 为保证 val 隔离，额外移除了 87,925 条 PPI（含所选 val）和 4,406 条 DDI。train/val 精确序列重叠为零。
- 剩余 PPI 重新做 70/70 聚类：15,144 个 cluster，均匀选 cluster 再选样本；DDI 均匀抽样。DDI 的既有 70/70 聚类用于已知正例屏蔽，与 PPI namespace 分离，仅通过完全相同序列合并。联合 cluster 编号乘数为 1,058,538。

## 模型与训练

- 加载原始 `tattabio/gLM2_650M_UR50`，revision `a5937fbb770bd8bae65e3034aacd823b041dcba3`；不读取 FlashPPI checkpoint 张量。
- 当前下载配置实际有 33 层；最后 16 层（零起始索引 17–32）的 Q/V 添加 LoRA，rank 8、alpha 16、dropout 0.05。基础权重全部冻结。
- CLIP 和 contact 任务头用仓库训练构造器重新初始化；contact 输出 bias = -3。两组使用相同随机种子和初始张量。
- 残基头为对照；实验组在 contact Transformer 前对 contextual residue embeddings 做掩码感知的均值池化，窗口 16、步长 4，末端保留不完整窗口。两组先都保留 2 层 contact Transformer。
- patch label：窗口对中有任意有效接触即为正；只有所有真实残基对均可观测且没有接触才为负；坐标缺失不能当负例。原始接触定义不变。
- LoRA learning rate = 1e-5，任务头 = 1e-4；AdamW weight decay 0.1，5 steps warmup + cosine；0.1 CLIP + 1.0 Focal contact loss。训练残基以 15% 概率独立 mask，评估不 mask。
- 全局 batch 1,024，microbatch 16，最大每链 512 残基，界面感知 crop。先每组运行 100 steps 的可行性对照，step 0/50/100 做 val；这不代表训练已收敛，也不是两轮完整训练。
- GPU 预检必须验证原始 backbone forward、实际 adapter 层、梯度、冻结权重、优化器恢复、相同初始状态和显存；通过后顺序启动两组。最终只评估官方 E. coli test。

## 固定入口与产物

Python：`artifacts/environments/flashppi_torch210_py311/bin/python`。

```bash
python tools/audit_conglab_ecoli.py --config configs_data/conglab_ecoli_audit_v1.json
python tools/prepare_conglab_ecoli.py --config configs_data/conglab_ecoli_v1.json
```

准备脚本只保留最终 parquet、审计匹配 ID、日志和 manifest；FASTA 和 MMseqs 中间数据库自动清理。已存在的训练 parquet 不覆盖。

- 数据：`data/conglab_ecoli_v1/{train,validation,interaction}.parquet`
- 参数：`configs_train/conglab_ecoli_{residue,patch16_s4}_fresh_lora_v1.json`
- 实验：`experiments/conglab_ecoli_fresh_lora_v1_20261013/`
- 新初始化：`flashppi/glm_initialization.py`
- 原微调试验已停止，记录保留，但不继承其参数或优化器状态。

## 学习率预检记录

原始 gLM2 权重 SHA256 为 `0ebe7894868bebbe5859f4ccccf38eec1a9f2e8b84b811992c64dfa14b8c29c2`。GPU 上原生 gLM2 与优化实现的测试输出最大绝对误差为 0；LoRA 确认只在最后 16 层。

第一次预检以任务头峰值学习率 5e-4 直接执行两个 32 对训练步；patch contact loss 从 0.7305 升至 242.906，梯度范数 2500.6，不能仅因有限值而当作稳定。正式 warmup 会降低初始学习率，但这里仍将两组任务头峰值统一降到 1e-4 后重新检查。第一次预检结果保留在 `preflight_head_lr5e4/`，不参与正式训练。

1e-4 复检通过：残基头 contact loss 为 0.8986→0.9276；patch 为 0.7305→0.7020，梯度范数 2.744→6.486。训练峰值显存分别 15.387 GiB、8.000 GiB。两组初始训练张量哈希一致，基础权重保持不变，adapter 有更新，参数与优化器恢复均逐值一致。预检全局批量为 32（两次 microbatch 16）；正式训练另验证全局 1024 和 512:512 的类型比例。这是运行正确性检查，不证明后续长期收敛。

启动入口：

```bash
python tools/run_motif_experiment.py \
  --output experiments/conglab_ecoli_fresh_lora_v1_20261013 \
  --residue-config configs_train/conglab_ecoli_residue_fresh_lora_v1.json \
  --patch-config configs_train/conglab_ecoli_patch16_s4_fresh_lora_v1.json
```

启动器要求通过原始 backbone 检查和与当前配置哈希一致的 GPU 预检。两组顺序执行；任一步失败即停止，不自动换设置重跑。两组结束后评估各自 step 100 的官方 E. coli，输出 `test_ecoli_comparison.json`。val AP 的正负比为 25:203，不能直接与 E. coli 650:65000 的 AP 数值比较。

首个正式全局 step 已完成：1,024 对正例，其中 PPI/DDI 各 512；72.39 秒，峰值 allocated 16.08 GB。CLIP loss 7.6733、contact loss 0.7949、梯度范数 0.8649，mask 比例 14.986%。这是 step 1 的运行记录，不是 val/test 性能结论。
