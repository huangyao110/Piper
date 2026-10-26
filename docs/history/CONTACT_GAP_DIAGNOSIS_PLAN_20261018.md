# 接触预测与 FlashPPI 的差距：排查顺序

2026-10-20 后续执行：[B400 → B600 续训](CONTACT_GAP_B600_CONTINUATION_20261020.md) 已按用户要求启动，追加 200 步，沿用原常数学习率与优化器状态。当前结果和下一阶段判断仍以固定 val 为依据。

2026-10-19 更新。用户已批准执行。官方复测、32 条真实样本审计、16 对小样本拟合及三批 full16 梯度诊断已完成；两档学习率探针均通过，两组各 200 step 对照已完成。B200 的 val 残基 P@K 较高，A200 的互作 AP 较高，详见 [执行报告](CONTACT_GAP_DIAGNOSIS_RESULTS_20261018.md)。随后按用户要求完成 [B200 → B400 续训及 B400 完整 test](CONTACT_GAP_B_EXTENSION_20261019.md)：17:43 完成训练，20:50 完成 test。原生产源码和 checkpoint 保留；架构对照尚未执行。

## 目前能够确认的事实

同一 E. coli benchmark：650 正例、65,000 负例；全长、原始方向、无检索过滤。

| 指标 | 官方 noecoli 历史实测 | 本轮 final step 1616 |
| --- | ---: | ---: |
| Contact head 互作 AP | 0.295173 | 0.146702 |
| 残基 contact P@K，12 Å | 0.182312 | 0.102194 |
| 残基 contact P@K，8 Å | 0.107563 | 0.051598 |
| CLIP AP | 0.250091 | 0.244315 |

上表官方列为历史 comparison。现已补齐当天复测：Contact AP 0.295202、12 Å P@K 0.182276、8 Å P@K 0.107526、CLIP AP 0.250087，与历史值基本一致。两份预测的 65,650 行顺序、标签、长度和数据文件哈希已核对。

- 本轮 full16 只有 step 1119–1616，共 498 步。切换后的 backbone LR 从 2.37117e-6 衰减至 0，均值 8.19430e-7；全阶段 1616 步包含此前的 LoRA。
- 官方当前公开代码的 cosine 设置 `min_lr_rate=0.1`；研究 trainer 的 cosine 下限为 0。这是已确认的实现差异，不等于已证明它造成了性能差距。
- 当前 gLM2 配置有 33 个 block。我们的 full16 训练 block 17–32，并冻结其他 backbone 参数；官方代码冻结前 16 个 block。应枚举实际可训练参数，核对 block 16、末端 norm 和其他非 block 参数，不能只按“后半层”的文字判断等价。
- 官方优化器对名称匹配 bias、LayerNorm.weight、logit_scale、alpha、beta 的参数排除 weight decay；研究优化器对 backbone/head 两组统一应用 0.1。不要把名称匹配规则解释为官方自动排除了所有 norm。
- 本地训练集为 256,596 PPI + 570,709 DDI。为隔离 val，额外排除了 87,925 条 PPI（含 val）和 4,406 条 DDI。论文报告的最终训练清单与本地清单不同，作者逐条处理清单尚未取得。
- val 有 100 个结构；互作 AP 仅使用其中可可靠注释的 25 个正例及 203 个负例。该 AP 不适合承担细微差异的唯一选模依据。
- 当前模型继承了我们的 patch/LoRA 实验权重；原始初始化使用 gLM2 UR50 和新任务头，并未加载官方训练好的 FlashPPI 任务权重。
- step 1102 的梯度诊断针对 LoRA；它不能直接代表当前 full16 的任务梯度关系。

## 1. 对齐评估与训练语义

先用官方 `flashppi_noecoli` 在固定数据、统一指标函数下复测，核对历史结果的差异；官方复测是基准核验，不用于选研究模型参数。对两模型记录模型/数据/源码哈希、精度、方向、序列长度、预测行身份。

抽取固定训练 PPI/DDI 各 16 条进行逐项检查：

- 三个编码器 token 与残基的索引对应，包括 BOS/EOS、padding、罕见残基、mask 和裁剪偏移。
- 标签接触阈值、坐标映射和可观测区域；缺失结构位置不计负例。记录裁剪前后真实接触数及其保留比例。
- 输出 `[B,L1,L2]` 是否覆盖有效残基；无效位置是否进入 loss 或 max contact score。
- 对固定 logits 独立重算 focal 正接触、结构内非接触、额外负蛋白对贡献，核对分母、microbatch 权重及梯度。
- 统计 microbatch 16 与官方配置 64 的分组归一化差异。Focal 当前按每个 microbatch 的正接触数归一化，全局 batch 相同不保证两者的 contact 梯度相同；固定 logits/负例再比较，避免把负采样变化混进去。
- 分别核对 hard negative、已知正例屏蔽、唯一异源负例与官方 self-negative 的差别。先记录差异，不同时修改。
- 枚举实际可训练参数和 optimizer 参数组，并测量一次独立更新的 `||Δθ|| / ||θ||`；有梯度不代表实际有足够更新。探针更新只在副本中执行。

如果发现标签、mask、归一化或更新错误，先修正并通过针对该错误的检查，再进行下面的训练实验。

## 2. 固定少量样本能否学会

从训练集选 16 个可观测接触充足的短结构（8 PPI + 8 DDI），固定样本、裁剪和标签。保留当前融合/gate 的结构与参数，从相同 checkpoint 的副本开始。

1. 关闭随机 sequence masking 和随机裁剪；冻结编码器、融合与 gate，缓存送入 contact head 的特征，仅训练 contact head。只使用结构内有效标签，暂不加入 CLIP 或额外负蛋白对。
2. 重复约 100–200 次便宜的 head 更新，记录固定样本的训练 P@K、正接触分数和分项 loss。缓存只用于这个明确冻结编码器的诊断，不能跨 step 复用可训练 gLM2 的 embedding。
3. 若不能明显拟合，缩至 1–2 对，并检查标签冲突、特征、梯度和更新幅度；短预算内未拟合本身不能证明存在代码 bug。
4. 能拟合后，分别从相同起点恢复额外负例、CLIP 和随机增强，逐项观察。需要诊断共享 backbone 时再解冻 full16，不能用“head-only 能拟合”证明全模型训练正常。

小样本高 P@K 只证明训练链路有拟合能力，不是泛化成绩。该诊断不改变正式门控设计。

## 3. 重新测 full16 的损失与梯度

固定 val 接触指标选出的 step 1500，使用 3 个固定正式规模训练批次，batch 1024 / microbatch 16。只前反向，不更新正式参数。

- 分解真实接触、结构内非接触、额外负例的 loss；同时报告样本数、有效位置数和正接触数。
- 分别计算 Contact 与乘权重 0.1 后 CLIP 对 full16 每层、融合、gate、两个 head 的梯度范数及余弦。
- 记录联合梯度与真实 AdamW 更新方向；两任务余弦为负只是局部证据，不能单独判定应删除 CLIP。
- 现有 `diagnose_contact_objectives.py` 把 `plm.*` 组标为 `shared_lora`，复用时需更名为 backbone 并核对内存预算，避免将 dense 权重梯度误报为 LoRA。

只有当前 full16 也表现出持续冲突，才增加只调整回传 backbone 的 CLIP 梯度权重的单变量对照。

## 4. 最小学习率对照

前面通过后，才启动两组短训练。共同起点为 **step 1500**，它按 val 残基 P@K 选出，非按 test 选出。

| 设置 | A | B |
| --- | ---: | ---: |
| backbone 峰值 LR | 1e-5 | 3e-5 |
| head/fusion/gate LR | 1e-4 | 1e-4 |
| 新阶段 warmup | 20 step | 20 step |
| warmup 后 | 保持本阶段 LR | 保持本阶段 LR |
| 检查点 | 0 / 50 / 100 / 200 | 0 / 50 / 100 / 200 |
| 最大初筛预算 | 200 step | 200 step |

这些 LR 是待检验候选，不是已证实的最优值；先做有限步稳定性探针。历史上 patch 小批量直接使用 head 5e-4 曾出现剧烈 loss 增长，不能直接照搬论文 LR 到当前续训。

两组继承相同权重和 AdamW 状态，使用相同新阶段样本序列、裁剪、mask seed、负例规则、数据和实际可训练参数。仅 A/B 的 backbone LR 不同。正式 gate、ESMC/ProtT5 冻结状态及 loss 权重不变。

这属于有意开启新阶段，不能直接修改旧配置后冒充原生 `--resume`。实现时需独立工作树、显式阶段迁移、原始来源与配置差异记录；严格恢复校验继续保留。

A/B 相互比较能判断 backbone LR 的影响；它们相对旧 checkpoint 的改善同时包含新训练预算和 head 学习率重启，不能全部归因于 backbone LR。

主要看固定 val 的残基 P@K，辅以 Contact 互作 AP、CLIP AP、固定训练样本 loss、梯度和参数更新幅度。按结构/同源簇做配对差异分析；100 step 只初筛，区分度不足则两组都到 200，不只延长早期偶然领先者。小 val 的微小改善不能宣称确定胜出。

当前正式步耗时约 116.4 秒，因此 100 step 约 3.23 小时/组，200 step 约 6.47 小时/组，尚不含验证与保存。先完成便宜的正确性检查和拟合诊断，再决定是否值得投入这个预算。

## 5. 区分训练问题与架构收益

当训练过程能有效更新后，才安排匹配的架构对照：

- 原版单 gLM2。
- 单 gLM2 + 当前 gate。
- gLM2 + ESMC/ProtT5 + 当前 gate。

使用相同原始 gLM2 初始化、匹配的原版 head 初始权重、数据划分、样本预算、负例策略、优化器和调度；新增残差分支初始应保持基线函数，记录所有差异。gate 的机制在相应实验中保持不变。

当前融合 checkpoint 已经过共同适配，直接关闭分支只能测试即时敏感性，不能当作从头训练的消融，也不能据此证明某个编码器无用。

若单 gLM2 在本地数据上也明显达不到官方，再优先排查数据覆盖、采样、增强和训练预算；若原版能达到而 gated 模型不能，才将重点放到融合/gate 或任务耦合。官方最终训练清单不可得时，称为“本地数据上的匹配基线”，不宣称精确复现论文训练。

后续参数选择使用预先固定的 val；官方 test 留给确定方案后的报告。此前已多次查看 test，若需要独立泛化结论，另保留未参与设计决策的确认集。

## 证据与入口

- [官方历史结果](../runs/lora_bs1024_micro16_interface_v2_torch210_resume1700_01/official_ecoli_step_002150/comparison.json)
- [本轮最终 test](../experiments/esmc_prott5_glm2_residue_gated_v1_20261015/test_ecoli_full16_final/metrics.json)
- [数据划分 manifest](../data/conglab_ecoli_v1/manifest.json)
- [数据来源与过滤审计](CONGLAB_ECOLI_FRESH_LORA_20261013.md)
- [此前 LoRA 梯度诊断](CONTACT_OBJECTIVE_DIAGNOSTICS_20261017.md)
- [full16 生产记录](GLM2_FULL16_PRODUCTION_20261017.md)
- [当前训练配置](../experiments/esmc_prott5_glm2_residue_gated_v1_20261015/manual_full16_transition_20261017_2229/migrated_state/config.json)
- [官方 train.py](https://github.com/TattaBio/FlashPPI/blob/main/flashppi/train.py)
- [官方训练配置](https://github.com/TattaBio/FlashPPI/blob/main/configs_train/flashppi.yaml)
- [论文](https://www.pnas.org/doi/10.1073/pnas.2610619123)

当前生产源码：`/home/zlab/.codex/worktrees/glm2-full16-production-20261017/FlashPPI`。这些旧源码保持可复现，后续实验在独立副本中实现。
