# 接触预测差距排查：执行记录

2026-10-20 18:50：按用户要求已启动 [B400 → B600 续训](CONTACT_GAP_B600_CONTINUATION_20261020.md)，保留 `3e-5 / 1e-4` 恒定 LR、Adam 状态及全部门控，每 50 步验证。下文为已完成的对照及 B400 test 结果。

2026-10-19 20:51 更新。两组各 200 step 对照及 **B200 → B400** 续训均已完成。用户随后要求完整 test：B400 的 Contact 互作 AP 为 0.199554、12 Å 残基 P@K 为 0.140650、CLIP AP 为 0.264520，分别高于上次最终 step 1616 的 0.146702、0.102194、0.244315；CLIP AUROC 则从 0.862234 降至 0.853630。详见 [B 续训与 test 记录](CONTACT_GAP_B_EXTENSION_20261019.md)，实验状态见 [status.json](../experiments/contact_gap_diagnosis_20261018/status.json)。以下保留已完成的 A/B 同预算结果。

## 200-step 最终结果

下表全部是固定 val 的结果。共同起点为原 checkpoint step 1500，A/B step 均指新增阶段步数。

| 指标 | 共同起点 | A 200，backbone LR 1e-5 | B 200，backbone LR 3e-5 |
| --- | ---: | ---: | ---: |
| 残基 contact P@K | 0.106058 | 0.110226 | 0.114963 |
| Contact 互作 AP | 0.563335 | 0.607724 | 0.574959 |
| CLIP AP | 0.560645 | 0.605645 | 0.590783 |
| Contact AUROC | 0.869458 | 0.878128 | 0.862167 |

B 的残基 P@K 比 A 高 0.004737（相对 +4.30%），比共同起点高 0.008904（相对 +8.40%）。B 在 step 50/100/150/200 的 P@K 为 0.109729 / 0.111123 / 0.113655 / 0.114963；A 在同样步数为 0.110438 / 0.108746 / 0.110060 / 0.110226。按预先指定的残基 P@K 主指标，B 200 是本轮候选；A 组自己的最高 P@K 仍在 step 50。

在这 100 个 val 结构上，B200 − A200 的配对 bootstrap 95% 区间为 [0.001821, 0.007884]，B200 − 起点为 [0.002138, 0.015770]。这些区间支持本次固定 val 上的定位改善，不包含训练随机种子的方差。互作 AP 使用的样本仅 25 正例和 203 负例，也不宜据此夸大微小差异。

| 训练 loss 均值 | A 前 20 步 | A 最后 20 步 | B 前 20 步 | B 最后 20 步 |
| --- | ---: | ---: | ---: | ---: |
| Contact loss | 0.361250 | 0.339575 | 0.361491 | 0.328525 |
| CLIP loss | 1.146583 | 1.074737 | 1.154798 | 1.034924 |

相同 step 181–200 的样本计划下，B 的平均 Contact loss 比 A 低 3.25%，CLIP loss 低 3.70%。前后窗口的批次不同，不能把窗口均值当成固定样本的精确收敛曲线；模型变化也会影响 hard-negative 选择。

本轮说明：提高 backbone LR 对当前配置的残基定位有收益，但并未让所有指标一起领先。A200 的 Contact AP 和 CLIP AP 高于 B200。相对旧起点的提升同时包含新增训练预算及共同的 head LR 重启，不能全部归因于 backbone LR，更不能据此认定已解决与官方模型的差距。

证据：[完整配对比较](../experiments/contact_gap_diagnosis_20261018/lr_comparison/comparison_000200.json)、[比较表](../experiments/contact_gap_diagnosis_20261018/lr_comparison/comparison_000200.md)。两组 checkpoint 均保留；A/B 对照期间没有新增 test。B400 训练完成后的单次 test 另行记录，后续架构对照尚未执行。

## 已完成的检查及其含义

| 检查 | 实测结果 | 能够得出的结论 |
| --- | --- | --- |
| 官方 noecoli 模型复测 | Contact AP 0.295202；残基 P@K 0.182276 | 历史基准可以复现，差距真实存在 |
| 32 条真实训练样本审计 | PPI/DDI 各 16 条；裁剪、有效结构区域、三编码器残基对齐及 focal 重算通过 | 抽查未发现标签错位、padding 或未知结构位置进入 loss 的问题 |
| 固定 16 对，仅训练 contact core | P@K 0.434762 → 1.000000；loss 0.247628 → 0.000420 | 接触头具备拟合这些样本的能力 |
| 同起点、加入固定额外负例 | P@K 0.434762 → 1.000000；loss 0.338167 → 0.000639 | 固定负例没有阻止这组小样本拟合 |
| full16 三个正式规模批次 | backbone 任务梯度余弦 +0.0347 / +0.0562 / −0.0259 | 尚不支持持续强梯度冲突的判断 |
| microbatch 归一化重分组 | 固定预测和负例时，16 对归一化相对 64 对总 loss 高 0.18%–0.55% | 存在加权差异，但这三批未出现总 loss 倍数异常 |

小样本拟合使用训练集的 8 PPI + 8 DDI，序列长度 32–256、可观测接触充分；编码器、融合、gate 和 CLIP 全部固定，仅训练 contact core 200 次。两种条件分别从相同权重开始，临时权重不用于后续训练。接近 1 的成绩是固定训练样本的记忆能力，不能作为泛化成绩。

证据：[样本审计](../experiments/contact_gap_diagnosis_20261018/data_audit/audit.json)、[拟合结果](../experiments/contact_gap_diagnosis_20261018/tiny_fit/results.json)、[full16 梯度](../experiments/contact_gap_diagnosis_20261018/full16_gradients/diagnostics.json)。

## 官方复测与最终模型

同一个 E. coli 文件，650 正例、65,000 负例，全长、原始方向、无检索过滤。数据哈希及 CSV 的行顺序、标签、两链长度均已核对。

| 指标 | 官方复测 | 我们 final step 1616 |
| --- | ---: | ---: |
| Contact 互作 AP | 0.295202 | 0.146702 |
| 残基 P@K，12 Å | 0.182276 | 0.102194 |
| 残基 P@K，8 Å | 0.107526 | 0.051598 |
| CLIP AP | 0.250087 | 0.244315 |
| Contact 阈值 0.5：TP / FP | 128 / 53 | 124 / 332 |
| Contact 阈值 0.5：Precision / Recall | 0.7072 / 0.1969 | 0.2719 / 0.1908 |

在 0.5 阈值下，两者召回接近，但我们的假阳性更多。阈值和校准会影响该对比；AP 本身也有明显差距，因此不能只用调整阈值解释问题。

官方 padding 检查分数差为 0，缓存特征与直接前向分数差约 1.92e-5。两者环境均为 Torch 2.8.0+cu128、Transformers 4.48.1，使用 FP32 权重、FP16 评估 autocast；不宣称两种实现逐位一致。

证据：[官方指标](../experiments/contact_gap_diagnosis_20261018/official_ecoli_recheck/metrics.json)、[原最终 test](../experiments/esmc_prott5_glm2_residue_gated_v1_20261015/test_ecoli_full16_final/metrics.json)、[评估对齐](../experiments/contact_gap_diagnosis_20261018/evaluation_alignment.json)。后续 LR 对照不重复使用 test 选参数。

## 损失、梯度和归一化

诊断起点为按 val 残基 P@K 选出的 step 1500；固定其权重，重放原计划中的 step 1501–1503，每批 1024 对、microbatch 16、最大长度 512。完成 6 次独立前反向，权重指纹保持相同。

- 真实接触占 Contact loss 的 77.0%–77.6%；结构内非接触占 16.3%–16.9%；额外负蛋白对占 5.5%–6.6%。
- 乘 0.1 后的 CLIP backbone 梯度范数为 Contact 的 1.33–1.42 倍。范数较大不等于必然压制另一目标。
- 联合梯度沿 Contact 方向的投影相对 Contact-only 为 1.049、1.075、0.963；融合层余弦为 0.953–0.974。局部层有负余弦，但三个批次未显示持续的整体强冲突。
- 固定同一 logits/targets/负例，将四个 microbatch 的分母合并，总 loss 变化较小。不过单个 microbatch 的 logit 梯度系数比可落在 0.706–1.672，因此 microbatch 16 和 64 不能称为数学上完全等价。
- 这里没有重新执行 64 对的 hard-negative mining，也没有证明归一化差异对长训练无影响。

因此本轮保持 CLIP 权重、负例规则及门控不变，先验证学习率和训练预算是否有作用。

## 与公开实现的确切差异

读取本地官方 Git revision `bb75bb1a960b3c8325a643b5cd7d7c11e41b5a0f` 中的函数，独立核对参数名称与负例逻辑，没有使用被研究代码修改过的工作区函数冒充官方版本。

1. 官方冻结前 16 个 block 后训练 block 16–32，共 345,418,240 个 backbone 参数；我们训练 block 17–32，共 325,099,520 个。差别恰为 block 16 的 7 个参数张量；两者 embedding 都冻结，当前 gLM2 主干没有额外末端 norm 参数。我们的全部可训练参数为 390,744,975，173 个张量。
2. 不能仅按配置中的 `self_neg_ratio=1` 断定官方额外加入了一批 self-negative。该公开函数的 self 分支查询的是已知正对 `(A_i,B_i)` 的对角线，并最终索引 `embed2`。实际 16 个训练正对、交换后 32 个查询、已知正例齐全时，该分支新增 0 个负例，hard 分支产生 64 次负例呈现。hard 分支仍可能经交换项选中同序列对。
3. 我们的策略对无序完整序列对去重、排除同序列对，再按本地历史预算补足；三批共选择 6134 / 6138 / 6138 个负例，重复、self 和预算缺口均为 0。该规则与公开函数有明确区别。
4. 官方 optimizer 的名称匹配规则排除 bias、`LayerNorm.weight`、`logit_scale`、`alpha`、`beta` 的 weight decay；研究 optimizer 对两组统一应用 0.1。官方 cosine 下限为初始 LR 的 0.1，研究旧阶段下限为 0。

负例分支核对采用真实样本身份和合成排序分数，证明分支行为，不代表官方训练时的实际负例名单。公开源码也不能直接证明发布 checkpoint 历史上使用了完全相同的版本。这些差异先记录，本轮不混入 LR 单变量对照。

证据：[公开实现语义审计](../experiments/contact_gap_diagnosis_20261018/public_semantics/results.json)、[当前参数清单](../experiments/contact_gap_diagnosis_20261018/tiny_fit/parameter_audit.json)。

## 学习率对照与后台执行

最值得检验的假设仍是：切换 full16 后的有效学习率及新增训练预算不足。旧 full16 阶段仅 498 步，backbone LR 从 2.371e-6 衰减至 0，均值约 8.194e-7；这还不是已证明的性能差距原因。

| 项目 | A | B |
| --- | --- | --- |
| 起点 | 相同 val-best step 1500 | 相同 val-best step 1500 |
| Backbone LR | 1e-5 | 3e-5 |
| Head / fusion / gate LR | 1e-4 | 1e-4 |
| 正式阶段 | 20 step warmup，随后常数 | 相同 |
| 训练预算 | 先 100，再到 200 | 同等预算 |
| 验证 | 0 / 50 / 100 / 150 / 200 | 相同 |

显式迁移已核验：初始权重逐文件一致；Adam moments、更新计数和 RNG 一致；A/B 配置只差 backbone LR。阶段 step 从 0 计数，原始来源仍是 step 1500。训练核心文件没有改动，原有输入、源码、checkpoint 的严格恢复校验保留。ESMC/ProtT5 冻结，gLM2 后 16 层、全部门控、单步内冻结 embedding 复用均保持原策略。

每组已在独立副本做两个正式规模、直接到峰值 LR 的更新探针，均通过有限步检查。相同第一批的 Contact loss 都为 0.356381、CLIP loss 都为 1.064289，实际 backbone `||Δθ|| / ||θ||` 分别为 9.307e-5 / 2.792e-4，约为 1:3；16 层都有有限非零梯度。第二批 Contact loss 为 0.368911 / 0.369868，梯度范数为 0.5925 / 0.7041。前后两步样本不同，不能把这些数值当成固定样本 loss 趋势。第一步实际 AdamW 更新与联合负梯度余弦约 0.260，表明 Adam 历史和预条件确实改变更新方向；它不是沿 Contact 单目标的直接度量。

探针权重全部丢弃，正式 A/B 从迁移起点重新加载。两步通过不代表长期稳定或泛化收益。证据：[A 探针](../experiments/contact_gap_diagnosis_20261018/lr_comparison/lr_a/peak_lr_probe/results.json)、[B 探针](../experiments/contact_gap_diagnosis_20261018/lr_comparison/lr_b/peak_lr_probe/results.json)。

后台顺序为 A 0–100 → B 0–100 → A 100–200 → B 100–200。两组稳定时都完成同等 200-step 预算，避免只延长早期偶然领先者。主指标为 val 残基 P@K；按相同结构配对，并按任一蛋白的同源簇连通分组 bootstrap。目前 100 个 val 结构不共享这些簇，因此得到 100 个独立分组；这不是多个训练随机种子的重复实验。

15:56 已核验 A 组正式第 1 步：耗时 116.40 秒，Contact loss 0.356381、CLIP loss 1.064289；第一批预测和负例统计与探针一致。warmup LR 为 backbone 5e-7、head 5e-6，16 个 block 均有梯度，冻结 embedding 捕获/复用各 128 次，PPI/DDI 各 512 对。详见 [首步核验](../experiments/contact_gap_diagnosis_20261018/first_training_step_verified.json)。

以约 117 秒/step 计，两组各 100 step 共约 6.5 小时；两组各 200 step 共约 13 小时，另加验证和保存。后台完成时生成 `lr_comparison/comparison_000100.json` 和 `comparison_000200.json`，以及对应 Markdown。出现失败会写入状态，不自动改超参数继续。

文件位置：

- 实验目录：`/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018`
- 独立代码：`/home/zlab/.codex/worktrees/contact-gap-diagnosis-20261018/FlashPPI`
- 后台入口：独立代码目录的 `tools/run_contact_gap_lr.py`
- [迁移证明](../experiments/contact_gap_diagnosis_20261018/lr_comparison/preparation.json)
- [后台启动记录](../experiments/contact_gap_diagnosis_20261018/lr_comparison/supervisor.launch.json)：历史 supervisor PID 1423700，现已正常退出。后续查询应以状态文件和实际进程为准，勿向历史 PID 发送信号。

## 验证与尚未完成的部分

新增 `tests/test_contact_gap_diagnostics.py` 的 8 项 CPU 检查通过：归一化、参数命名、阶段迁移与 Adam 状态、实际参数更新、配对分组与行身份、监督脚本识别提前停止并允许策略内的负例缺口。已有 contact objective 8 项、full finetune 10 项在本次排查中通过。GPU 检查使用一张 RTX 5090 32 GB。

32 条抽查不是对全部原始结构距离的重建。短样本拟合不能代替全模型泛化检查。full16 梯度仅取固定权重下三个批次。学习率对照已完成，结论受固定小 val 和单次训练 seed 限制。

单 gLM2、单 gLM2 + gate、三编码器 + gate 的匹配架构实验仍未启动；需先确认当前训练能取得有效改善，再判断下一轮投入。现阶段没有证据足以断言 gate、ESMC 或 ProtT5 是性能差距的主因。
