> 2026-10-17 22:43 更新：已从 step 1118 的 LoRA 切换为 gLM2 后 16 层全量微调，step 1121 恢复检查通过。当前源码、运行目录和清理记录见 [全量微调运行记录](/media/zlab/lexar4t/FlashPPI/docs/GLM2_FULL16_PRODUCTION_20261017.md)。下文先前运行状态为历史记录。

# 保留门控的残基级 concat head 续训

用户于 2026-10-15 授权：核对 FlashPPI 官方源码，完成代码检查，停止当前 patch 训练，并从当前 checkpoint 启动残基级训练。

## 源码依据和模型

- 官方仓库：https://github.com/TattaBio/FlashPPI
- 核对版本：`bb75bb1a960b3c8325a643b5cd7d7c11e41b5a0f`。
- `flashppi/model.py` 中的 ContactHead：segment embedding → 沿序列维 concat → 两层 Transformer → LayerNorm → Q/K 投影 → 多头残基对点积 → output_mix。
- 本地 ContactHead 去除未使用的可选 pair-features 返回接口后，AST 与官方完全相同；`layers.py` 与官方逐字节相同。沿用现有实现，无须重写核心 head。

新路径：三个编码器的对齐残基 embedding → 原有残差融合 → 逐残基残差通道 gate → 原版 ContactHead。输入 `[B,L1,D]`、`[B,L2,D]`，输出 `[B,L1,L2]`。

保持 ESMC-300M、ProtT5 冻结；gLM2 保持现有 Q/V LoRA。融合公式、CLIP residue-gated weighted mean pooling、通道 gate 的公式和权重均不变。原 `patch_gate` 改名为 `residue_gate`，在每个残基上调用同一模块，不再做 mean pool。

## 标签、迁移与训练设置

- 使用原始残基接触矩阵及结构可观测 mask，不再把一个残基接触扩展为多个 patch 正例。
- 删除 patch coverage 分支及其损失；只删除其两个参数张量。
- 所有其他可训练参数必须严格匹配名称、形状、dtype 并逐个精确复制；不允许静默跳过或重置。加载时校验 checkpoint、旧代码快照和冻结输入的哈希。
- 从旧训练安全停止时保存的 checkpoint 开始新阶段；优化器与学习率调度重新初始化，阶段 step 从 0 开始。不是旧优化器状态的原样恢复。
- 保留全局 batch 1024、microbatch 16、最大训练长度 512、PPI/DDI 1:1、15% sequence masking、唯一异源负例策略、CLIP/contact 损失权重 0.1/1。
- LoRA 学习率 `1e-5`，其余可训练参数 `1e-4`，81 step warmup、cosine，2 个采样 epoch / 1616 step；新阶段随机种子 20261015。
- 验证用原始全长序列。接触 checkpoint 选择指标为残基 contact Precision@K；额外保留 3/2 patch grid 作为评估汇总，**不用于训练、模型输入或标签池化**。
- 训练结束后由 launcher 自动执行完整官方 E. coli test；不以 test 选择训练 checkpoint。

## 代码和检查

- `flashppi/gated_modules.py`: ResidueGatedContactHead，直接调用原版 core。
- `flashppi/residue_warm_start.py`: 严格迁移及配置一致性检查。
- `configs_train/esmc_prott5_glm2_residue_gated_v1.json`: 实际运行配置，已绑定旧训练安全停机保存的 step 477。
- `tests/test_residue_gated_transition.py`: 原版 head 等价性、非零门控、padding/NaN mask、反向梯度、拒绝错误迁移、稀疏标签不扩张。
- 先通过 CPU 回归和与旧训练并行的小型真实 GPU optimizer step，再停止旧训练，使用最终停机 checkpoint 执行实际 1024/16/512 的完整 batch 检查。正式训练重新加载源 checkpoint，不加载检查期间更新过的权重。
- `tools/run_motif_training.py --residue-preflight ...` 在启动前校验完整 batch 检查、模型/数据/代码哈希及源 checkpoint。历史名称 `motif` 仅是该通用 launcher 的输出子目录；本阶段模型是残基级。

运行记录：`/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015`。原 patch 训练目录和已上传服务器包保留，不覆盖。

## 已执行的迁移和验证

- 旧训练收到 SIGTERM 后完成当前 optimizer step，安全保存 `esmc_prott5_glm2_gated_v1_20261014/training/checkpoints/step_000477`，状态 `stopped_at_checkpoint`。旧 supervisor 和 trainer 均已退出。
- 首轮 CPU 回归（残基迁移、门控、patch、coverage、负采样、sequence masking）59 项通过；扩展回归（残基迁移、门控、handoff、分布式训练工具）71 项通过。两组有重叠，不相加为独立测试数。
- 小型真实 GPU 检查通过后，再以最终 step 477 权重执行完整 batch：RTX 5090 32 GB，global batch 1024、microbatch 16、max length 512；512 个 PPI + 512 个 DDI。
- 完整 batch 完成前向、反向、optimizer update 和 checkpoint 保存：耗时 159.40 秒，allocated 23.38 GB、reserved 25.60 GB；contact loss 0.562879，CLIP loss 1.918592，梯度范数 0.619294。负例重复、自配对、预算短缺均为 0；没有 coverage loss。
- 125 个保留的可训练张量精确迁移，只有 coverage 输出层的 weight、bias 被移除；没有新增或静默重置的可训练张量。
- 正式启动时间：2026-10-15 23:07:54 +08:00；supervisor PID 1199357，trainer PID 1199429。启动校验 `launch_checks.json` 已通过。
- 正式运行重新从 step 477 初始化，未使用 preflight 更新后的权重。训练进度见 `motif/status.json`、`motif/history.jsonl` 和 `motif.log`。
- 初始全长 validation 已完成：228 个互作样本、100 个结构样本；残基 contact Precision@K 为 0.080826，互作 contact AP 为 0.393343，CLIP AP 为 0.521780。这是新阶段 step 0 的验证基线，不是 test 指标，也不是续训后的提升。
- 首个正式 optimizer step 已完成，后台继续运行：contact loss 0.561932，CLIP loss 1.927654，梯度范数 0.591906，耗时 159.28 秒；负例重复、自配对、预算短缺均为 0。ESMC/ProtT5 融合系数和 residue gate 系数均已更新。

本次代码位于独立工作树 `/home/zlab/.codex/worktrees/residue-gated-20261015/FlashPPI`，分支 `codex/esmc-prott5-residue-gated-20261015`，以保护旧训练代码和工作目录中的其他修改。

## 2026-10-16：step 230 完整 test 与原状态续训

为运行用户请求的 test，训练完成 step 230 后保存完整 checkpoint 并退出，释放 GPU；完整官方 E. coli test 已完成，65,650 行（650 正例、65,000 负例），数据哈希和行顺序与上次 test 一致。

- Contact AP 0.093129、AUROC 0.769715；CLIP AP 0.232951、AUROC 0.866362。
- 统一 3/2 patch 网格 AP 0.113997、Precision@K 0.128671。
- 原始残基接触 Precision@K 0.071883，649 个可评估正例。这是 test 指标，不与 validation 的 8.08% 起始值直接比较。
- 相比旧 patch step 450，CLIP 与局部 patch 排序小幅上升，整体 Contact 读出下降；未依据本次 test 调整训练或选择 checkpoint。

本次使用原生 `--resume` 恢复 step 230 的模型、优化器及 RNG 状态，保持原 1616 step 总预算、学习率计划和完全相同的采样顺序。由于 trainer 要求输出新目录，**当前训练目录改为**：

`/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_test_000230`

旧 `motif` 目录保留 step 0–230 的记录与 checkpoint。查看当前状态时读取实验根目录 `status.json` 的 `training_dir` 字段，不要只看旧 `motif/status.json`。新 supervisor PID 1244207，trainer PID 1244571；完整训练结束后仍会自动运行最终 test。

测试结果在 `test_ecoli_step_000230/metrics.json`，对比说明在同目录 `comparison.md`；恢复操作记录在 `manual_test_20261016/`。

## 2026-10-16 下午：step 358 完整 test

用户再次请求查看当前 test。训练安全保存于 step 358，完整测试 65,650 行后通过原生 `--resume` 恢复，总步数和学习率计划不变。

相对本模型 step 230：残基 Precision@K 7.19% → 7.81%；统一 3/2 Patch AP 0.1140 → 0.1215、Patch Precision@K 0.1287 → 0.1364；Contact AP 0.0931 → 0.0959、AUROC 0.7697 → 0.7993；CLIP AP 0.2330 → 0.2403、AUROC 0.8664 → 0.8652。测试数据哈希及每行标签、顺序均通过一致性核对，未按 test 改动训练计划。

**最新训练目录**为实验根目录下的 `motif_resume_after_test_000358`，历史训练链为 `motif`（0–230）→ `motif_resume_after_test_000230`（231–358）→ 当前目录。始终以根目录 `status.json` 的 `training_dir` 为准。新 supervisor PID 1261808，trainer PID 1262210。

本次结果：`test_ecoli_step_000358/metrics.json`；对比报告：`test_ecoli_step_000358/comparison.md`；操作记录：`manual_test_20261016_1508/`。

## 2026-10-17：step 563 完整 test

用户再次请求完整 test。训练安全保存于 step 563，全部 65,650 对样本评估成功，随后通过原生 `--resume` 接回训练，总预算仍为 1616 step，采样、优化器和学习率计划保持连续。

相对 step 358：残基 Precision@K 7.81% → 8.55%；统一 3/2 Patch AP 0.1215 → 0.1306、Patch Precision@K 0.1364 → 0.1453；Contact AP 0.0959 → 0.1167、AUROC 0.7993 → 0.8128；CLIP AP 0.2403 → 0.2247、AUROC 0.8652 → 0.8594。接触分支改善、CLIP 回落；未按 test 改动训练或 checkpoint 选择规则。

**最新训练目录**为实验根目录下的 `motif_resume_after_test_000563`。完整训练链为 `motif`（0–230）→ `motif_resume_after_test_000230`（231–358）→ `motif_resume_after_test_000358`（359–563）→ 当前目录。以根目录 `status.json` 的 `training_dir` 为准。新 supervisor PID 1288836，trainer PID 1289273。

结果：`test_ecoli_step_000563/metrics.json`；对比报告：同目录 `comparison.md`；操作记录：`manual_test_20261017_0022/`。

## 2026-10-17 上午：step 755 完整 test 与续训

用户要求“跑一下 test，然后接上”。训练在 step 755 安全保存完整 checkpoint，完成官方 E. coli test：65,650 对样本（650 正例、65,000 负例），测试数据哈希、逐行标签/顺序/ID/长度与 step 563 一致，预测分数均有限。

相对 step 563：残基接触 Precision@K 8.55% → 9.15%；统一 3/2 Patch AP 0.1306 → 0.1392、Patch Precision@K 0.1453 → 0.1530；互作 Contact AP 0.116689 → 0.116742、AUROC 0.8128 → 0.8233；CLIP AP 0.2247 → 0.2251、AUROC 0.8594 → 0.8641。局部接触排序继续改善，两个互作 AP 基本持平。上述为点估计，未作统计显著性检验；未依据 test 修改训练配置或选择 checkpoint。

通过原生 `--resume` 恢复 step 755 的模型、优化器和 RNG，总预算仍为 1616 step；采样计划及 cosine 学习率保持连续。已实际完成恢复后的 step 756：contact loss 0.382389、CLIP loss 1.270444、梯度范数 0.432902，负例重复、自配对和预算短缺均为 0。LoRA/head 学习率分别为 5.940927e-6 / 5.940927e-5。

**最新训练目录**为实验根目录下的 `motif_resume_after_test_000755`，上一段 `motif_resume_after_test_000563` 保留 steps 564–755。当前 supervisor PID 1318208，trainer PID 1318776。读取根目录 `status.json` 的 `training_dir`，再读取该目录的 `status.json` 获取实时 step；完整训练结束后仍会自动运行最终 test。

结果：`test_ecoli_step_000755/metrics.json`；对比：`test_ecoli_step_000755/comparison.md`、`test_step_000755_comparison.json`；恢复核验：`manual_test_20261017_0903/resume_verified.json`。

## 2026-10-17 中午：启用单步冻结 embedding 复用

用户最终只选择训练加速。原训练安全保存 step 814 后，原版与缓存版从同一完整断点分别执行 step 815、816。真实 1024/16/512 GPU 对照中，平均 optimizer step 耗时 158.25 → 117.60 秒，减少 25.69%；125 个可训练张量、AdamW 状态、RNG、loss、梯度范数及采样统计完全一致。相关回归 70 项通过。

正式训练从 step 814 恢复到 `motif_resume_frozen_cache_000814`，当前源码为 `/home/zlab/.codex/worktrees/frozen-cache-production-20261017/FlashPPI`。仅增加 `frozen_embedding_cache: cpu`，保持原始残基标签、门控、batch、loss、优化器状态、采样和 1616-step cosine 计划。每步重新计算 ESMC/ProtT5 输出并在本步回放中复用；100→200 调参未运行。

2026-10-17 12:01 +08:00 已确认正式 step 815 完成：117.94 秒，contact loss 0.387819、CLIP loss 1.378182，128 次缓存复用生效，训练数值与对照同一步完全一致。supervisor PID 1328990、trainer PID 1330087，后台继续训练，结束后仍自动运行最终 test。

继续以实验根目录 `status.json` 的 `training_dir` 获取当前训练目录。操作记录位于 `manual_cache_upgrade_20261017_1145/`，包括 `comparison.json` 和 `resume_verified.json`；说明见 [单步缓存生产记录](FROZEN_CACHE_PRODUCTION.md)。

## 2026-10-17 下午：step 951 完整 test 与缓存版续训

按用户请求，训练安全保存 step 951 后完成官方 E. coli test：65,650 对（650 正例、65,000 负例）。相对 step 755，数据哈希、评估源码、全长策略、方向、逐行标签/顺序/ID/长度及 patch 目标一致，所有互作分数均有限。

相对 step 755：残基接触 Precision@K 9.15% → 9.69%，8Å Precision@K 4.44% → 4.77%；统一 3/2 Patch AP 0.1392 → 0.1446、Patch Precision@K 0.1530 → 0.1581；互作 Contact AP 0.1167 → 0.1235、AUROC 0.8233 → 0.8357；CLIP AP 0.2251 → 0.2346、AUROC 0.8641 → 0.8630。多数主指标小幅改善，CLIP AUROC 略降；以上为点估计，未做显著性检验，未依据 test 修改训练配置或选择 checkpoint。

原生 `--resume` 从 step 951 恢复权重、优化器和 RNG。正式 step 952 已完成：117.27 秒，contact loss 0.384809、CLIP loss 1.371667，128 次缓存复用正常；负例重复、自配对和预算短缺均为 0。采样计划、cosine 学习率、门控和总预算 1616 step 保持连续。

**当前训练目录**为 `motif_resume_after_test_000951`；上一段 `motif_resume_frozen_cache_000814` 保留 steps 815–951。源码仍为 `/home/zlab/.codex/worktrees/frozen-cache-production-20261017/FlashPPI`，缓存开关为 `cpu`。supervisor PID 1343742、trainer PID 1344112；后续读取实验根目录 `status.json` 的 `training_dir`，再读取该目录下的实时状态。训练结束后仍自动运行最终 test。

结果：`test_ecoli_step_000951/metrics.json`；对比：同目录 `comparison.md` 和 `test_step_000951_comparison.json`；操作与恢复核验：`manual_test_20261017_1630/`。

## 2026-10-17 晚间：step 1071 完整 test 与续训

用户询问最新 test，训练安全保存 step 1071 后评估全部 65,650 对 E. coli 样本（650 正例、65,000 负例）。与 step 951 的数据哈希、评估源码、全长策略、方向、逐行标签/顺序/ID/长度和 patch 目标一致，预测分数均有限。

相对 step 951：残基接触 Precision@K 9.69% → 9.63%，8Å Precision@K 4.77% → 4.82%；统一 3/2 Patch AP 0.1446 → 0.1445、Patch Precision@K 0.1581 → 0.1572；互作 Contact AP 0.1235 → 0.1336、AUROC 0.8357 → 0.8330；CLIP AP 0.2346 → 0.2421、AUROC 0.8630 → 0.8662。两项互作 AP 继续提高，局部接触排序整体基本持平。以上为点估计，未做统计显著性检验；未依据 test 修改配置或选择 checkpoint。

原生 `--resume` 从 step 1071 恢复完整训练状态，2026-10-17 20:43 +08:00 已核验正式 step 1072 完成：117.10 秒，contact loss 0.370891、CLIP loss 1.148885，128 次冻结 embedding 复用生效，负例重复、自配对及预算短缺均为 0。优化器、RNG、采样计划、门控、cosine 学习率和总预算 1616 step 保持连续。

**当前训练目录**为 `motif_resume_after_test_001071`；上一段 `motif_resume_after_test_000951` 保留 steps 952–1071。源码仍为 `/home/zlab/.codex/worktrees/frozen-cache-production-20261017/FlashPPI`，缓存为 `cpu`。supervisor PID 1355531、trainer PID 1355859。实时进度继续读取实验根目录 `status.json` 指向的训练目录；结束后仍自动执行最终 test。

结果：`test_ecoli_step_001071/metrics.json`；对比：同目录 `comparison.md` 和 `test_step_001071_comparison.json`；操作及恢复核验：`manual_test_20261017_2036/`。

## 2026-10-17 21:59：接触损失与任务梯度诊断

在 step 1102 固定权重上检查三批正式规模数据（1024/16/512），无 optimizer 更新。真实接触点、结构内非接触点、额外负蛋白对分别占 Contact loss 的 78.68%、15.58%、5.74%。共享 LoRA 三批梯度余弦均负（均值 -0.1543），加权 CLIP/Contact 范数比均值 1.2636；融合层同向（均值 +0.9308），所有共享参数合计仍略同向（+0.1270）。这些是局部梯度冲突证据，不等同于已证明验证停滞根因。

8 项诊断回归检查通过。已按原配置从 1102 恢复并完成 1103，loss 与负采样重现诊断第一批；未调整门控、冻结、损失权重或学习率。当前运行 `/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_diagnostics_001102`，supervisor/trainer PID 为 1360135/1361106。

完整结果见 [诊断报告](CONTACT_OBJECTIVE_DIAGNOSTICS_20261017.md)。

## 2026-10-17 22:17：后 16 层全量微调显存实测

独立载入 step 1108、合并并移除 LoRA，完整解冻最后 16 层共 325,099,520 参数；包括其他分支共 390,744,975 可训练参数。保留 1024/16/512、BF16、梯度检查点、冻结 embedding CPU 复用。两次 optimizer 更新都约 117.25 秒；状态常驻后的峰值 allocated 25.895 GiB，reserved 28.082 GiB。双链 512 残基的 16 对压力 backward（96 对负例）也通过。

本次只确认显存可行性，未将全量微调切入正式训练。原 LoRA 已从 1108 恢复并完成 1109，当前目录 `/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_full16_probe_001108`，supervisor/trainer PID 为 1362066/1362839。详见 [实测报告](GLM2_FULL16_MEMORY_PROBE_20261017.md)。
