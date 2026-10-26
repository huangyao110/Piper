# B 阶段累计 2 epoch：测试与续训记录

> 2026-10-22 15:27 最终 test 更新：step 1616 的 65,650 对完整评估已完成并复核。Contact AP 0.2610，残基 P@K（12 Å）0.1717、（8 Å）0.1087，CLIP AP 0.2665。相较 B1413，接触互作排序略降，残基定位与 CLIP 小幅改善。训练和 test 均已完成。详见 [最终 test 报告](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/test_final_20261022_152556/REPORT.md)。下文保留此前阶段记录。

> 2026-10-22 15:18 完成更新：累计 1616 step / 2 个名义 epoch 已正常完成，最终 checkpoint 权重和完整训练状态哈希校验通过，最终 val 已完成，进程退出、GPU 释放。最终 val：Contact AP 0.5989、残基 P@K 0.1208、CLIP AP 0.5548；各任务历史最佳仍分别保留。最近一次完整 test 为 step 1413。详见 [训练完成记录](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/COMPLETION.md)。下文保留此前阶段记录。

> 2026-10-22 13:57 清理更新：移除 27 个过时中间 checkpoint 的权重/训练状态及 1 个可重建评估缓存，释放 51.31 GiB，实验分区可用约 227 GiB。当前恢复点、历史最佳、正式 test 模型及全部评估结果保留；清理后训练正常推进至 1574/1616。此次已删除部分普通中间模型，旧记录中“所有模型权重保留”仅描述当时的操作。详见 [清理报告](/media/zlab/lexar4t/FlashPPI/experiments/cleanup_20261022_intermediates/REPORT.md)。

> 2026-10-22 08:48 test 更新：step 1413 的 65,650 对完整 test 已完成。对比 B922，Contact AP 0.2287→0.2652，残基 P@K（12 Å）0.1597→0.1696、（8 Å）0.0999→0.1067；CLIP AP 0.2691→0.2619。原状态续训已实际完成 step 1414，目标保持 1616。新输出为 `train_after_test_001413_20261022_084327_001413`。详见 [测试及续训报告](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/test_latest_20261022_084217/REPORT.md)。下文为此前阶段记录，当前进度以 [动态状态](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/status.json) 为准。

> 2026-10-21 17:00 test 更新：step 922 的 65,650 对完整 test 已结束。对比 B521，Contact AP 0.2164→0.2287，残基 P@K（12 Å）0.1460→0.1597、（8 Å）0.0843→0.0999；CLIP AP 0.2775→0.2691。已从原 checkpoint 恢复并核实实际完成 step 923，目标仍为 1616。新输出为 `train_after_test_000922_20261021_165422_000922`。详见 [测试及续训报告](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/test_latest_20261021_165339/REPORT.md)。下文保留此前阶段记录，当前状态以 [动态状态](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/status.json) 为准。

> 2026-10-21 恢复更新：主机重启后已从 step 700 完整恢复；新输出为 `train_recovery_20261021_000700`，目标保持 1616。未保存的 701–704 重算，首步 701 数值已逐项对照一致。当前预计 10 月 22 日下午完成。详见 [恢复报告](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/recovery_20261021/REPORT.md)。

更新时间：2026-10-20 23:04:11 +0800。用户要求先查看最新 test，再把当前 B 训练拉到 2 个 epoch；随后要求清理磁盘。

## 已完成的 test

正式训练在 step 521 完整保存并暂停，完整评估 65,650 对（650 正例、65,000 负例）。与 B400 使用相同输入哈希、原始行顺序、完整序列、方向和全部 30 个核心代码文件；从 CSV 独立重算的汇总指标一致。按用户要求测试最新保存点，续训计划不根据 test 选点或改参数。

| 指标 | B400 | B521 | 变化 |
| --- | ---: | ---: | ---: |
| Contact AP | 0.199554 | 0.216403 | +0.016848 |
| Contact AUROC | 0.864170 | 0.865651 | +0.001480 |
| Residue P@K (12A) | 0.140650 | 0.145997 | +0.005347 |
| Residue P@K (8A) | 0.081768 | 0.084258 | +0.002490 |
| CLIP AP | 0.264520 | 0.277506 | +0.012986 |
| CLIP AUROC | 0.853630 | 0.866108 | +0.012478 |
| Patch AP | 0.198936 | 0.205385 | +0.006449 |
| Patch P@K | 0.204894 | 0.210950 | +0.006056 |

[完整 test 与复核](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/test_ecoli_step_000521/comparison_audit.json)。以上为点估计变化。

## 续训设置与验证

当前有效训练记录 827,305 条，batch 1024，ceil(827305/1024)=808 step/epoch。因此 B 阶段累计 2 个 epoch 为 **1616 step**，从 521 恢复，继续 522–1616。仍使用有放回的 1:1 PPI/DDI 抽样；epoch 是预算口径，不保证每条记录恰好看过两遍。

沿用 gLM2 后 16 层全量微调，ESMC-300M 与 ProtT5 冻结；残基接触 head、gate、Contact/CLIP loss 权重 1.0/0.1、512 PPI + 512 DDI、microbatch 16、单步内 frozen embedding CPU 复用均不变。Backbone/head 学习率继续恒定 3e-5/1e-4，warmup 不重启；没有启用新的 GPU 优化。

显式迁移只修改配置 steps: 600→1616、epochs: null→2。权重文件、完整 Adam 状态、step、RNG 和历史最佳路径逐项核对一致。唯一核心源码变更是允许采样器在最后一个 200-step 块只取前 16 步；仍先抽完整 DDI 块以保持 RNG 的批内打乱顺序。新计划的前 614,400 个索引与原 600-step 计划完全一致，1616-step 全计划也等于 1800-step 计划的前缀；每步保持 PPI/DDI 各 512。

CPU 检查 23 项通过。实际 GPU 首步为 522，Contact loss 0.288980、CLIP loss 0.858768；16 层均有有限非零梯度，缓存捕获/复用各 128 次，负例无重复和 self pair，学习率正确。首步是恢复检查，不作为 loss 趋势结论。

继续每 50 step 验证和保存；额外在 808、1616 验证并保留完整 checkpoint。下一次常规验证为 550。仍依据 val 选择最佳模型，不自动追加 test。预计剩余约 35–36 小时（约 10 月 22 日上午，取决于后续验证与保存耗时）。

[恢复证据](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/first_step_verified.json) · [迁移核对](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/extension/preparation.json) · [检查记录](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/checks.json)

## 磁盘清理和长跑保留规则

内存仍有约 102 GiB 可用，紧张的是磁盘。本次移除 22 个已被替代且不受恢复依赖保护的 training_state.pt，释放 **64.05 GiB**，可用磁盘由 44.21 GiB 增至 108.26 GiB（df 显示约 109G）。所有模型权重、正式 val/test、数据集、预训练权重保留；82 个完整恢复点保留。360 个模型/评估文件的 inode、大小和修改时间核对未变，训练未中断。清理过的普通保存点仍可预测，但不再支持完整 Adam/RNG 恢复；其 retention.json 标明这一点。

本轮长跑保留最近两个完整恢复点、当前各任务最佳和 epoch 边界。其他新保存点的模型权重验证后转存到 /home/zlab/.cache/flashppi_checkpoint_archive/b_twoepochs_20261020，并保留原路径链接；仅清理其 optimizer 状态，所有模型权重和验证结果继续保留。旧实验不由该轮换程序处理。

[清理记录](/media/zlab/lexar4t/FlashPPI/experiments/cleanup_20261020_twoepochs/report.json) · [清理后验证](/media/zlab/lexar4t/FlashPPI/experiments/cleanup_20261020_twoepochs/verification.json)

## 后台位置

- 代码：`/home/zlab/.codex/worktrees/contact-gap-b-twoepochs-20261020/FlashPPI`
- 训练输出：`/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/train_000521_001616`
- supervisor PID：1877862；trainer PID：1878216（启动时快照）。
- [动态状态](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/status.json) · [启动命令](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/train.launch.json) · [训练日志](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/training.log)
- [交接摘要](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/handoff_summary.json) · [本段验证历史](/media/zlab/lexar4t/FlashPPI/experiments/contact_gap_diagnosis_20261018/b_twoepochs_20261020/validation_history.json)
