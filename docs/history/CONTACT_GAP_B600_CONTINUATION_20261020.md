# B400 继续训练至 B600

> 2026-10-20 23:00 更新：已在 B521 完成最新完整 test，并从同一点恢复至 B 阶段累计 2 epoch（1616 step）；以下 B600 计划为历史记录。当前设置与结果见 [CONTACT_GAP_B_TWO_EPOCHS_20261020.md](CONTACT_GAP_B_TWO_EPOCHS_20261020.md)。

> 2026-10-20 耗时诊断后更新：正式训练在 step 432 完整保存，独立计时结束后从同一保存点原样恢复。当前输出为 `train_resume_after_runtime_profile_000432`，原 `train_000400_000600` 保留 step 401–432。总预算仍为 B600，未接入优化。详见 [耗时诊断报告](TRAINING_RUNTIME_PROFILE_20261020.md) 和下方动态状态链接。

2026-10-20，用户要求继续训练并由助手判断设置。已选择从 B400 连续追加 200 步至 B600；北京时间 18:50 启动后台任务，18:53 已核验实际 step 401 完成。实际进度见 [status.json](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/status.json)，首步证据见 [first_step_verified.json](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/first_step_verified.json)。

## 设置及依据

| 项目 | 设置 |
| --- | --- |
| 恢复点 | 前段 B 训练的 `step_000400` |
| 本段步数 | 401–600，追加 200 步，到 600 结束 |
| Backbone LR | 恒定 `3e-5` |
| Head / fusion / gate LR | 恒定 `1e-4` |
| Optimizer / RNG | 完整继承 AdamW 状态、计数及 RNG |
| Warmup | 不重启，仍沿用早已完成的 20 步 warmup |
| gLM2 | 训练 block 17–32，共后 16 层 |
| ESMC / ProtT5 | 冻结，保留单步内 CPU embedding 复用 |
| Gate / contact head | 保持原残基级 contact head、残差通道 gate、CLIP residue gate |
| Loss 权重 | Contact 1.0，CLIP 0.1 |
| Batch | 1024，PPI / DDI 各 512，microbatch 16 |
| 采样 | 原 seed、固定 200 步采样块，延续原样本计划 |
| 验证与保存 | 450、500、550、600 |
| 主指标 | 固定 val 残基接触 P@K；同时报告 Contact 互作 AP 和 CLIP AP |

B301–320 的 Contact loss 均值为 0.313036，B381–400 为 0.302737；对应 CLIP loss 为 0.921305 → 0.877429，后段梯度有限且没有持续增大的迹象。B400 的 val 残基 P@K 0.128164 为当前阶段最高。这些观察支持先延续已有学习率，再用有限预算观察收益，目前尚无充分依据立刻降 LR。

本段只改变训练配置中的 `steps: 400 → 600`。在 B600 完成后根据 val 的完整轨迹判断下一段是否需要降学习率；本次没有预先安排更长训练或自动衰减。历史 B400 及各任务最佳 checkpoint 保留，B600 不会仅因步数更新就替代验证成绩更好的模型。

本段主对比为 B600 与 B400；A 仍停在 200，不是另一轮同预算 A/B 实验。训练期间及结束时不自动新增 test。此前 B400 test 已单独报告，不用于在本段多个 checkpoint 间选优。

## 恢复与检查

- B400 的全部 30 个训练、模型和评估核心源文件哈希与本次运行完全一致；没有改变计算图或采样实现。
- 显式迁移保留权重、全部 optimizer 状态、随机数状态和 step 400；历史最佳 checkpoint 路径重定位后仍指向原文件。
- 新 600 步计划的前 409,600 个采样索引与已执行的 400 步逐项、逐序一致；后 200 步为新增批次。各步仍含 512 PPI 和 512 DDI。
- 准备与监督工具改为读取恢复步数，支持已有 200 步块的连续扩展；旧 B200→400 的代码快照和结果保持不变。
- CPU 回归覆盖 B200 和 B400 的恢复、Adam 下一步更新逐值一致、学习率不重启、历史采样前缀、结束步数及四次验证完整性；相关混合采样与分布式回归一起 **23 项通过**。
- 已用真实 B400 验证文件检验汇总流程，起点 P@K 与 0.12816447330086567 一致。
- RTX 5090 首步实际从 401 开始，LR 为 `3e-5 / 1e-4`，16 个 backbone block 均有有限非零梯度；缓存捕获 / 复用各 128 次，无重复或 self 负例，预算缺口为 0。耗时 116.25 秒，Contact loss 0.309896、CLIP loss 0.944209；单步用于确认恢复成功，不能单独作为 loss 趋势。

证据：[preparation.json](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/preparation.json)、[checks.json](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/checks.json)、[工具快照](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/diagnostic_code/sha256.json)。

## 后台位置

- 代码目录：`/home/zlab/.codex/worktrees/contact-gap-b-extension-20261019/FlashPPI`
- 实验目录：`experiments/contact_gap_diagnosis_20261018/b_extension_20261020`
- 训练输出：该目录下 `train_000400_000600`
- [后台启动命令](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/supervisor.launch.json)
- [训练日志](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/train_000400_000600.log)
- [验证比较表](../experiments/contact_gap_diagnosis_20261018/b_extension_20261020/validation_comparison.md)

按上一段约 116.4 秒/step，新增 200 步约需 6.5 小时，另加启动、验证和保存。启动时预计 **10 月 21 日北京时间 01:25–01:40** 完成。监督程序每 15 秒更新状态，异常时记录失败并请求保存停止，不自动修改参数重试。
