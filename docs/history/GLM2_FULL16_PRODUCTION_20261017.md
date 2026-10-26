> 2026-10-18 10:04 更新：step 1464 完整 test 已完成，训练已从该 checkpoint 恢复并完成 step 1465。当前目录 `motif_full16_resume_after_test_001464`，supervisor/trainer PID 为 1403456 / 1403821。[本次 test 对比](/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/test_ecoli_step_001464/comparison.md)。以下切换时的 PID/目录为历史记录，实时状态以实验根目录 `status.json` 指向的训练目录为准。

# gLM2 后 16 层全量微调：生产切换

2026-10-17，用户授权停止 LoRA，继承当前 checkpoint，改为 gLM2 后 16 层全量微调。

## 实现与继承规则

- 运行源码：`/home/zlab/.codex/worktrees/glm2-full16-production-20261017/FlashPPI`。
- LoRA 最后一步：1118；来源为原运行 `motif_resume_after_full16_probe_001108/checkpoints/step_001118`。
- 在 CPU FP32 合并每个已训练 Q/V LoRA 增量，随后移除 LoRA 模块。
- 冻结 gLM2 embedding 和第 0–16 层；全量训练第 17–32 层的 attention、FFN、norm。
- gLM2 可训练参数 325,099,520；连同融合、门控及预测头共 390,744,975。
- ESMC-300M、ProtT5 保持冻结；原残基级 concat head、残差通道门控及 CLIP 残基加权池化保留。
- 原门控、融合及预测头的权重和 61 个参数的 Adam 历史按名称精确继承。LoRA 因子的一阶、二阶矩不能转换为稠密权重的矩，因此新解冻 gLM2 参数从空 Adam 状态开始。
- 移除 LoRA 后，其 dropout 也随之移除；合并保留已有 Q/V 权重增量，不承诺训练模式与 LoRA 逐位等价。

## 不重置采样和学习率进度

保留原步数、RNG、输入校验值和采样计划。训练从 step 1119 开始，仍以总 step 1616 为本次两 epoch 的终点。

- global batch 1024、microbatch 16、最大长度 512；PPI/DDI 1:1。
- 15% 残基掩码、interface crop、唯一异蛋白接触负采样。
- Contact / CLIP loss 权重 1 / 0.1。
- gLM2 基础学习率 1e-5，heads 基础学习率 1e-4；继续原 cosine 进度，不重启 warmup。
- FP32 参数/梯度/Adam 状态，BF16 train、FP32 全局 CLIP logits、FP16 eval。
- 梯度检查点及单步 CPU 冻结 embedding 复用保留。
- 每 50 step validation、每 25 step 恢复 checkpoint；完成后自动跑 E. coli test。

当前参数使用 `mode=full_last_layers`、`full_finetune.last_layers=16`、`backbone_learning_rate=1e-5`。
配置位于实验根目录下 `manual_full16_transition_20261017_2229/migrated_state/config.json`。

## 检查记录

1. 46 项 CPU 回归检查通过：合并输出、精确选择解冻层、冻结分支、门控、残基标签、负采样、缓存复用、Adam 迁移以及连续训练与恢复训练逐位一致。
2. 真实 step 1108 checkpoint 的 CPU 预检查通过：合并 Q/V 完全一致、冻结前缀不变、继承权重不变、模型重新构造和加载后的可训练权重完全一致。
3. 正式 step 1118 checkpoint 安全保存并校验后迁移；源 checkpoint 未修改，普通 resume 的 config/source/input/world-size 严格校验保持启用。
4. 正式规模 step 1119–1120 验收通过并保留实际训练更新；全部 16 层都有非零梯度、稠密 Q/V 权重均已更新，173 个参数张量都有 Adam 状态。
5. 从完整 step 1120 checkpoint 原生恢复后，step 1121 已完成。旧 heads 的 Adam 步数按原历史递增，新 backbone 的 Adam 状态成功恢复；源码、配置、输入、采样计划、学习率及缓存复用均通过核验。

| Step | Contact loss | CLIP loss | 秒/step |
|---|---:|---:|---:|
| 1119 | 0.358631 | 1.182675 | 116.98 |
| 1120 | 0.365479 | 1.163297 | 117.08 |
| 1121（恢复后） | 0.365771 | 1.194672 | 117.60 |

GPU 实测峰值 allocated 约 25.90 GiB；恢复后峰值 reserved 约 28.45 GiB，二者不能相加。含完整 Adam 状态的 checkpoint 约 4.37 GiB。

验收进度和最终启动核验以以下文件为准：

- 实验根目录：`/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015`。
- `manual_full16_transition_20261017_2229/regression_tests.json`。
- `manual_full16_transition_20261017_2229/migrated_state/migration.json`。
- `manual_full16_transition_20261017_2229/acceptance_verified.json`。
- `manual_full16_transition_20261017_2229/resume_verified.json`。

## 运行与后续查询

Supervisor 为 `switch_full16_20261017_2229.py`，负责安全切换、两步验收、正式续训及结束后的 test。
截至 2026-10-17 22:43，supervisor/trainer PID 为 1364837 / 1365545；正式运行目录是 `motif_full16_resume_001120`。
下一次计划 validation 为 step 1150。
先读取实验根目录 `status.json` 的 `training_dir`，再读取该目录的 `status.json` 和 `history.jsonl`。
根目录的 `step` 可能是恢复起点，不能当作实时训练步数。

普通恢复必须在上述固定源码目录执行，使用该 checkpoint 对应的完整配置；不要修改正在训练的固定源码。
保存的全量微调 checkpoint 包含后 16 层的全部参数，加载时只依赖配置中的固定底座，无须旧 LoRA checkpoint。

保留的历史最佳模型可能来自 LoRA 阶段，用同一套 validation 指标继续比较。初始几步 loss 只能说明运行情况，不能证明全量微调已改善接触定位。

## 结果清理

用户授权删除过期 checkpoint 和临时测试权重，保留恢复点、历史最佳及正式评估。
清理脚本逐项保护 latest/final/epoch/best、正式 test 模型、warm-start 依赖，以及被保留 checkpoint 内部指向的历史最佳模型。
全部 loss 历史、配置、源码快照、日志、诊断报告及正式 val/test 输出保留。删除记录与原 checkpoint 元数据保存在 `experiments/cleanup_20261017_full16/`。
已删除 103 个过期/临时 checkpoint，释放 81,230,056,350 字节，约 75.65 GiB（81.23 GB）；保留 61 个 checkpoint。
289 个原有正式 val/test 文件的 SHA256 在清理前后完全一致，恢复点及历史最佳模型清理后再次核验通过。
磁盘可用空间从约 139.38 GiB 增加至 215.04 GiB；详细删除清单和元数据归档见 `experiments/cleanup_20261017_full16/report.json`。
