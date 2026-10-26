> 2026-10-17 22:43 更新：已从 step 1118 的 LoRA 切换为 gLM2 后 16 层全量微调，step 1121 恢复检查通过。当前源码、运行目录和清理记录见 [全量微调运行记录](/media/zlab/lexar4t/FlashPPI/docs/GLM2_FULL16_PRODUCTION_20261017.md)。下文先前运行状态为历史记录。

# 单步内冻结 embedding 复用：生产训练与实测

2026-10-17，按用户“目的就是加快训练，完成 1 就行”的要求，仅启用单步内 ESMC、ProtT5 输出复用。原 1616-step 残基级门控训练继续；100→200 step 调参流程未启动。

## 实现与训练语义

生产源码：`/home/zlab/.codex/worktrees/frozen-cache-production-20261017/FlashPPI`，分支 `codex/frozen-cache-production-20261017`。

配置增加 `"frozen_embedding_cache": "cpu"`。在一个 optimizer step 的第一遍前向中，将冻结且处于 eval 的 ESMC、ProtT5 原始残基 embedding 按原 dtype 保存到 CPU pinned memory；梯度回放时取回相同输出，结束后清空缓存。

每次回放校验 tokens、逻辑 padding mask、调用顺序、设备、autocast 精度及编码器身份。冻结分支消耗 RNG、参数发生变化或回放不完整时拒绝继续。缓存不跨 step，不按蛋白 ID 复用。

gLM2 LoRA、残差融合、投影、residue channel gate、CLIP weighted-mean gate 和 concat head 的前向、反向照常进行。原始残基标签、全局 batch 1024、microbatch 16、最大长度 512、PPI/DDI 1:1、15% masking、负采样、loss 权重及学习率计划沿用。

相对之前运行的源码，只修改 `research_model.py`、`research_distributed.py`、`research_train.py`，增加 `frozen_embedding_cache.py`。精确差异保存在操作目录的 `upgraded_state/source_changes.patch`。

## 真实 GPU 对照

硬件为 RTX 5090 32 GB；生产配置为 BF16 train、FP32 CLIP logits。以相同 step 814 checkpoint、AdamW 状态、RNG 和完整采样计划为起点，原版与缓存版分别通过原生 trainer 执行 step 815、816。

| Step | 原版耗时（秒） | 缓存版耗时（秒） |
|---|---:|---:|
| 815 | 158.3401 | 117.7434 |
| 816 | 158.1564 | 117.4576 |
| 平均 | 158.2483 | 117.6005 |

本次两步实测，optimizer step 耗时减少 **25.69%**，吞吐约为原来的 **1.346 倍**。计时涵盖训练前向、反向、梯度裁剪和 optimizer update；不含进程启动、模型加载、数据整理、validation 与 checkpoint 写入。两步结果不代表整次训练总墙钟时间已减少同一比例。

- 两步所有训练数值完全一致，包括 contact/CLIP loss、梯度范数、门控/融合系数、学习率、mask 统计和负采样统计。
- 两步结束后的 125 个可训练张量、AdamW 状态、CPU/CUDA/Python/NumPy RNG、步数和输入哈希逐项完全一致。
- 采样计划文件逐字节一致：`4f15e1c8624c8089ca5e1441dea3bbda5df670d1e4feeba0965aa9d293c724f0`。
- 每步捕获 128 次冻结编码器输出、复用 128 次；原本 256 次计算降为 128 次。
- 每步缓存 6,174,015,488 字节，即 5.75 GiB CPU embedding；GPU 峰值 allocated 23.9135 GB、reserved 26.2207 GB，与原版两步对照相同。
- 相关回归测试 **70 项通过**。小型 CPU/CUDA 两步测试另直接比较了每个梯度；完整生产 batch 对照比较梯度范数、更新后权重及优化器状态，未另外导出全部原始梯度。

本次验证是训练等价性与性能检查，没有重新运行 E. coli test 或据此调整超参数。

## 断点切换与后台训练

实验根目录：`/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015`。

旧训练完成 step 814 后保存完整 checkpoint 并退出。`tools/enable_frozen_cache_checkpoint.py` 校验原 checkpoint、源码快照及冻结输入，复制权重并保留优化器、RNG、步数和最佳 checkpoint 引用，仅更新缓存配置及所需源码哈希。原 checkpoint 和旧源码均保留。

原生 trainer 的严格 resume 校验保持完整；没有加入跳过源码或配置哈希的入口。`tools/compare_frozen_cache_runs.py` 对两步结果执行精确检查，并要求实际加速后才接入正式训练。若对照失败，supervisor 会从原保存点接回旧版训练。

正式训练从迁移后的 step 814 重新恢复，不采用诊断运行更新后的权重。当前目录为：

`motif_resume_frozen_cache_000814`

当前 supervisor PID 1328990，trainer PID 1330087。总计划仍为 2 epochs / 1616 steps，完整训练结束后自动执行最终 E. coli test。后续查看状态时，先读实验根目录 `status.json` 的 `training_dir`，再读该目录下的 `status.json`；根目录的 `step` 是恢复起点，不是实时步数。

2026-10-17 12:01 +08:00 已核验正式训练完成 step 815：耗时 117.94 秒，contact loss 0.387819、CLIP loss 1.378182、梯度范数 0.418158；所有训练数值与缓存版对照的同一步完全一致。128 次缓存复用实际生效，负例重复、自配对和预算短缺均为 0。LoRA/head 学习率分别为 5.3425437e-6 / 5.3425437e-5，符合原 cosine 计划。后台继续训练。

本次操作目录为 `manual_cache_upgrade_20261017_1145/`，主要证据：

- `regression_tests.json`：70 项检查的命令与结果。
- `original_checkpoint.json`、`upgraded_state/upgrade.json`：原始断点、源码变更和状态保留记录。
- `benchmark_reference/`、`benchmark_cached/`：原生训练的两步日志和完整 checkpoint。
- `comparison.json`：精确比较、耗时和内存结果。
- `resume_verified.json`：正式训练实际完成更新后的恢复核验。

## 重复检查

在生产源码目录运行，使用项目环境：

```bash
TRAIN_PY=/media/zlab/lexar4t/FlashPPI/artifacts/environments/esmc_prott5_gated_py312/bin/python
"$TRAIN_PY" -m pytest -q tests/test_frozen_cache_upgrade.py tests/test_frozen_embedding_cache.py tests/test_residue_gated_transition.py tests/test_gated_ensemble.py tests/test_research_multigpu.py tests/test_unique_contact_negatives.py tests/test_sequence_masking.py
```

既有缓存版 checkpoint 后续可以直接用原生 `--resume` 续训，并使用与保存时相同的配置和源码。迁移工具只处理本次旧源码到缓存版源码的切换，不用于重置训练或改变学习率预算。

## 后续运行位置：2026-10-17 16:38

step 951 完整 test 完成后，训练通过原生 `--resume` 继续到 `motif_resume_after_test_000951`。缓存开关保持 `cpu`，已核验 step 952 完成并复用 128 次输出，耗时 117.27 秒。当前 supervisor/trainer PID 为 1343742 / 1344112；恢复证明见实验目录 `manual_test_20261017_1630/resume_verified.json`。上文的 step 814 目录和 PID 是首次启用缓存时的记录；实时状态以实验根目录 `status.json` 指向的训练目录为准。

## 后续运行位置：2026-10-17 20:43

step 1071 完整 test 后，训练原生续接至 `motif_resume_after_test_001071`，已核验 step 1072 完成，耗时 117.10 秒且缓存复用 128 次。当前 supervisor/trainer PID 为 1355531 / 1355859；恢复证据为实验目录 `manual_test_20261017_2036/resume_verified.json`。缓存设置和原 1616-step 计划保持连续；实时进度仍以实验根目录 `status.json` 的 `training_dir` 指向为准。

## 后续运行位置：2026-10-17 21:59

完成 step 1102 的只读损失/梯度诊断后，正式训练位于 `/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_diagnostics_001102`，supervisor/trainer PID 为 1360135/1361106。首步 1103 已核验；缓存继续启用，128 次捕获与复用，约 118 秒/step。详见 [诊断与续训报告](CONTACT_OBJECTIVE_DIAGNOSTICS_20261017.md)。

## 后续运行位置：2026-10-17 22:17

完成独立全量微调显存实测后，原 LoRA 训练继续位于 `/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_full16_probe_001108`。supervisor/trainer PID 为 1362066/1362839。从 step 1108 恢复，首步 1109 已核验，单步缓存保持 128 次捕获/复用。实测权重未接入正式训练；见 [显存报告](GLM2_FULL16_MEMORY_PROBE_20261017.md)。
