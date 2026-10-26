# 单步冻结 embedding 复用与 100→200 step 调参

> 2026-10-17 更新：用户最终选择仅启用第 1 项训练加速，100→200 调参未运行。生产训练已在 step 814 接入单步缓存，完整对照平均耗时减少 25.69%，正式 step 815 恢复核验通过。当前实现与结果见 [单步缓存生产记录](FROZEN_CACHE_PRODUCTION.md)。下文保留最初两项方案的说明；其中尚未接入生产的描述属于历史记录。

实现位置：`/home/zlab/.codex/worktrees/frozen-cache-tuning-20261017/FlashPPI`，分支 `codex/frozen-cache-tuning-20261017`。从当前残基训练工作树复制其源码改动后独立开发，未修改正在训练的源码或配置。

## 哪部分保持训练策略，哪部分是新实验

单步缓存是等价计算优化。ESMC、ProtT5 参数冻结且处于 eval，同一份裁剪、mask 后的 tokens 对应相同输出。保存第一轮的原始残基 embedding，第二轮复用，可以省去一次冻结编码器前向。gLM2 LoRA、fusion、gate、CLIP/concat head 的前向和梯度计算仍然进行；batch 1024、microbatch 16、长度 512、标签、负采样、损失权重和学习率计划由原配置决定。

100→200 是独立的调参实验预算，不等同于当前 1616-step 正式训练。每组从同一个残基 checkpoint 复制所有可训练权重，重新初始化优化器，预先设置 200-step cosine、10-step warmup。第 100 步只决定哪些组继续，入选组通过原生 `--resume` 恢复模型、优化器、RNG、采样进度以及同一个学习率计划，累计训练到 200 步。不会在 100 步后重新 warmup 或训练额外 200 步。

## 缓存实现

- 配置开关：`"frozen_embedding_cache": "cpu"`，缺省为 `"off"`。
- `flashppi/frozen_embedding_cache.py`：每个 optimizer step 一个缓存，结束或异常时释放。没有跨 step、跨 epoch 或基于蛋白 ID 的缓存。
- 保留原 dtype，使用 CPU 存储；CUDA 路径使用 pinned memory。比较回放的 tokens、逻辑 padding mask、调用次序、设备、autocast 状态和精度。输入不一致时直接报错。
- 仅接纳参数冻结且子模块均处于 eval 的分支；检查参数/已有 buffer 版本与 RNG 消耗，避免把随机输出当成确定性输出复用。
- `ResearchModel.encode_protein` 只对 ESMC、ProtT5 原始输出调用缓存。可训练 fusion、投影和门控输出从不缓存；gLM2 LoRA 也不缓存。
- `GlobalGradientCache.step` 管理 capture/replay 的生命周期，保留原来的 CLIP 全局梯度回放、负例池及梯度同步。日志增加缓存调用数及 CPU embedding 字节数。

原生 resume 仍要求源码、配置、输入和 world size 完全一致。旧源码训练生成的 checkpoint 不会被这版代码静默接管；没有放宽现有的哈希检查。将优化切换到正在运行的长实验，还需要单独验证并记录源码升级及恢复过程。

## 已完成的核验

- 相关回归：109 项通过，覆盖缓存、调参、原有分布式梯度、残基门控、三编码器融合、handoff、负采样及 sequence masking。
- 后续增加缓存跨 checkpoint 恢复、多进程空 rank，以及调参输出配置/来源核验；专项复查 27 项通过。两组有重叠，不相加。
- 小型 CPU 与 CUDA BF16 两步对照：loss、每个梯度、参数更新、AdamW 状态、CPU/CUDA RNG 和负采样结果完全一致。冻结分支调用次数减半。
- 真实 step 650 checkpoint 与真实 ESMC/ProtT5/gLM2 权重：CPU FP32 两步对照，4 对样本、microbatch 2、裁剪长度 16；125 个可训练张量的梯度/更新、优化器、loss 和 RNG 完全一致。
- 真实核验结果：`/media/zlab/lexar4t/FlashPPI/experiments/frozen_cache_tuning_checks_20261017/real_checkpoint_cpu_fp32/result.json`。
- 小规模检查用于证明一致性，不能据此推算 1024/16/512 正式 GPU 训练的加速比例。生产吞吐仍需在空闲 GPU 上单独测量。

## 调参流程

默认计划为 `configs_train/residue_gated_tuning_v1.json`：3 组学习率，100 步后按 `validation contact_precision_at_k` 保留 2 组，继续到 200 步。每组都保存完整 validation 指标；test 不参与筛选，也不会在调参结束时自动运行。

| 组名 | Head/fusion/gate LR | gLM2 LoRA LR |
|---|---:|---:|
| baseline | 1e-4 | 1e-5 |
| lr_half | 5e-5 | 5e-6 |
| lr_one_half | 1.5e-4 | 1.5e-5 |

三组使用相同 seed、200-step 采样计划、裁剪和 masking 规则、batch/microbatch、门控结构、原始残基标签和负例策略。只允许调学习率、weight decay、损失权重及梯度裁剪上限；默认计划仅改变学习率。架构、LoRA、数据或采样规则的改变会被拒绝。

准备阶段不占 GPU，固定 checkpoint、配置及源码哈希，写出所有试验配置：

```bash
cd /home/zlab/.codex/worktrees/frozen-cache-tuning-20261017/FlashPPI
TRAIN_PY=/media/zlab/lexar4t/FlashPPI/artifacts/environments/esmc_prott5_gated_py312/bin/python
"$TRAIN_PY" tools/run_residue_tuning.py prepare \
  --checkpoint /media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_test_000563/checkpoints/step_000650 \
  --output /media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_residue_tuning_20261017_lr_v1
```

step 650 是当时 validation 残基 contact Precision@K 的最佳 checkpoint；不是按 test 选择。已准备的目录不可重复 prepare。GPU 空闲后，执行以下命令才会启动新训练：

```bash
"$TRAIN_PY" tools/run_residue_tuning.py run \
  --output /media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_residue_tuning_20261017_lr_v1
```

结果文件为 `promotion.json`、`results.json` 和各组 `trials/<name>/stage_*/`。第一阶段三组共 300 步，第二阶段两组共新增 200 步，总计 500 optimizer steps；实际速度需要实测。

运行器使用独占锁，并转发停止信号让训练保存 checkpoint。重新运行时会核验并跳过已完整完成的阶段。若一个阶段在 100/200 步前中断，会明确报错并保留现场，需检查及恢复该阶段，不能把未完成的结果用于筛选。

## 重复验证

```bash
PYTHONPATH=. "$TRAIN_PY" -m pytest -q tests/test_frozen_embedding_cache.py tests/test_residue_tuning.py
PYTHONPATH=. "$TRAIN_PY" tools/check_frozen_embedding_reuse.py \
  --checkpoint /path/to/residue/checkpoint \
  --output /path/to/new/check_output \
  --device cpu --pairs 4 --microbatch 2 --max-length 16 --steps 2
```

CPU 真实模型核验将模型转为 FP32，避开 CPU 下 FP16 ProtT5 与 FP32 fusion 缺少 CUDA autocast 的精度接口差异。`--device cuda` 使用生产的 BF16 autocast；此检查包含诊断性梯度复制，其耗时也不等于正式训练吞吐。
