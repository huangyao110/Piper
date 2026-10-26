> 2026-10-17 22:43 更新：已从 step 1118 的 LoRA 切换为 gLM2 后 16 层全量微调，step 1121 恢复检查通过。当前源码、运行目录和清理记录见 [全量微调运行记录](/media/zlab/lexar4t/FlashPPI/docs/GLM2_FULL16_PRODUCTION_20261017.md)。下文先前运行状态为历史记录。

# gLM2 后 16 层全量微调显存实测

2026-10-17，在本机 NVIDIA GeForce RTX 5090 32GB、PyTorch 2.8.0+cu128 上实测通过。

## 测试设置

- 来源 checkpoint：step 1108，独立载入后合并当前 LoRA 的 Q/V 增量，再移除全部 LoRA 模块。
- 冻结 embedding 与前 17 个 gLM2 block，全量训练后 16 个 block（索引 17–32）的 attention、FFN 和 norm。
- gLM2 可训练参数 325,099,520；加上 heads、融合和门控共 390,744,975。
- ESMC / ProtT5 冻结，既有门控结构保留。global batch 1024、microbatch 16、最大长度 512、PPI/DDI 1:1、15% 掩码及原接触负采样。
- FP32 参数/梯度/AdamW 状态，BF16 autocast，梯度检查点，单步 CPU 冻结 embedding 复用。

## 实测结果

| 项目 | 第一步 | 第二步（AdamW 状态已常驻） |
|---|---:|---:|
| 耗时 | 117.26 秒 | 117.25 秒 |
| PyTorch 峰值 allocated | 22.983 GiB | 25.895 GiB |
| PyTorch 峰值 reserved | 25.205 GiB | 28.082 GiB |

AdamW 状态张量共 2.911 GiB。第二步结束时 CUDA 报告空闲 2.192 GiB，可见总量 31.350 GiB。reserved 包含 allocated，二者不能相加。

另通过 16 对真实长序列、双链均为 512 个非 padding 残基的压力 backward：96 对唯一接触负例，无短缺；峰值 allocated 为 24.422 GiB。该压力批的 CLIP 条目为 32；前两步正式规模测试的 CLIP 条目为 2048。

原 LoRA 最近 50 step 平均 117.08 秒；本次两步平均 117.25 秒。短测速度接近，长期训练/验证/保存开销尚未测定。

## 核验与适用范围

- 2 项 CPU 检查通过：Q/V 合并在 eval 下保持输出、精确解冻指定 block，非法选择在修改前拒绝。
- 两个实测 optimizer step 都确认后 16 层梯度非零、冻结参数无梯度、loss 与梯度有限。
- 原 checkpoint 的 trainable 和 training_state 文件在实测前后 SHA256 一致；实测权重未保存为正式训练 checkpoint。
- 显存可行不代表接触预测一定提高；全量微调是否缓解先前测得的任务冲突，仍需验证指标。
- 本次 optimizer 状态在临时模型中重新建立；正式迁移需要明确定义新解冻参数状态及已有 heads 状态的继承策略。
- 默认 AdamW 的 CUDA foreach 实现可产生额外临时显存，这次实测已沿用默认选择。[PyTorch 2.8 AdamW 文档](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html)

[原始结果](/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/manual_full16_probe_20261017_2207/probe/results.json) · [CPU 检查](/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/manual_full16_probe_20261017_2207/regression_tests.json) · [测试脚本](/home/zlab/.codex/worktrees/frozen-cache-production-20261017/FlashPPI/tools/probe_full16_memory.py)

## 正式训练已恢复

2026-10-17 22:17，原 LoRA 训练从 step 1108 恢复，已完成 step 1109。配置、源码、输入、采样计划和学习率均通过核验；原生恢复 optimizer/RNG。正式运行没有使用临时全量微调的两步权重。

- 当前目录：`/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_full16_probe_001108`
- supervisor/trainer PID：1362066/1362839
- [恢复核验](/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/manual_full16_probe_20261017_2207/resume_verified.json)
