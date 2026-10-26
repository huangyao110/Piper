# Contact 与 CLIP 诊断

固定 checkpoint：step 1102。检查 3 个完整训练批次，每批 1024 对，microbatch 16，最长 512。

沿用正式训练的 BF16、15% 掩码、接口裁剪、PPI/DDI 1:1 和原负采样。CLIP 使用完整 2048 个方向条目。
所有计算均未执行 optimizer.step；前后可训练参数指纹一致。

## 接触损失分解

| 项目 | 平均 loss | 占比 | 平均预测分数 | 分数 ≥ 0.5 的比例 |
|---|---:|---:|---:|---:|
| 真实接触点 | 0.294596 | 78.68% | 0.2417 | 8.7000% |
| 结构内非接触点 | 0.058314 | 15.58% | 0.0416 | 0.0085% |
| 额外负蛋白对 | 0.021493 | 5.74% | 0.0124 | 0.0004% |

loss 保留正式训练的正接触数分母以及 microbatch 权重；未观测位置与 padding 均排除。预测分数和阈值比例在所有有效位置上汇总，不等同于全长验证/test指标。

## 参数梯度

CLIP 梯度已乘正式训练权重 0.1，Contact 权重为 1。以下为各批次的均值与余弦范围。

| 参数组 | Contact 范数 | 加权 CLIP 范数 | CLIP/Contact | 余弦均值（范围） | 负余弦批次数 |
|---|---:|---:|---:|---:|---:|
| all_shared | 0.070575 | 0.093439 | 1.330 | 0.1270 (0.0309～0.2175) | 0/3 |
| shared_lora | 0.062439 | 0.077806 | 1.264 | -0.1543 (-0.1763～-0.1225) | 3/3 |
| shared_fusion | 0.031749 | 0.051471 | 1.652 | 0.9308 (0.8666～0.9719) | 0/3 |
| contact_head | 0.136811 | 0.000000 | 0.000 | 不适用 | 0/3 |
| contact_gate | 0.000777 | 0.000000 | 0.000 | 不适用 | 0/3 |
| clip_heads_and_gates | 0.000000 | 0.288363 | 不适用 | 不适用 | 0/3 |

余弦 > 0 表示该批次两个梯度整体同向，< 0 表示存在一阶方向冲突；不能由此直接推断长期泛化或 AdamW 实际更新。

## 核验与限制

- 8 项 CPU 检查通过；包含独立损失/导数核验、无效位置排除、两个任务梯度之和与联合训练梯度的比较。
- 两次正式规模回放的 loss 和负采样计数完全一致；冻结编码器无梯度。
- 分析的是固定权重下 3 批训练数据；没有比较不同学习率或不同损失权重的训练效果。
- 本次不能直接确定正接触损失随历史 step 的变化，也不能证明某个参数调整一定改善验证定位。

原始结果：`diagnostics/diagnostics.json`；逐批结果：`diagnostics/batch_*.json`；续训核验：`resume_verified.json`。

## 解释与下一步

- 三批均以真实接触点为最大损失来源；不能仅凭这个占比推断正样本权重不足或长期训练轨迹。
- 共享 LoRA 的余弦范围为 -0.1763～-0.1225，三批均为负；加权 CLIP/Contact 梯度范数比为 1.13～1.48。
- 各批出现负余弦的 LoRA 层数为 [14, 16, 16]，共 16 层。
- 在原始梯度的一阶几何关系中，CLIP 使 LoRA 联合梯度沿 Contact 梯度方向的投影减少约 13.9%～26.1%。这不是 AdamW 更新幅度，也不是模型性能损失比例。
- 融合层三批同向（平均余弦 0.9308）；所有共享参数合起来仍略同向（平均余弦 0.1270）。因此观察到的是局部的任务冲突，不能认定 CLIP 整体有害。
- concat 接触头与接触 gate 的 CLIP 梯度为零，与模型分支结构一致；最近 50 个正式训练 step 均未触发全局梯度裁剪。
- 建议优先验证只减弱 CLIP 回传共享 LoRA 的梯度，保留 CLIP 头、融合层、门控结构和冻结编码器设置。是否有效以验证集残基接触 P@K 为准。
- 本次只做诊断，未改变学习率、损失权重或模型结构；按原状态自动续训。

## 续训核验

2026-10-17 21:59，已从 step 1102 原生恢复并完成 step 1103。

- 当前训练目录：`/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/motif_resume_after_diagnostics_001102`。
- supervisor PID：1360135；trainer PID：1361106。
- 配置、来源代码、输入、采样计划与学习率安排核验通过；使用原 checkpoint 的 optimizer/RNG 状态。
- 续训首步的 Contact loss、CLIP loss、负采样数量与诊断第一批完全一致。
- 分任务梯度相加所得范数与正式联合梯度范数的相对差为 6.1859101e-06（仅范数比较；完整梯度线性检查在 CPU 回归测试中完成）。
- 单步冻结 embedding 缓存仍启用：128 次捕获 / 128 次复用；首步耗时 117.92 秒。

[原始诊断结果](/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/manual_contact_diagnostics_20261017_2143/diagnostics/diagnostics.json) · [续训核验](/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_residue_gated_v1_20261015/manual_contact_diagnostics_20261017_2143/resume_verified.json) · [诊断脚本](/home/zlab/.codex/worktrees/frozen-cache-production-20261017/FlashPPI/tools/diagnose_contact_objectives.py)
