# CongLab motif：两轮训练

> 2026-10-13 更新：本阶段在 step 35 保存后迁移到存在性＋两侧覆盖率版本。
> 总预算仍为 1,616 steps，当前配置、日志和验证记录见
> [覆盖率续训记录](MOTIF_COVERAGE_20261013.md)。下文保留本阶段历史。

## 固定模型与数据

根据用户 2026-10-13 的指示，仅训练 motif 版本。旧残基对照在 step 11
安全保存后停止；旧队列中的 100-step patch 训练取消，已有记录保留。

- 从原始 gLM2-650M 预训练权重初始化，CLIP/contact 任务头重新初始化，不继承对照权重。
- Q/V LoRA 仅最后 16 层（零起始索引 17–32），rank 8、alpha 16、dropout 0.05。
- motif 窗口 16、stride 4；先池化残基 embedding，再进入 2 层 contact Transformer。
- patch label 为窗口对中存在任一已观察接触；负 patch 要求所有真实残基对均可观察。
- 保持已审计的 CongLab 训练集和小 val，最终 test 仅官方 E. coli，样本与标签不变。
- 训练候选 827,305 条：PPI 256,596，DDI 570,709。数据来源、过滤差异及限制见
  [数据审计记录](CONGLAB_ECOLI_FRESH_LORA_20261013.md)。

## 训练预算与采样

- 每个全局 batch 为 1,024 对，PPI/DDI 各 512；microbatch 16，界面 crop 每链最多 512 残基。
- PPI 均匀抽 cluster 后抽结构记录；DDI 均匀抽结构记录。两者均有放回。
- 一个采样 epoch 为 `ceil(827305 / 1024) = 808` steps，2 epoch 共 **1,616 steps**。
  总采样 1,654,784 次，PPI/DDI 各 827,392 次；不意味着每条记录恰好被遍历两次。
- 训练独立随机 mask 15% 残基，验证和测试不 mask。
- 损失为 `0.1 × CLIP + 1.0 × Focal contact`；任务头峰值 LR 1e-4、LoRA 1e-5。
  AdamW weight decay 0.1，warmup 为总步数的约 5%（81 steps），随后 cosine 衰减。
- 每 50 steps 和每轮末尾做 val；每 25 steps 保存恢复 checkpoint。
  保留各 val 指标的 best 指针、每轮 checkpoint 和 final checkpoint。
- 完成 step 1,616 后自动评估官方 E. coli；test 不参与 checkpoint 选择。

## 启动与监控

固定参数：`configs_train/conglab_ecoli_motif16_s4_lora16_2epochs_v1.json`。
环境：`artifacts/environments/flashppi_torch210_py311/bin/python`，RTX 5090 32 GB。

```bash
artifacts/environments/flashppi_torch210_py311/bin/python -u tools/run_motif_training.py \
  --config configs_train/conglab_ecoli_motif16_s4_lora16_2epochs_v1.json \
  --preflight-config configs_train/conglab_ecoli_patch16_s4_fresh_lora_v1.json \
  --preflight-experiment experiments/conglab_ecoli_fresh_lora_v1_20261013 \
  --output experiments/conglab_ecoli_motif16_s4_lora16_2epochs_v1_20261013
```

输出目录下 `motif.log` 为训练日志，`motif/history.jsonl` 为逐步指标，
`motif/status.json` 为训练进度，`status.json` 为调度状态。
`test_ecoli_final/` 保存最终测试结果；`data/active_motif_run.json` 指向当前任务。

复用已通过的 motif GPU 预检（microbatch 16、512 残基，峰值 allocated 8.000 GiB），
不是重新训练对照。启动器逐项确认参数只改了 steps、epochs、warmup_steps，
核验训练源码未变，并记录 `launch_checks.json`。正式全局 batch 为 1,024，
显存和耗时以实际训练日志为准。新配置的全部 1,616 个学习率步已单独检查。
启动器拒绝覆盖已有训练目录；SIGTERM 会让训练器完成当前步、保存后停止，并取消后续 test。

## 首步运行核验

正式 step 1 已完成：1,024 对中 PPI/DDI 各 512，耗时 49.99 秒，
峰值 allocated 8.144 GB。CLIP loss 7.6689、patch Focal loss 0.72923、
梯度范数 2.9601，实际 mask 比例 14.9278%。首步 LoRA/任务头学习率分别
为 `1e-5/81` 和 `1e-4/81`，符合新 warmup。无 OOM；这仅验证运行正常，
尚不能判断收敛或改进效果。证据保存在实验目录 `first_step_verified.json`。
