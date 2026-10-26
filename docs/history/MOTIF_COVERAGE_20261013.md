# Motif 接触存在性与两侧覆盖率

## 本次改动

保留窗口 16、stride 4、2 层 contact Transformer、gLM2 最后 16 层 Q/V LoRA。
CLIP 和 `sigmoid(max(valid contact logits))` 的互作打分定义保持不变。
在 contact head 的 8 维 pair features 上增加 `Linear(8, 2)`，共 **18 个参数**，
分别输出 A、B 两侧的条件覆盖率；没有复制 Transformer，也没有新增互作分类分支。

对窗口对 `(a,b)`，存在性标签仍是任意已观测残基接触的 OR。
新增 `fA = A 窗口中至少接触 B 一次的残基数 / A 窗口真实长度`，B 同理。
一个残基接触多个对侧残基只计一次。例如 2 个 A 残基与 12 个 B 残基完全连接，
两侧窗口均为 16 时，覆盖率为 `2/16` 与 `12/16`，不是 `24/256`。

## 监督和损失

- 覆盖率只监督**已发生接触、全部真实残基对坐标可观测**的结构 patch。
  有接触但坐标不完整的 patch 仍可参与存在性监督，不能参与覆盖率监督。
- padding 不进入分母；末端短窗口按实际长度计算。交换链时同时转置矩阵并交换覆盖率通道。
- 挖掘得到的负蛋白对只参与原有存在性损失，不赋予额外几何标签。
- 覆盖率采用 FP32 `BCEWithLogits` 连续目标损失。每个结构先对其有效正 patch 和两个通道求平均，
  再对全局 batch 中有覆盖率监督的结构求平均；空监督结构不进入分母。
  该归一化与 microbatch 切分无关，多 GPU 时分母与梯度同步求和。
- 总目标：`0.1 * CLIP + 1.0 * Focal existence + 0.1 * coverage BCE`。
  新覆盖率输出不参与整对蛋白的最终互作分数；在负 patch 上不具有经过监督的条件覆盖率含义。

覆盖率权重通过训练样本 GPU 梯度预检确定：上限 0.1，且加权辅助梯度范数不超过原任务共享参数
梯度范数的 20%。本次原梯度范数 3.60838、未加权覆盖率梯度 0.12112，选中 0.1，比例约 **0.336%**。
这是初始化附近、32 对训练样本上的检查，不证明长期训练中该比例不变，也不证明性能提升。

## 迁移与训练预算

旧任务在 **step 35** 安全保存。迁移逐值保留所有原有训练参数、94 个参数的 Adam 状态、
RNG 状态和全局 step；新增 18 个覆盖率参数单独初始化，优化器状态从零开始。
预检使用的临时更新不进入正式迁移 checkpoint。

数据、种子、抽样顺序、15% mask、学习率/warmup、PPI/DDI 1:1、batch 1024、microbatch 16 不变。
总预算仍为 **1,616 steps / 2 个采样 epoch**，从 35 继续，剩余 1,581 steps；没有重置训练轮次。
存在性、CLIP 和 patch AP 的已有 best 指针继续保留，后续按原验证周期更新。

## 验证记录

- CPU：`32 passed, 2 subtests passed`，覆盖标签、末端窗口、坐标缺失、链交换、
  负对屏蔽、microbatch/GradCache 梯度一致性、优化器参数顺序迁移及既有回归测试。
- RTX 5090 GPU：microbatch 16，32 对 PPI/DDI 混合预检，每链最多 512 残基；
  allocated 峰值 **8.016 GiB**。存在性输出加入辅助头前后逐值一致；原权重和 Adam 状态保留，
  冻结基础权重未改变，新输出和 LoRA 均有梯度；保存再恢复后的参数和优化器更新逐值一致。
- 正式 train/val/test 输入 SHA256 保持不变；测试形式的预检保留输入行数和标签。
- 增加两侧 coverage MAE、平均 MAE，以及 `sqrt(fA*fB) <= 0.25` 的小覆盖 patch 的 MAE、
  存在性分数 ≥0.5 的召回率。这里 0.25 仅用于诊断分组，0.5 不是经过校准的概率阈值。
  指标先按结构求平均，不能把重叠 patch 当独立样本。

迁移后的 step 35 固定 val 基线：CLIP AP 0.20704、互作 AP 0.18578、patch AP 0.34640、
patch P@K 0.33020；新覆盖率输出初始化时 MAE 0.26605。尚未经过覆盖率训练，不能据此声称收益。

## 固定入口与产物

配置：`configs_train/conglab_ecoli_motif16_s4_lora16_coverage_2epochs_v2.json`。
实验目录：`experiments/conglab_ecoli_motif_coverage_v2_20261013/`。

```bash
artifacts/environments/flashppi_torch210_py311/bin/python -u tools/prepare_motif_coverage.py \
  --source experiments/conglab_ecoli_motif16_s4_lora16_2epochs_v1_20261013/motif/checkpoints/step_000035 \
  --config configs_train/conglab_ecoli_motif16_s4_lora16_coverage_2epochs_v2.json \
  --output experiments/conglab_ecoli_motif_coverage_v2_20261013

artifacts/environments/flashppi_torch210_py311/bin/python -u tools/run_motif_training.py \
  --config configs_train/conglab_ecoli_motif16_s4_lora16_coverage_2epochs_v2.json \
  --resume experiments/conglab_ecoli_motif_coverage_v2_20261013/migration/checkpoints/step_000035 \
  --coverage-preflight experiments/conglab_ecoli_motif_coverage_v2_20261013/gpu_preflight/checks.json \
  --output experiments/conglab_ecoli_motif_coverage_v2_20261013
```

已有目录不能直接覆盖重跑。训练日志为 `motif.log`，逐步指标为 `motif/history.jsonl`，
`migration.json` 保存状态迁移审计，`gpu_preflight/checks.json` 保存 GPU 检查与源码/配置哈希。
每 50 steps 验证，每 25 steps 保存恢复 checkpoint；保留 epoch/final checkpoint。
完成后自动评估 final 的官方 E. coli test，test 不参与 checkpoint 选择。

## 首个正式续训步

step 36 已完成，采样序列及全部输入哈希与旧任务一致，学习率按原 81-step warmup 连续推进。
1,024 对中 PPI/DDI 各 512；1,022 对存在可监督覆盖率的 patch，共 311,688 个正 patch。
CLIP loss 6.19889、存在性 Focal 0.48165、未加权 coverage BCE 0.69371，梯度范数 0.36151。
耗时 50.62 秒，峰值 allocated 8.655 GB，无 OOM。核验记录为 `first_step_verified.json`。
