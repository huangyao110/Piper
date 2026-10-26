# 冻结编码器长度分桶：短程对照与接入（2026-10-22）

用户同意尝试此前约 24% 的提速后，完成了新的数值诊断、16 项 CPU 检查和 10 步配对训练。候选通过运行前写下的容差检查，已选择从候选 step 1711 继续正式训练，目标仍为 step 2616。

当前配置为 `frozen_prefix_cache="cpu"`、`frozen_length_bucket_size=32`。ESMC/ProtT5 对本步已经裁剪、masking 的输入按长度分组计算，随后恢复原顺序和 padding；gLM2 冻结前缀继续单步复用。生产模型和训练源码沿用前次固定版本，本次没有再次修改。

## 对照设计与速度

两组从同一 step 1701 的权重、Adam、RNG 状态出发，分别完成 step 1702–1711。基线已启用冻结前缀复用；候选在此基础上增加长度分桶。设备为 RTX 5090 32 GB，batch 1024、microbatch 16、最大长度 512，训练使用原 BF16 autocast。

| 指标 | 基线：前缀复用 | 候选：前缀复用＋长度分桶 |
|---|---:|---:|
| 去掉首步后 9 步平均计算时间 | 106.9274 s | 79.1334 s |
| 10 步平均 contact loss | 0.41277233 | 0.41274882 |
| 10 步平均 CLIP loss | 2.72676051 | 2.72661455 |

相对当前已启用前缀复用的版本，计算耗时减少 **25.99%**，吞吐提高约 **35.12%**。这两个百分比含义不同。计时包含前向、反向、梯度裁剪和 Adam update，排除诊断导出、启动、数据读取、val 和 checkpoint 写入；不能据此声称整次训练总时间必然减少同一比例。此前约 24% 来自不同起点的一步实验，不能与本次百分比相加。

## 固定 val 的实际结果

评估使用相同冻结文件、全长输入、FP32 权重/FP16 autocast、原方向和原聚合规则。互作集 228 对（25 正、203 负），结构集 100 条。没有用 test 调整开关。

| 指标 | 共同起点 step 1701 | 基线 step 1711 | 候选 step 1711 | 候选减基线 |
|---|---:|---:|---:|---:|
| 残基接触 Precision@K | 0.141851 | 0.141004 | 0.141038 | +0.000035 |
| 接触头互作 AP | 0.677038 | 0.674368 | 0.674875 | +0.000507 |
| 接触头互作 AUROC | 0.906010 | 0.920394 | 0.920788 | +0.000394 |
| CLIP AP | 0.751794 | 0.690908 | 0.683517 | −0.007391 |
| Patch AP（诊断网格） | 0.219215 | 0.216763 | 0.216689 | −0.000074 |

接触相关指标在这次短程对照中接近。CLIP AP 的候选结果略低，绝对差为 0.007391，处于预设 0.01 容差内；不能把“通过容差”描述成“所有准确率完全不变”。两组 CLIP AP 都比共同起点下降，基线也存在这一变化，因此不能把从起点开始的全部下降归因于分桶。

这是一次 **10 步工程验收**，不是长期准确率等价的统计证明，也没有证明模型整体效果变好。后续沿用既有固定 val 节点观察，下一节点为 step 1716。模型仍以残基级 contact label 训练；patch 指标仅用于诊断。

## 数值差异来自哪里

在第一批真实训练数据中，固定 8 条蛋白（长度 108、119、53、93、55、126、118、39），比较宽度 512 与 126 的 padding，并单独比较 batch 反序：

| 比较 | 有效残基输出相对 L2 差异 |
|---|---:|
| BF16，仅 batch 重排 | 0 |
| BF16，仅缩短 padding | 0.796902% |
| FP32，仅缩短 padding | 0.0000983% |

原生 T5 的残基 token、mask token、EOS、attention mask、有效位置相对位置偏置均精确一致。BF16 的差异在第一层 Transformer block 后即可观察到。FP32 下明显缩小的差异支持“padding 改变计算形状后，低精度舍入发生变化”这一解释；本次没有定位到某个具体 GPU kernel，也不据此保证所有输入情形都等价。

FP32 仅用于诊断。模型原有 171 个 FP16 和 24 个 FP32 参数张量逐一恢复到各自 dtype，完整冻结权重哈希与诊断前一致，没有把诊断精度或参数改动带入训练。

## 输入、梯度与负例

- 10 步的完整输入哈希一致，包含 tokens、mask、标签和顺序；采样计划、数据文件哈希、31 个生产源码文件哈希一致。两组初始 val 完全一致；173 个可训练张量的名称、顺序和大小一致。
- 第一批全部裁剪后梯度相对 L2 差异 **0.84417%**，余弦相似度 **0.99996437**。10 步的梯度范数最大相对差异 **0.26042%**，没有非有限 loss 或梯度范数。
- 逐步 contact loss 最大相对差异 **0.02491%**，CLIP loss 最大相对差异 **0.02075%**；平均 loss 的相对变化分别为 −0.00569%、−0.00535%。
- 负例候选池构造和预算保持原规则，逐步可用数量、实际总数、重复数、自配对数和缺额一致。实际有向负例索引的替换比例为 **0.816%–1.757%**。小幅 embedding 差异确实影响了部分排序，训练轨迹并非逐位一致。

数据过滤协议 `conglab_ecoli_v1_locked`、原 CongLab 成员、扩展数据索引、train/val/test 文件与哈希、PPI/DDI 比例、簇均衡/去重权重、裁剪/masking、标签、门控、负例选择算法、loss 权重和学习率全部沿用原设置。

## 预设准入、恢复与证据

GPU 实验前保存了 `acceptance_policy.json`：均值 loss 最多增加 2%，逐步 loss 差异最多 5%，梯度范数差异最多 10%，首步全梯度相对 L2 最多 5% 且余弦至少 0.99；固定 val 的最大绝对下降按指标分别为 0.005 或 0.01；计算耗时至少减少 10%。22 项检查全部通过，运行后没有放宽门槛。

原训练在完整 step 1701 checkpoint 暂停。显式元数据迁移仅设置 `frozen_length_bucket_size=32`，保持权重、Adam、RNG、步数和数据/源码哈希，warmup 没有重启；原生严格 resume 校验保留。候选完成的 10 次 optimizer update 被保留，从候选 step 1711 接续，不重复这 10 步。

CPU 检查命令（项目 Python 环境，禁用 GPU）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  artifacts/environments/esmc_prott5_gated_py312/bin/python -m pytest -q \
  tests/test_frozen_bucket_alignment.py tests/test_frozen_encoder_speedup.py
```

结果 **16 passed**，两个 SWIG 弃用警告。新增检查覆盖真实 ProtT5 包装层的残基/token/EOS/context/位置对齐、重复序列、分桶顺序恢复、缓存复用和空 rank。

证据目录为 `experiments/frozen_length_bucket_trial_20261022/`。主要文件包括 `comparison.json`、`acceptance_policy.json`、`paired_start_verified.json`、`preflight_provenance.json`、`baseline/numerical_probe.json`、两组 `trial_steps.json`/`val_final/metrics.json`、`migration/migration.json`、`decision.json` 和 `launch.json`。

正式输出为 `experiments/frozen_length_bucket_trial_20261022/train_prefix_bucket_001711`，实时运行以 `experiments/active_training.json` 为准。原始 step 1701、迁移恢复点和两个 step 1711 对照 checkpoint 均保留。

正式首步 **1712** 已完成：计算 **79.9932 s**，contact loss **0.40386859**，CLIP loss **2.80564070**；后 16 层梯度均有限且非零。冻结编码器捕获/复用各 128 次，gLM2 前缀各 64 次，长度分桶已生效。原生 resume 恢复 Adam/RNG，无额外配置变化，数据/源码哈希、采样计划和参数顺序与候选一致，核验保存于 `formal_resume_verified.json`。

正式续训核验后仅删除两份临时 `first_gradients.pt`，释放 **2.91 GiB**；完整梯度比较结果、文件哈希及删除记录保存在 `comparison.json` 和 `diagnostic_cleanup.json`，checkpoint 与评估结果保留。
