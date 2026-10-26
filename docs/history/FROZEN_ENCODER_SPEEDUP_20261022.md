# 冻结编码器提速实测与正式接入（2026-10-22）

> 后续更新（同日）：用户同意尝试数值容差方案后，新的 10 步长度分桶对照通过检查，已选择候选 step 1711 接续训练。详见 [长度分桶验收报告](FROZEN_LENGTH_BUCKET_TRIAL_20261022.md)。以下保留此前逐位一致实验及当时决策的原始记录；当前配置以新报告和 `experiments/active_training.json` 为准。

已接入 **gLM2 冻结前 17 层的单步内 CPU 缓存**。长度分桶保留为实验开关，本次未启用。训练从原始 step 1679 接回，目标仍为 step 2616。

## 实测结果

设备 RTX 5090 32 GB；同一 step 1679 checkpoint、Adam、RNG 和采样计划。完整 batch=1024，microbatch=16，最大长度 512，BF16 训练。

| 版本 | Step 1680（预热） | Step 1681 | Step 1682 | 后两步平均 |
|---|---:|---:|---:|---:|
| 原版 | 112.905 s | 112.593 s | 112.636 s | 112.615 s |
| 冻结前缀复用 | 107.008 s | 106.775 s | 106.805 s | 106.790 s |

计算耗时减少 **5.17%**，吞吐提高约 **5.45%**。计时包含前向、反向、梯度裁剪和 Adam update；剔除诊断哈希/导出开销，不含启动、数据读取、val、checkpoint 写入。三步小样本性能结果不能等同于整次训练总耗时减少同一比例。

三步的完整输入（tokens、mask、labels、顺序）、实际选择的有向负例索引、全部裁剪后梯度、更新后 173 个可训练张量、Adam、CPU/CUDA/Python/NumPy RNG、loss 与训练指标逐项完全一致。比较使用完整张量 SHA-256。CPU 回归另外直接比较原始梯度和 Adam 张量。

## 为什么没有接入长度分桶

冻结 ESMC、ProtT5 按当前 step 裁剪/masking 后的序列长度分组计算，再恢复原 microbatch 的顺序与 padding。模型参数、序列、标签、负例池以及采样规则均未改变。

一个完整 step 从 112.905 s 降到 **85.786 s（约减少 24.0%）**。但它不满足本次 GPU 测试前明确写下的完全一致准入条件：

- 在固定的前 4 个 microbatch、共 128 条蛋白的编码器对照中，ESMC 输出完全一致；ProtT5 输出的相对 L2 差异为 **1.299%**。
- 原版/分桶版 contact loss 为 0.4232432265 / 0.4232281386，CLIP loss 为 2.7522335052 / 2.7523014545。仅看 loss 会低估差异。
- 实际选择的负例索引不同；hard-seed 数为 2922 / 2920，补充负例数为 2634 / 2636，总负例数都为 5556。
- gLM2 梯度相对 L2 差异 **1.009%**，contact head 梯度相对 L2 差异 **1.539%**。

改变计算形状后，ProtT5 的浮点输出发生差异，并伴随负例排序及训练更新差异。因此当前只启用通过完全一致验证的前缀复用。分桶只测了一步，未完成稳态性能验证；这些差异也不能说明最终准确率一定下降。后续若推进分桶，需要单独验证数值来源和训练效果，不能把它称为与原训练逐项等价。

## 实现和状态保持

- 新增 `flashppi/frozen_prefix_cache.py`，仅在一个 GradCache step 内缓存冻结边界张量。原生 gLM2 继续处理位置、mask、unpadding；后 16 层及其 activation checkpointing 保持原计算。
- 原精度缓存，无量化或 dtype 转换。每步捕获/复用 64 次，增加 **5.00 GiB CPU 缓存**；现有 ESMC/ProtT5 缓存为 5.75 GiB。
- 缓存校验 tokens、逻辑 mask、调用顺序、autocast、冻结参数版本和随机数行为；step 结束或异常时恢复原方法并清空。训练模型参数名、参数顺序及 checkpoint 权重格式不变。
- 新配置只有 `"frozen_prefix_cache": "cpu"`。`frozen_length_bucket_size` 默认 0，在正式运行中保持关闭。
- 33 项 CPU 检查通过，覆盖两步梯度/Adam/RNG/负例一致性、padding、二维 packed 和三维边界、activation checkpoint、变更拒绝及异常清理。
- 显式 checkpoint 元数据迁移保留原权重、Adam、RNG、步数、数据哈希和历史最佳 checkpoint 引用，仅更新允许的配置/源码哈希。原生严格 resume 校验没有绕过入口。
- `conglab_ecoli_v1_locked`、train/val/test 成员、PPI/DDI 1:1、簇均衡和去重权重、crop/masking、残基标签、门控、负例策略、loss 权重、学习率、warmup、总步数均保持原样。

## 正式运行核验

原运行在 step 1679 完整保存退出。诊断的更新全部丢弃，正式运行从这个原始保存点恢复。

正式 step **1680**：计算 **106.991 s**，contact loss **0.4232432265**，CLIP loss **2.7522335052**；与原版诊断同一步的训练指标、后 16 层梯度范数、门控参数和采样统计完全一致，完整采样计划哈希一致。

当前运行：`experiments/frozen_encoder_speedup_20261022/train_prefix_001679`。后续查看实时状态请以 `experiments/active_training.json` 为准，避免把诊断目录中的 step 1682 当成正式进度。

证据目录：`experiments/frozen_encoder_speedup_20261022/`，包括 `acceptance_policy.json`、`cpu_tests.xml`、`source_changes.patch`、`prefix_comparison.json`、`bucket_comparison.json`、`analysis.json`、`migration/migration.json` 和各诊断日志。诊断大张量/权重在报告完成后清理，检查哈希和 checkpoint 元数据保留；原始恢复点、迁移恢复点及正式运行目录保留。
