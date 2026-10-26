# B 组从 200 续训至 400 step

2026-10-20 后续：用户已要求继续训练，[B400 → B600](CONTACT_GAP_B600_CONTINUATION_20261020.md) 于 18:50 启动。以下保留 B200→400 及 B400 test 的已完成记录。

用户于 2026-10-19 明确要求继续延长 B。北京时间 11:09 启动，恢复 B200，连续训练 step 201–400，17:43 正常完成，最终 checkpoint 已保存。用户随后要求检查 test，B400 于 20:50 完成一次完整 E. coli 评估。训练及监督进程均已退出，未自动继续训练。训练记录见 [动态状态](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/status.json)。

## 完成结果与完整 test

B400 的 val 残基 P@K 为 0.128164，相对 B200 的 0.114963 提高 11.48%，是本阶段按该主指标选出的 checkpoint。val Contact 互作 AP 为 0.582036，CLIP AP 为 0.549634；两项互作指标均有波动。B381–400 的训练均值：Contact loss 0.302737、CLIP loss 0.877429，均低于 B181–200。

以下 test 在选择 B400 之后执行，沿用同一份 650 正例、65,000 负例、全长、原始方向、无过滤的 E. coli benchmark。模型和评估源码、冻结模型及数据哈希与旧版一致，预测行顺序及指标重算通过。

| Test 指标 | 上次最终 step 1616 | B400 | 相对变化 | 官方复测 |
| --- | ---: | ---: | ---: | ---: |
| Contact 互作 AP | 0.146702 | 0.199554 | +36.03% | 0.295202 |
| 残基 P@K，12 Å | 0.102194 | 0.140650 | +37.63% | 0.182276 |
| 残基 P@K，8 Å | 0.051598 | 0.081768 | +58.47% | 0.107526 |
| CLIP AP | 0.244315 | 0.264520 | +8.27% | 0.250087 |
| CLIP AUROC | 0.862234 | 0.853630 | −1.00% | 0.884567 |

Contact 阈值 0.5 下，TP 124 → 144，FP 332 → 257，Precision 0.271930 → 0.359102，Recall 0.190769 → 0.221538。接触预测有改善，与官方仍有差距。CLIP AP 在本次 test 高于旧最终模型，因此不能仅凭小 val 末次回落认定 CLIP 全面退化；CLIP AUROC 仍有下降。

比较基准是上次正式最终 step 1616，B 阶段起点则为旧 step 1500；不能把上述 test 差值全部归因于 B200 → B400 的额外 200 步。没有测试其他 B checkpoint 来选择模型。

证据：[完整 test 对比](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/test_ecoli_b400/comparison.md)、[原始指标](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/test_ecoli_b400/metrics.json)、[对齐与重算核验](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/test_ecoli_b400/comparison_audit.json)。

## 配置与评估

| 项目 | 本次设置 |
| --- | --- |
| 来源 | B 组 `train_000100_000200/checkpoints/step_000200` |
| 步数 | 保留 B 阶段计数 200，新增 200 步，终点 400 |
| Backbone LR | 常数 `3e-5` |
| Head / fusion / gate LR | 常数 `1e-4` |
| Warmup | 不重启；沿用已经完成的 20 步 warmup |
| 优化器 | 原 AdamW moments、计数及参数分组全部保留 |
| 可训练主干 | gLM2 block 17–32，共后 16 层 |
| ESMC / ProtT5 | 继续冻结，单步内 CPU embedding 缓存继续启用 |
| 门控和 head | 原残基级 contact head、残差通道 gate、CLIP residue gate |
| 批次 | 每步 1024 对，PPI / DDI 各 512，microbatch 16 |
| 其他训练策略 | 裁剪、mask、负例规则、loss 权重与 B200 一致 |
| 验证及保存 | 250、300、350、400；主指标固定 val 残基 P@K |
| 硬件 | 单张 RTX 5090 32 GB |

B 阶段最初来自旧模型的 step 1500；本页的 200 / 400 均为该阶段计数。本次仅延长 B，A 保留在 200。新增结果用于观察 B 相对 B200 的变化；不能把 B400 与 A200 当作同预算的学习率比较。

B200 的固定 val 基准：残基 P@K **0.114963**、Contact 互作 AP **0.574959**、CLIP AP **0.590783**。后续每次验证自动写入 [比较表](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/validation_comparison.md) 和 [配对统计](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/validation_comparison.json)。原 B200 及历史最佳 checkpoint 均保留。训练过程中未新增 test，也未以 test 选 checkpoint；用户在训练完成后另行要求的 B400 test 记录在本页上方。

启动时按上一段约 116–117 秒/step，估计新增 200 步约需 6.5 小时，预计 17:45–18:00 左右结束；实际于 17:43 完成训练、验证及保存。

## 续训一致性与采样处理

原采样器先抽取整个预算的 DDI，再用同一 NumPy RNG 做批内洗牌。因此，仅把总步数 200 改成 400，会改变已经执行过的前 200 步的批内顺序。

本次增加可选 `sampling_block_steps=200`：PPI 抽样流延续，DDI 抽样和批内洗牌按固定 200 步块使用连续 RNG。未配置该选项时，原逻辑保持原样。真实数据核验确认前 204,800 个抽样索引及其顺序与 B 原计划完全相同，后 200 步为新追加批次；全部 400 个批次仍各含 512 PPI 和 512 DDI。之后若继续延长，应保留 200 步块设置，并按完整块扩展。

训练核心只修改 `flashppi/mixed_sampling.py` 和 `flashppi/research_train.py`，后者仅传递该选项并记录采样元数据。为延长预算显式生成新的恢复目录，保留原有源码、输入、配置、checkpoint 的严格校验。原 A/B 目录、权重及代码均未改写。

[真实 checkpoint 核验](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/preparation.json) 已通过：

- 权重文件 SHA256 相同；173 个可训练张量及其 optimizer 映射一致。
- Adam 全部状态、随机数状态、step 200、输入哈希完整保留。
- 历史最佳模型的相对路径已重定位，解析后仍指向同一文件。
- 配置仅变更 `steps: 200 → 400`，以及新增 `sampling_block_steps: 200`。
- 原 200 步采样前缀 SHA256：`a8a06862d19d3a08c4f3ea97eefcf262b6bd5532e6d036323c02d2d7ea182e9b`。
- 新 400 步采样 SHA256：`0f51d5162d6f456a93ec43c553924fcbd45e10c8cc06f56dd9e9c23108c2931b`。

## 检查与后台记录

CPU 检查覆盖默认采样兼容性、扩展前缀一致性、非法块大小、学习率衔接、Adam 下一次更新逐值一致，以及监督程序识别错误 step / LR / 负例 / 非有限 loss。相关分布式与混合采样回归也通过：第一次 11 项，加入监督程序检查后的第二次 13 项，合计 19 个不同用例。真实 B200 的 100 个结构验证结果也已用于检验汇总流程。详见 [checks.json](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/checks.json)。

11:12 已确认首个 GPU 更新完成，实际 **step 201**，LR 为 `3e-5 / 1e-4`，16 个 block 均有有限非零梯度，缓存捕获及复用各 128 次，PPI / DDI 各 512 对；额外负例无重复、无 self、无预算缺口。单步耗时 116.68 秒，Contact loss 0.342227、CLIP loss 1.141141。详见 [首步核验](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/first_step_verified.json)。单步 loss 不代表趋势，后续结合 20 步窗口及固定 val 判断。

- 独立代码：`/home/zlab/.codex/worktrees/contact-gap-b-extension-20261019/FlashPPI`
- 输出：`experiments/contact_gap_diagnosis_20261018/b_extension_20261019/train_000200_000400`
- 后台入口：独立代码目录下的 `tools/run_b_extension.py`
- [启动记录](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/supervisor.launch.json)、[原生训练命令](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/train.launch.json)
- [准备及监督代码快照](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/diagnostic_code/sha256.json)
- [原生训练日志](../experiments/contact_gap_diagnosis_20261018/b_extension_20261019/train_000200_000400.log)

监督程序每 15 秒刷新状态，每次验证后更新比较；若异常则记录失败并请求训练进程保存停止，不自动更改参数或重启。正常完成后验证 200 次新增更新、4 次验证及最终 step 400 checkpoint，再标记完成。
