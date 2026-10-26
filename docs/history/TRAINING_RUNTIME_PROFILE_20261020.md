# B600 训练耗时诊断

2026-10-20，用户同意在完整保存点短暂停训、做独立诊断后原样恢复。此次只增加独立诊断工具，未接入提速实现，未改变训练配置或 30 个核心源文件。

正式训练保存于 B step 432。诊断从该完整 checkpoint 恢复，在独立进程中执行 step 433 预热、434 和 435 计时；临时更新的权重全部丢弃，不生成可恢复的诊断 checkpoint。正式训练随后仍从原 step 432 恢复到 B600。

## 实测

RTX 5090 32 GB；FP32 参数、梯度和 AdamW 状态，BF16 训练前向；global batch 1024、microbatch 16、最大长度 512。保留原有冻结编码器单步复用、梯度检查点、15% masking、PPI/DDI 1:1、原负采样以及全部门控。

| 独立测量 step | 整步耗时 |
| --- | ---: |
| 433，预热 | 116.63 秒 |
| 434，计时 | 116.44 秒 |
| 435，计时 | 116.48 秒 |
| 两个计时 step 平均 | **116.46 秒** |

| 主要计算范围 | 平均秒/step | 约占整步 |
| --- | ---: | ---: |
| 整体反向，含梯度检查点重计算 | 41.33 | 35.5% |
| 冻结 ProtT5 前向 | 33.98 | 29.2% |
| gLM2 两遍完整前向合计 | 24.17 | 20.8% |
| 残基级 concat head 前向，含门控 | 7.85 | 6.7% |
| 冻结 ESMC 前向 | 7.78 | 6.7% |
| 其余时间差额 | 1.36 | 1.2% |

计时使用 CUDA events，不在每个模块后同步。以上是 CUDA stream 上的时间区间，可能包含 CPU 发射间隙或等待，并非硬件 kernel 自身运行时间之和。五项主要范围互不包含；“其余”由整步墙钟时间减去上述范围得到，是近似差额。细项中 `glm_frozen_prefix` 包含于 `plm`，`frozen_cache_*` 包含原始冻结前向，`gradient_cache_total` 又包含大部分训练，不能重复累加这些嵌套范围。

正式恢复后的 step 433 为 116.62 秒，与诊断预热相近。仅两个计时 step，尚无优化版本或吞吐提升的实测。

## 对提速优先级的影响

### gLM2 冻结前 17 层复用

第二遍前向中的冻结前缀平均占 **6.216 秒/step，约 5.34%**。其输入/输出为 FP32、不需要梯度；当前输入形状为 `[32, 512, 1280]`。在保持当前裁剪、掩码、精度及后 16 层梯度路径的前提下，具备单步复用的基本条件，但本次没有实现或检验该优化。

若忽略新增传输与管理开销，1616 step 理想上可省约 **2.79 小时**；实际净收益需另测。保存完整 padded FP32 前缀输出，每个 optimizer step 将额外需要约 **5 GiB CPU 缓存**。当前冻结前缀处于原生训练模块中，接入现有缓存还需处理模块边界、模式和回放校验，不能仅加一个缓存开关。

### Padding

两步实际有效位置平均占约 **39.75%**，但 64 个 microbatch 中分别有 **54、51 个**的最长序列已达到 512。在不重新分组的情况下，仅裁去各 microbatch 超过其最长序列的尾部，平均只减少约 **2.31% 的输入位置数**。这是位置数变化，不能直接当作耗时降幅。

更充分的去 padding 需要考虑变长计算、原 microbatch 负例池、位置编号、序列边界、mask 和标签对齐。按长度重新分组会影响当前负例选择，不能视为直接等价的提速。

### 下一步建议

优先进一步区分 **ProtT5 的 attention 和 FFN 耗时**，再决定是否评估高效注意力或局部编译。ProtT5 前向虽占约 34 秒，尚不能据此认定这 34 秒主要消耗在注意力；任何提速比例都需实测。

整体反向的 41 秒同时包含 gLM2、contact head、门控以及检查点重计算，本次未拆成各模块的 backward；不能据此将全部反向开销归因于某一个模块。显存已接近上限，关闭全部梯度检查点或增大 microbatch 仍需单独评估。

gLM2 冻结前缀复用可作为较小幅度的候选优化，现有数据不足以支持只靠它取得大幅提速。

## 诊断隔离和恢复

- 诊断沿用原生 trainer 的完整权重、optimizer、RNG 恢复和源码/输入哈希校验。
- 当前 B600 总预算、学习率、训练样本计划、损失、负采样和门控均不变；不新增 test。
- 原 step 432 checkpoint 的权重及 optimizer 文件在诊断前后 SHA256 一致。
- CPU 回归 **6 项通过**：包含计时钩子保持前向、梯度、AdamW 更新和 RNG 不变，异常时移除钩子，以及诊断失败后仍尝试恢复训练。
- 恢复后的 step **433、434、435** 已全部完成核验；与独立诊断相同 step 的 32 项训练字段逐值相同，包括 loss、梯度范数、门控系数、学习率、mask 统计、负采样及缓存统计。step 433 的 16 层梯度范数也逐值相同。真实规模未额外导出并逐张量比较全部梯度、权重或 AdamW 状态，不能将指标一致扩大解释为已完成这种逐张量检查。
- 先前 step 401–432 和恢复后的 step 433–600 分别保存在独立目录；监督程序检查拼接历史连续，并汇总 450、500、550、600 的正式 validation。

当前正式训练输出：
`experiments/contact_gap_diagnosis_20261018/b_extension_20261020/train_resume_after_runtime_profile_000432`

实际训练/诊断工具工作树：
`/home/zlab/.codex/worktrees/contact-gap-b-extension-20261019/FlashPPI`

证据：

- [汇总数值](../experiments/contact_gap_diagnosis_20261018/runtime_profile_20261020/analysis.json)
- [原始计时](../experiments/contact_gap_diagnosis_20261018/runtime_profile_20261020/probe/results.json)
- [保存点核验](../experiments/contact_gap_diagnosis_20261018/runtime_profile_20261020/pause_verified.json)
- [恢复首步核验](../experiments/contact_gap_diagnosis_20261018/runtime_profile_20261020/first_resumed_step_verified.json)
- [相同步数训练数值对比](../experiments/contact_gap_diagnosis_20261018/runtime_profile_20261020/diagnostic_replay_comparison.json)
- [工具及测试快照](../experiments/contact_gap_diagnosis_20261018/runtime_profile_20261020/code/)
