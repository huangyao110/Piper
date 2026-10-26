# FlashPPI PPI-only 微调准备

当前已完成数据、训练入口和短训练验收；尚未启动正式 1,000 步微调。原始数据库只读，官方 E. coli test 的样本、标签及既有基线保持不变。现有 `AGENTS.md` 未修改。

## 固定输入与数据划分

- 原库：`/media/zlab/lexar4t/ppidb/distributions/ppidb-training-20261009_101259`。
- 划分脚本：`tools/prepare_ppi_finetune_split.py`，参数由其固定 Policy 定义。先排除 test 同源，再抽取 500 个 val 正对，再排除 val 同源，最后训练采样聚类及来源统计。严格采用任一端 identity >30%、双向 coverage >50% 即排除的规则；论文正文与 Methods 的措辞差异见划分 manifest。
- 固定结果：`data/ppi_finetune_split_v1/`。训练含 1,120,675 个结构记录、153,001 个不同序列对、39,198 个采样簇。划分脚本的工作文件在临时目录中清理，仅保留最终数据和审计清单。
- 起始模型：官方 `flashppi_noecoli`，revision `aebaded7533d1e27d83a855b13b6499352e82894`。不能用随机初始化训练入口替代。新入口拒绝缺失、多余或形状不匹配的权重，并校验模型和划分文件 SHA256。

## 固定验证与选模

`tools/prepare_finetune_validation.py` 保留全部 **500** 个正对做接触图评估；互作 AP 子集包含 **81 正对 + 4,119 候选负对**，覆盖 10 个物种，正对覆盖率 **16.2%**。

互作子集要求 UniProt 序列比对确认对应关系（identity、片段覆盖均至少 90%）、同物种、无歧义 STRING v12 映射；排除同一 STRING 蛋白的配对。不能依据 `original_id` 顺序或 L/R 标记直接推定序列对应关系。负对仅从同物种验证蛋白构造，排除原始 PPI 库已知互作（序列对及可映射蛋白对）和 STRING v12 physical score ≥700 的互作。每物种至多取每正对 100 个负对，候选不足时取全部，不补跨物种负对。

固定数据、排除原因及公共接口响应存于 `data/ppi_finetune_validation_v1/`。接口响应是可复现所需的最终来源证据；可离线重放，结果已验证逐字节一致。未观察到的配对仍是推定负例，不能视为实验确认阴性。

| 未微调官方模型的 val 基线 | 数值 |
| --- | ---: |
| 接触最大分数的互作 AP（主选模指标） | 0.567471 |
| 互作 AUROC | 0.932307 |
| CLIP AP（辅助） | 0.645173 |
| 接触图平均 precision@K，K 为有效真实接触数 | 0.226836 |

结果见 `runs/finetune_validation_baseline/`。验证不截断序列，使用 FP32 权重、FP16 autocast、原始方向和 `sigmoid(max(valid contact logits))`。结构缺失位点仅从接触监督和接触 precision@K 中屏蔽，不从序列编码或互作打分中移除。

每 100 步依据固定 val 互作 AP 保存训练期间最佳模型，同时记录全部 500 个结构的接触指标。最终必须与未微调基线比较；如果没有改善，应保留原模型。val 子集较小且与官方预训练数据的历史重合尚未完整核验；它不能替代最终独立 test，也不能将其 AP 与 E. coli test AP 直接比较。test 不参与调参或逐 checkpoint 选模。

## 训练配置与已修复问题

固定配置：`configs_train/finetune_head_v1.json`。冻结整个 gLM2 骨干，训练原有 CLIP/接触头及 logit scale；不改变网络结构、接触输出或打分定义。LoRA 留待这一基线完成后单独比较，本入口当前只支持 head-only。

参数：seed=20261010，batch=64，GradCache chunk=8，训练截取前 512 残基，BF16，AdamW lr=1e-5、weight_decay=0.01，50 步 warmup，梯度裁剪=1，CLIP/contact loss 权重=0.1/1，focal loss；hard/self/random negative ratio=2/1/0，启用双向配对池。仅使用 PPI，按训练蛋白簇均匀抽样，再在簇内抽结构。候选不足时实际负例数会少于请求数。接触损失为按微批样本数加权的 focal loss 均值，因此 chunk 大小属于固定实验参数。

修复包括：self-negative 正确生成 `(A,A)`；hard-negative 不再回填被屏蔽正例；随机负采样不绕回原配对；交换序列时同步交换结构掩码；GradCache 未定义变量及 logit-scale 梯度权重；BF16/FP16 切换时重建官方代码未按 dtype 刷新的非持久 RoPE 缓存。

## 运行与复现

从仓库根目录运行，已验证解释器为 `/home/zlab/miniconda3/envs/esmfold2/bin/python`，GPU 为 RTX 5090 32GB。依赖列于 `requirements-finetune.txt`，实际版本记录见 `runs/finetune_preflight/environment.json`。输出目录必须不存在。

```bash
# CPU 回归检查，无需下载权重
python tests/test_finetune_training.py

# 离线重放固定验证集；原始数据库仍只读
python tools/prepare_finetune_validation.py \
  --snapshot data/ppi_finetune_validation_v1/provenance.json \
  --output /tmp/flashppi_validation_reproduced

# 四步短训练，含中途保存、完整保存与重载检查
python -m flashppi.finetune --smoke --output runs/head_smoke_new

# 从短训练第 2 步恢复到第 4 步
python -m flashppi.finetune --smoke \
  --resume runs/head_smoke_new/resume_probe --output runs/head_resume_new

# 评估未微调模型；加 --resume CHECKPOINT 可评估对应权重
python -m flashppi.finetune --evaluate-only --output runs/val_baseline_new

# 正式第一轮：1,000 步，每 100 步验证（尚未执行）
python -m flashppi.finetune --output runs/head_finetune_v1
```

## 验收结果

6 项 CPU 回归检查通过。`runs/finetune_head_smoke/checks.json` 确认初始预测完全一致、骨干权重不变、接触头更新、梯度有限、保存重载预测差异为 0，峰值 CUDA 分配约 8.72 GB。四步使用 batch=16、chunk=8，其余训练设置相同；这只是运行验证，不是性能提升证据。

`runs/finetune_preflight/resume_equivalence.json` 确认断点续训与连续训练最终 safetensors SHA256 完全相同。`input_integrity.json` 确认原库 21 个 PPI 分片、序列 FASTA、清单及官方 test 哈希未变；`validation_replay.json` 确认无网络重放得到相同验证集。完整长程收敛、最佳超参数和泛化提升仍需正式实验验证。

## 已授权的后续串行实验

执行器为 `tools/continue_finetune_experiments.py`，当前任务状态见 `runs/finetune_followup/status.json`，总日志见 `runs/finetune_followup.log`。它等待第一轮正常结束，评估 `last/` 的完整官方 E. coli test，然后进行以下两阶段，不改动正在运行的第一轮配置。

第二阶段配置：`configs_train/finetune_head_bs1024_v1.json`。从官方起点重启，batch=1,024、chunk=8、63 步（64,512 次正对采样，对比第一轮 64,000 次），lr=1e-5。warmup 按 3,200 次采样折算为 3.125 个大 batch 步，第 4 步达到固定学习率；每 6 步约 6,144 次采样验证一次，最后第 63 步也验证。先单独进行一个完整大 batch 的训练、保存重载验收，验收通过后才正式开始。该短验收的模型不用于正式对照起点。

第三阶段先运行 `tools/diagnose_finetune_validation.py`：比较 val 的物种和长度分组，审计簇采样后的序列对重复、截断后有效接触、已知互作过滤后的接触负样本数量。分支仅使用 val：若大 batch 最近连续三次 val 互作 AP 均高于官方起点，且接触 precision@K 均不低于起点，则从该轮 final 继续到累计 126 步；否则进入诊断分支，不盲目扩大训练量。这是操作性检查，不代表统计显著或跨种子复现。

延长训练仅通过 `--resume ... --extend-steps 126` 增大总步数预算，优化器、随机状态、学习率和其余配置保持不变。第二、三阶段不自动反复评估 test。上述诊断已完成；后续 LoRA 作为独立实验执行，详见下一节。

## LoRA 对照（2026-10-10）

前两轮 head-only 实验已完成，汇总见 `runs/finetune_complete_results/REPORT.md`。
新的 LoRA 实验从官方 `flashppi_noecoli` 重新开始：仅最后 8 个 gLM2 层
（0-based 25–32）的 fused QKV 投影中 Q、V 部分增加适配器，K 与原始骨干权重冻结。
rank=8、alpha=16、dropout=0.05，共 327,680 个 LoRA 参数；原有 CLIP 与接触预测头继续训练。
不改变接触预测结构及 `sigmoid(max(valid contact logits))` 互作打分。

按用户追加要求，固定 effective batch=1024、2 个 epoch、lr=1e-5；每轮 1,120,675 次簇均衡有放回采样、1,095 步，共 2,190 步。每轮末批保留 419 对，不丢弃样本。每 100 步及 epoch 结束验证，并每 100 步保存滚动 `latest/`；同时保存 `epoch_01/`、`epoch_02/`、best 和 final。其他数据与损失设置沿用大 batch 对照。
先使用 `--smoke --stress-smoke` 测试双方均长于 512 残基的样本，分别实测微批 16、32，
选择能留出显存余量的设置；不按 test 表现选微批。随后运行相同微批的 head-only 对照，
因为微批会改变接触 hard-negative 候选池及 focal loss 的分组归一化。

`python tests/test_lora.py` 检查零初始化、K 不变、冻结参数无梯度、合并导出、
适配器恢复，以及 dropout + activation checkpointing 下 GradCache 的梯度一致性。
GPU 验收另外检查真实优化器更新、冻结骨干哈希、重载输出和断点恢复一致性。
训练使用 BF16 与非重入梯度检查点，验证仍使用全长序列与 FP16 autocast。

检查点同时保存 `model.safetensors`（合并 LoRA 后、兼容官方推理架构）和
`trainable.safetensors`（未合并的适配器与预测头）。恢复训练时使用相同配置执行
`python -m flashppi.finetune --config CONFIG --resume CHECKPOINT --output NEW_RUN`；
程序重新加载固定官方基座，再恢复适配器、预测头、优化器与 RNG，避免重复合并。
`best_step_*/` 按 val 互作 AP 选择，`last/` 始终保存最终状态。

串行执行器为 `tools/run_lora_comparison.py --microbatch 16`（或实测通过的 32）。
它检查压力测试与续训验收、存档代码和配置、依次训练 LoRA 和匹配对照，
在两组 val 选模都固定后评估各自 best/final 的完整官方 test，最终生成 `REPORT.md`。

首次试运行曾选定微批 **32**（随后发生 OOM，现已回退到 16，见下节）：两步 512×512 残基压力测试均通过，峰值 allocated 为 22.86 GB，
确定性模式 reserved 为 31.66 GB（PyTorch 缓存预留包含可回收块，不等于活跃张量）。
确定性模式实测约 59 秒/优化步；原微批 16 压力测试约 55.5 秒，并未观察到增大微批带来速度提升。
首次试运行配置为 `configs_train/finetune_lora_bs1024_micro32_v1.json` 与
`configs_train/finetune_head_control_bs1024_micro32_v1.json`。
状态文件：`runs/lora_comparison_micro32_v1/status.json`；两组都完成后生成同目录 `REPORT.md`。

两组均启用 `torch.use_deterministic_algorithms(True)` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，用于 GPU 反向传播与恢复训练的精确复现。随机种子与确定性标志写入固定配置。
为避免漏掉前期最佳点，额外在第 6、12、24、48 步验证；之后维持每 100 步及 epoch 边界验证。

### OOM 修正（2026-10-10 21:46）

微批 32 的正式任务在完成 5 步后、第 6 步反向传播发生 OOM；尚未进行首次验证，
也没有可恢复的训练 checkpoint。两步固定长序列压力测试未覆盖长时间、动态负对数量造成的显存分配变化，
不能据此判定长程训练稳定。失败日志保留在 `runs/lora_comparison_micro32_v1/lora_train.log`。

正式方案回退到微批 **16**，effective batch=1024、两轮 2,190 步保持不变；匹配的 head-only 对照也使用 16。
从官方基座重新开始，配置使用 `finetune_lora_bs1024_micro16_v1.json` 与
`finetune_head_control_bs1024_micro16_v1.json`。
执行器启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并在每步日志中记录显存峰值。
新流程状态写入 `runs/lora_comparison_micro16_v1/status.json`，不覆盖原失败记录。

微批 16 的确定性压力测试与断点恢复已通过，但正式长训练尚未重启：用户提出前缀裁剪会丢失结合界面，当前正在修订裁剪方案。
`runs/crop_audit_v1.json` 对相同种子的 64,512 次抽样复核：8,058 次涉及截断；357 次原本无有效接触，
另 988 次因截断丢失全部有效接触。8,004 次原本有接触且涉及截断的记录中，2,694 次保留不足一半接触。
这些是抽样记录数，不能直接视作全训练库的唯一样本数。状态见 `runs/lora_oom_recovery_v1/status.json`。

### 当前裁剪方案：interface_v2

界面裁剪与无有效接触记录处理的固定策略、训练池数量变化、验收命令见
[INTERFACE_CROPPING.md](INTERFACE_CROPPING.md)。新配置使用 `*_micro16_interface_v2.json`，
两轮为 2,150 步；旧的 v1 配置仅保留用于复现历史实验。本轮只进行裁剪验收，不自动启动两轮长训练。
