# RF2-PPI v3 从头训练：固定 4 个 epoch

用户指定 4 个 epoch、每 200 step 跑一次 test，并确认每批 CongLab DDI / CongLab PPI / 蒸馏 PPI 的比例为 50% / 25% / 25%。本次不使用 val。

配置：`configs_train/conglab_distilled_rf2ppi_v3_4epochs.json`。训练入口仍为 `flashppi.research_train`；未新增数据构建器，未改动 `tools/train_test.py`、过滤锁、索引成员、接触标签或有效位点 mask。

## 起跑参数

| 参数 | 固定设置 |
|---|---|
| 初始化 | 原始 `tattabio/gLM2_650M_UR50`；融合层、门控、CLIP 和 contact head 重新初始化；新的 AdamW 状态 |
| 可训练编码器 | gLM2 后 16 层全量微调，无 LoRA；其余 gLM2 层冻结 |
| 辅助编码器 | ESMC-300M、ProtT5 XL encoder 冻结 |
| gLM2 学习率 | 峰值 `1e-5`，末步 `1e-6` |
| 融合、门控、预测头学习率 | 峰值 `1e-4`，末步 `1e-5` |
| 调度 | 前 200 step 线性 warmup；其后 cosine，到峰值的 10% |
| AdamW | betas `(0.9, 0.999)`、eps `1e-8`、weight decay `0.1`，沿用原参数分组规则 |
| 全局 batch / microbatch | 1024 / 16；单卡梯度缓存计算完整 batch 的 CLIP loss |
| 梯度裁剪 | 全局范数 1.0 |
| contact / CLIP loss 权重 | 1.0 / 0.1；沿用原 focal/contact 目标 |
| 最大训练长度 | 每条链 512 残基；interface crop，16 候选、near-best fraction 0.9 |
| 残基 masking | 15%；三个编码器处理同一裁剪、masking 输入 |
| CLIP 门控 | residue-gated weighted mean，hidden_dim 64 |
| concat contact head | 残基级；residual channel gate，hidden_dim 128；不做 patch 输入池化 |
| 训练负例 | 沿用 `unique_heteroprotein_v1`、原预算与 CLIP 排序规则；已知正例屏蔽只来自 train |
| 数值与提速 | BF16 训练，FP32 CLIP logits；冻结 embedding CPU 复用、冻结 gLM2 前缀 CPU 复用、长度分桶 32 |
| seed | 20261024 |

这是一套预先固定的起跑设置，不是已验证的最优超参数。相比旧分支的 backbone `3e-5` 常数学习率，本次降低 backbone 峰值，并延长 warmup、加入后期衰减；新初始化预测头仍保留 `1e-4` 峰值。原损失、门控、裁剪、masking 和负例策略保持不变，避免同时重新设计这些部分。

## 数据、采样和 epoch 定义

索引：`data/conglab_distilled_ecoli_rf2ppi_v3`。

- train 1,078,413 条：CongLab DDI 572,272、CongLab PPI 298,984、蒸馏 PPI 207,157。
- 每个 batch 固定抽 512 条 CongLab DDI、256 条 CongLab PPI、256 条蒸馏 PPI。
- 每组先均匀选择一个已有序列簇，再按索引的 `within_pair_weight` 抽取含该簇的记录。两个端点属于同一簇时只计一次。
- 8 个蒸馏来源合为一个池，在池内统一均衡序列簇；不设每个蒸馏来源的等额配额。
- PPI 与 DDI 都使用 v3 已存入索引的 70/70 序列簇。DDI 旧结构簇字段不参与本次抽样。
- 组内序列簇数：CongLab DDI 1,046,319、CongLab PPI 16,359、蒸馏 PPI 70,001。PPI 两组有共有簇，合并唯一簇数 81,998，不能把组内数量直接相加。
- 固定来源配额后，保证的是各来源组内部的簇均衡，不声称全体来源合并后的每簇概率相等。

有放回采样下，epoch 是抽样预算单位，不是遍历承诺：`ceil(1,078,413 / 1024) = 1054 step/epoch`。每个 epoch 抽 1,079,296 条，4 个 epoch 共 **4216 次优化器更新、4,317,184 次正例抽样**。epoch 结束点为 1054、2108、3162、4216；不重置优化器、学习率或采样 RNG。

过滤沿用已验收的 RF2-PPI v3 协议：identity >30%、E-value <1e-5、任一侧 coverage >50%，双端分别命中固定 E. coli 参考才排除该训练对。允许单端同源，不称 C3。协议锁 SHA256：`c7b87860e08c17a911eae7a01f13b973c34dcd60ff36ffd5998ef9c731f7f7b0`。

## Test 与恢复

- Test 保持原 65,650 对 E. coli：650 正例、65,000 负例；全长输入、原始方向、无检索预筛选。
- 测试点为 200、400、…、4200，最后 4216 再测一次，共 **22 次**。不额外在 1054/2108/3162 跑 test，不运行 val。
- 报告 Contact AP/AUROC、CLIP AP/AUROC、阈值混淆矩阵、12Å/8Å 残基 Contact P@K，以及原 patch 诊断指标。Patch 仅为评估网格，不改变残基级监督。
- 采用原来的 `sigmoid(max(valid contact logits))` 接触头互作评分；测试与独立 `tools/evaluate_research_checkpoint.py` 共用同一实现。
- 每次 test 前先保存完整恢复点，清空已完成更新的梯度；暂存 Adam 张量到 CPU，测试结束后原样还原。保留 Python/NumPy/Torch CPU/CUDA RNG，下一步接续原训练序列。
- 这些 test 结果用于固定节点监测，不自动据此挑选模型、提前停止或更改学习率。预定终点模型为 step 4216。历史 E. coli benchmark 已反复查看，不描述成新的盲测。
- 每 100 step 保存恢复点；保留最近两个完整恢复点及四个 epoch 结束点。所有正式 test 预测和指标保留。测试点权重可能随恢复点轮换删除，epoch 及最终模型保留。
- 若 test 中断，从其测试前 checkpoint 恢复时先重做该次 test，再继续更新；若最后一次 test 中断，可从 4216 只补完测试，不重复训练。

## 运行与核验

运行记录位于 `experiments/rf2ppi_v3_fresh_4epochs_20261024/`，大产物写入 `/home/zlab/flashppi_runs/rf2ppi_v3_fresh_4epochs_20261024/`。源码保存独立运行快照，数据与预训练模型仍引用原文件。

先用完整配置执行两个真实训练 step 并保存，再验证恢复；这两步接入正式训练，不重复计算。另用包含最长测试链和正负标签的小批次检验完整长度的测试路径及 Adam/RNG 恢复，仅作工程检查，不将其当完整 test 指标或用于调参。检查通过后从 step 2 接续到 4216。

当前静态/回归检查结果、实际首步耗时和启动记录以实验目录的 JSON 和训练日志为准；后台统一入口是 `experiments/active_training.json`。

## 已完成的启动核验

- 121 项回归检查通过，包含历史 v1 采样兼容性、v3 来源配额、DDI 序列簇、200-step 测试日程、学习率恢复、门控/冻结缓存，以及控制器的有限重试。
- 输入和预训练模型 SHA256 核验通过；实际初始化记录确认未加载旧 FlashPPI 任务权重。
- 两个真实训练步均为完整 batch 1024；实际来源配额为 512/256/256，所有损失/梯度有限，后 16 层梯度均非零。
- 第 1 / 2 步计算耗时为 86.39 / 84.25 秒；第二步数据读取 3.94 秒；第二步峰值已分配显存 27.81 GB、缓存保留 30.01 GB（十进制）。首步之后 Adam 状态已分配，显存以第二步更有代表性。
- 工程评估子集共 16 对，包含 1430 残基最长链和最大接触矩阵 1886852 个位置；测试期间 Adam 暂存到 CPU。测试前后全部参数、Adam 张量和 Python/NumPy/Torch CPU/CUDA RNG 完全一致。
- 两步 checkpoint 已保留，控制器从 step 2 原样接续，不重启 warmup。只对历史出现过的 native SIGSEGV 允许有限恢复，总计最多 3 次正式尝试；其余失败记录并停止，不自动改超参数。

正式模型状态仍以 `experiments/active_training.json` 所指的实时文件为准；此文中的两步损失/耗时属于启动核验，不表示训练效果已经提高。
