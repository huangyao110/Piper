# ESMC–ProtT5–gLM2 gated 模型方案

日期：2026-10-14。用户已选择并授权准备代码，当前训练成功结束后自动衔接新阶段。

## 1. 已确定的结构

- gLM2：继承当前训练最终检查点的骨干 LoRA，继续训练最后 16 层 Q/V LoRA；基础权重冻结。
- ESMC-300M：冻结全部编码器参数，始终使用 eval 模式。
- ProtT5：采用 `Rostlab/prot_t5_xl_half_uniref50-enc`，冻结全部编码器参数，始终使用 eval 模式。
- 三路表示逐残基对齐，先融合，再分别进入 CLIP 与 concat。
- CLIP：resi-gated weighted mean pooling，每个残基一个标量 gate；Q/K 头各自学习 gate。
- concat：window=3、stride=2 的 masked mean pooling，保留所有 patch；每个 patch 使用逐通道残差 gate，再进入现有两层 concat Transformer。
- 保留接触存在性、两侧 coverage 辅助头及 `sigmoid(max(valid contact logits))` 的互作打分。

这是一项新结构实验，不声称各项改动已经证明有效。后续消融分别考察 CLIP gate、patch 尺寸、patch gate、ESMC 与 ProtT5；本次自动衔接启动用户指定的联合结构。

## 2. 输入、对齐与冻结

首版采用相同裁剪在线编码：三路模型读取同一条裁剪后的序列，保留一致的残基坐标。全长 embedding 后切片不是此版本的替代实现，因为编码上下文不同。

沿用 15% 序列 masking：先确定一次遮蔽位置，三路编码器使用同一位置集合。gLM2/ESMC 使用自身 mask token；ProtT5 用单个 `<extra_id_0>` 占据每个被遮蔽的残基位置，不进行 span 合并，保留一残基一位置。该 ProtT5 输入约定必须通过真实 tokenizer 检查，不声称与其预训练遮蔽协议完全相同。罕见残基 U/Z/O/B 在 ProtT5 分支映射为 X，长度不变。

移除 ESMC 的 BOS/EOS、ProtT5 的 EOS，严格检查输出长度与 gLM2 残基 mask 一致；padding 不进入池化分母、不产生有效 patch。结构坐标未知与序列 padding 使用不同 mask。

冻结编码器在 `no_grad` 中运行；融合投影和 gate 在此上下文之外训练。冻结降低反向开销，但仍需要模型存储和前向计算。使用独立环境，避免改变当前训练环境的依赖。

## 3. 残基级融合

设 gLM2 为 1280 维，ESMC 为 960 维，ProtT5 为 1024 维：

`H = H_glm2 + alpha_esmc * P_esmc(LN(H_esmc)) + alpha_prott5 * P_prott5(LN(H_prott5))`

两个投影均输出 1280 维。alpha 初始化为 0，投影使用非零初始化；保留 gLM2 路径，使新增残差可以逐步学入。不能同时把 alpha 与整个投影置零而阻断学习。

拼接后投影也可实现；本版选择残差融合以便继承旧模型。三个模型的未经归一化表示不直接相加。

## 4. CLIP 残基 gate

`score_i = Linear(SiLU(Linear(LN(H_i))))`

`g_i = sigmoid(score_i)`

`pool = sum(mask_i * g_i * H_i) / sum(mask_i * g_i)`

打分输出层初始化为零，从等权池化开始。使用 FP32 权重归一化和求和，显式处理空 mask，屏蔽 padding 中的非有限值。保留旧 Q/K MLP 权重、输出维度和 L2 归一化。gate 仅依赖本蛋白，支持单蛋白预编码。

## 5. concat patch gate

窗口从固定 stride 网格开始，末端允许短窗口：例如 1–3、3–5、5–7。先按真实残基取均值：

`p_j = masked_mean(H[window_j])`

`p'_j = p_j + beta * sigmoid(G(LN(p_j))) * F(LN(p_j))`

G 输出逐通道权重，F 为瓶颈特征变换。beta 为可学习标量，初始化为 0；F 非零初始化。gate 后再次清零无效 patch，所有有效 patch 均送入 concat，不做全局求和或硬删除。

CLIP gate 和 patch gate 不共享参数。patch gate 是特征适配器，不是接触概率，也不能保证恢复 mean pooling 已丢失的信息。

## 6. 监督和评估口径

接触标签继续采用窗口内任意已观测残基接触的 OR；负 patch 只有全部真实残基对坐标可观测时才监督。coverage 继续表示两侧参与接触的残基比例，仅监督完整观测的正 patch。

3 残基窗口的正 coverage 最小为 1/3，旧的 `sqrt(fA*fB) <= 0.25` 分组不再适用。新版本显式使用 `sqrt(fA*fB) <= 1/3 + 1e-7` 作为最低覆盖组，并在报告写明阈值和窗口；旧实验的阈值仍为 0.25。该组与旧 small-patch 指标不能直接横比。

保留 CLIP AP、互作 AP/AUROC、patch AP/P@K、coverage MAE，并记录 gate/融合系数、冻结参数检查、吞吐量与显存。不同 patch 网格的 AP 单独报告；检验门控收益需要 3/2 网格的匹配对照。test 仅在新阶段最终检查点评估，不用于选择模型或超参数。

512 残基下：16/4 有 125 个 patch，3/2 有 256 个；二次方矩阵元素数为原来的 4.194304 倍。该比例不是总训练耗时或显存增长预测，必须实测。

## 7. 新阶段训练与继承

衔接来源固定为当前 `conglab_ecoli_motif_unique_negatives_v3_20261014` 成功完成后的 final 检查点，不按 E. coli test 成绩选源。

- 继承所有已有可训练参数：gLM2 LoRA、CLIP Q/K MLP、温度、concat core、coverage head；逐参数检查名字和 shape，不能静默跳过。
- 新增投影和 gate 独立初始化；不加载旧 Adam 状态，不继承旧 best 指针。
- 新阶段 step 从 0 开始，沿用全局 batch=1024、microbatch=16、max_len=512、PPI/DDI 1:1、unique heteroprotein negatives。
- 新阶段预算为 2 个采样 epoch / 1616 steps；warmup=81，随后 cosine 衰减；LoRA LR=1e-5、heads LR=1e-4。
- loss 权重沿用 CLIP=0.1、contact=1.0、coverage=0.1；每 50 步验证、每 25 步存恢复检查点。
- 这是带旧权重初始化的新训练阶段，不是旧训练的逐位 resume。新模型自己的中断恢复必须保持配置、数据、源代码、优化器和 RNG 一致。

若 full-length、完整 batch 预检失败，监督器记录失败并停止，不擅自改 microbatch、负采样池或训练预算。

## 8. 自动衔接

使用独立准备目录、源码快照和环境。后台等待器只在旧 supervisor 标记 `complete`、旧 trainer 到达 1616 步、旧最终评估产物完整且旧训练进程退出后继续。失败、人工停止或进程意外消失不视为成功，不自动重启旧任务。

按顺序执行：验证数据/模型/source hash → 新模型 GPU 预检 → 从旧 final 建立新阶段初始化 → 新阶段正式训练 → 新阶段 final E. coli 评估。

GPU 预检包括真实 tokenizer 对齐、冻结权重不变、旧参数完整继承、恒等初始化、padding/批次组成一致性、有限 loss/梯度、各新增分支可学习、保存重载和恢复重放、一整个真实全局 batch。预检更新为一次性检查，不进入正式训练。

监督器使用单实例锁、明确状态和子进程 PID，重复启动不能产生两份训练。源码和配置在排队时固定。所有失败保留日志并停止。旧训练及其原始源码、配置、checkpoint 不作原位修改。

## 9. 交付记录

实现计划：`docs/superpowers/plans/2026-10-14-esmc-prott5-glm2-gated.md`。

源码位于独立工作目录 `/home/zlab/.codex/worktrees/5f44/FlashPPI`，分支为 `codex/esmc-prott5-glm2-gated-20261014`。当前训练使用的原目录不合并这些改动。

- 新配置：`configs_train/esmc_prott5_glm2_gated_v1.json`。
- 模型实现：`flashppi/gated_modules.py`、`flashppi/frozen_prott5.py`、`flashppi/gated_warm_start.py`，已接入现有 research trainer 与评估器。
- 独立 Python：`/media/zlab/lexar4t/FlashPPI/artifacts/environments/esmc_prott5_gated_py312/bin/python`；torch 2.8.0+cu128、transformers 4.48.1、esm 3.2.3、sentencepiece 0.2.1。完整版本记录在新实验的 `preparation/requirements.lock.txt`。
- ESMC 权重：`artifacts/esmc_300m`；ProtT5 权重：`artifacts/prot_t5_xl_half_uniref50_enc`。各自 manifest 固定版本与校验值。
- 新实验输出：`/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_gated_v1_20261014`。

模型阶段验证：56 项相关 CPU 回归通过；真实 tokenizer 对齐通过；29 个模型及数据相关文件校验通过；从旧 step 1275 继承的短序列 GPU 前向通过，峰值分配显存约 7.71 GB、保留显存约 8.04 GB。该前向检查不是完整预检，也不能用于推断完整 batch 显存需求。原目录 135 个源码及配置文件核对未改变。详细记录在新实验 `preparation/` 下。

接续工具为 `tools/run_gated_after_training.py`，真实模型预检为 `tools/check_gated_ensemble.py`。相关组合 CPU 测试 83 项通过；实际旧任务 `--check-only` 返回正常等待。依赖导入存在第三方 sentencepiece/SWIG deprecation 提示，未屏蔽其他 warning。

真实 GPU quick 预检已通过（`preparation/quick_preflight_v1/checks.json`）：31 个新增可训练张量和 LoRA 均取得梯度；冻结权重摘要前后一致；三路输入对齐；保存重载预测、恢复后的参数、优化器和 RNG 均一致。耗时约 95 秒，峰值分配显存约 8.64 GB。初始化状态下 padding/批次组成变化导致的最大 contact logit 差约 0.00195，采用报告中明确的 FP16 容差。该预检使用旧 step 1275 的一次性副本，`quick=true`、`full_batch_proof=false`，不允许直接启动正式训练。

正式接续固定使用旧 step 1616；旧最终测试完成且监督进程与子进程退出后，自动执行完整 batch=1024、microbatch=16、max_len=512 预检。新阶段输出在 `training/`（旧阶段为 `motif/`），最终测试输出在 `test_ecoli_final/`。

最终审查通过，无 Critical/Important 问题。已知验证范围限制：端到端 padding/批次组成比较在新增残差系数为零、resi gate 等权的初始化状态执行；CPU 测试覆盖门控自身的 padding 行为，但尚未做更新后整个三路模型的端到端 padding 比较。

### 后台队列已启动

2026-10-14 20:50（Asia/Shanghai）以独立 session 启动，PID **1073197**。已回读确认状态 `waiting_predecessor`、`child_pid=null`、单实例锁持有，尚未生成正式 preflight/training/test 输出。停止信号修复后的 40 项接续回归通过；最终审查的实现提交为 `19ff3c0`。

状态文件：`experiments/esmc_prott5_glm2_gated_v1_20261014/status.json`。队列固定配置、源码、输入、Python 与依赖版本的记录为同目录 `queue_manifest.json`；启动命令/PID 及运行核对记录为 `preparation/queue_launch.json`、`preparation/queue_verification.json`。启动器日志为 `preparation/queue_launch.log`；后续各阶段分别写入 `preflight.log`、`training.log`、`evaluating_ecoli.log`。

实际启动参数如下（当前已由后台进程执行，重复启动会被锁或已有产物拒绝）：

```bash
cd /home/zlab/.codex/worktrees/5f44/FlashPPI
/media/zlab/lexar4t/FlashPPI/artifacts/environments/esmc_prott5_gated_py312/bin/python -u tools/run_gated_after_training.py \
  --predecessor /media/zlab/lexar4t/FlashPPI/experiments/conglab_ecoli_motif_unique_negatives_v3_20261014 \
  --config /home/zlab/.codex/worktrees/5f44/FlashPPI/configs_train/esmc_prott5_glm2_gated_v1.json \
  --output /media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_gated_v1_20261014 \
  --poll-seconds 30
```

查看状态：

```bash
cat /media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_gated_v1_20261014/status.json
```

如需人工取消，先用 `ps -p 1073197 -o pid,args` 确认该 PID 仍是上述等待器，再执行 `kill -TERM 1073197`。等待阶段只取消新任务；若已进入新阶段，则向当前新阶段子进程转发停止信号，并禁止后续阶段。该操作不会停止旧训练。失败后保留所有产物，不自动重试或调整 microbatch。

参考：[ProtT5 encoder model card](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)、仓库现有 `flashppi/research_model.py`、`flashppi/patch_contact.py`。
