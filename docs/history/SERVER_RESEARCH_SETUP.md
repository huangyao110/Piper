# 7×RTX 4090 服务器研究准备

本方案用于新实验：固定现有 train/val/官方 E. coli test，先比较单 gLM2 与
gLM2＋冻结 ESM C 300M。原始数据库只读；当前单卡训练入口、配置和权重不受影响。
本地 GPU 检查不能替代服务器上真实 4/7 卡 NCCL、显存与吞吐量验收。

## 本地已完成的验证

独立 Python 3.12 环境安装和依赖检查通过。22 项 CPU 回归检查通过，其中新增检查覆盖
三进程全局梯度、非整批/空 rank、接触标签掩码、残基对齐和融合初始化。
RTX 5090 上分别验证了冻结 gLM2 和 gLM2 LoRA 的真实融合模型：初始预测与原模型一致，
融合参数能够更新，冻结权重未变，保存重载预测一致。
最终入口的单卡连续 4 步与 2 步后断点恢复结果逐位一致（参数及优化器）。
这些是短序列功能检查，不代表 24GB 卡上长序列训练的显存或速度。
完整记录见 `runs/server_preparation_20261011/REPORT.md`。

## 模型与实验边界

每条蛋白以相同的界面裁剪输入两个 PLM；按原始残基位置对齐，去除 ESM C
BOS/EOS 和 padding。融合为 `gLM2 + alpha * Linear(LayerNorm(ESMC))`，输出仍为
1280 维；`alpha=0` 初始化。现有 CLIP 头、接触网络和最大有效接触分数公式保持原样。
ESM C 始终冻结并处于 eval 模式。第一版不包含 ESM C LoRA、SaProt、FSDP 或全参数微调。

| 配置文件（位于 configs_train/） | 用途 |
|---|---|
| server_glm_head.json | 冻结 gLM2，仅训练两个头的匹配基线 |
| server_glm_fusion_control.json | 相同可训练融合容量；输入来自 gLM2 的固定随机投影 |
| server_glm_esmc300_frozen.json | 两个 PLM 均冻结，训练融合模块与两个头 |
| server_glm_lora.json | gLM2 最后 8 层 Q/V LoRA 的匹配基线 |
| server_glm_esmc300_glmlora.json | ESM C 冻结，同时更新融合层、两个头和 gLM2 LoRA |

固定全局 batch=1024，微批=8，每链最多 512 残基，1,099,957 个有有效接触的训练记录，
2 个 epoch/2150 step，聚类采样有放回。原无接触标签记录仍参与已知正互作屏蔽。
验证使用全长，不通过官方 test 选择 checkpoint。

## 多卡实现与可比性

入口为 `flashppi.research_train`。各 rank 按轮转方式接收固定的全局微批，跨卡汇总
CLIP 向量和 cluster ID，得到全局对比池；重放各自的计算后，每个优化步对可训练参数梯度
做一次分桶 SUM 同步。温度参数的全局梯度单独缩放，避免重复累加。
这是同步数据并行，并非 FSDP 或自动将显存合为一个池。

- 不丢弃或填充重复样本；支持末尾不足一批和空 rank。
- contact 难负例池仍在固定微批内部；增加卡数不会暗中改变其大小。
- 接触损失仍为按样本数加权的微批 focal loss，未同时改变负例策略/归一化。
- 新入口使用 FP32 CLIP logits、基于全局微批的裁剪/重放随机种子；与旧单卡轨迹不逐位相同。
  新旧结论应以相同新入口、微批和精度的配对实验比较。
- 全部 PLM FP32 权重，训练 BF16 autocast，验证 FP16 autocast；本地先使用 SDPA。

## 迁移与安装

打包只选择源码、固定数据、所需基模型和历史指标，不包含原始结构库或运行中的 checkpoint。
ESM C 权重固定到 revision `7f10b20ae75017b2dbc884070e03434515709a8d`，SHA256 记录在其 manifest。

```bash
python tools/download_esmc_for_fusion.py
python tools/prepare_server_bundle.py --output artifacts/server_research_20261011.tar
```

将 tar 和对应 `.sha256` 文件传到服务器，在新目录解包：

```bash
sha256sum -c server_research_20261011.tar.sha256
mkdir FlashPPI-research
tar -xf server_research_20261011.tar -C FlashPPI-research
cd FlashPPI-research
python3 tools/prepare_server_bundle.py --verify .
SERVER_PYTHON=python3.12 bash tools/setup_server_env.sh
```

安装脚本新建 `.venv/server`，不升级系统或原实验环境。主依赖固定为 PyTorch 2.8.0 CUDA 12.8、
ESM 3.2.3、Transformers 4.48.1；ESM 旧版对 Transformers 有上限，因此必须隔离环境。
服务器需支持 CUDA 12.8 运行时的 NVIDIA 驱动，以及可以访问 Python 包下载服务。
完整实装版本写入 `.venv/server/installed-lock.txt`。
迁移包还包含 `requirements-server-lock.txt`；安装脚本优先使用该完整依赖锁定文件。

## 租机后的验收顺序

先确认所有卡在同一台机器且可用，记录 CPU、内存、磁盘、GPU 拓扑和实际 collective 性能：

```bash
.venv/server/bin/python tools/server_preflight.py --expect-gpus 7 --output runs/server_acceptance/inventory.json
.venv/server/bin/torchrun --standalone --nproc_per_node=7 tools/server_preflight.py --expect-gpus 7 --output runs/server_acceptance/nccl_7gpu.json
.venv/server/bin/python tests/test_research_multigpu.py
.venv/server/bin/python tools/check_research_model.py --output runs/server_acceptance/model_smoke --memory-fraction 0.9
```

先做小规模多卡 smoke，再做 full-length 压力测试。每次 output 使用新目录：

```bash
bash tools/run_server_training.sh configs_train/server_glm_esmc300_frozen.json runs/server_acceptance/smoke4 0,1,2,3 --smoke-steps 4 --smoke-global-batch 32 --smoke-length 64
bash tools/run_server_training.sh configs_train/server_glm_esmc300_frozen.json runs/server_acceptance/stress4 0,1,2,3 --smoke-steps 100 --smoke-global-batch 1024 --smoke-length 512 --stress-smoke
```

`--stress-smoke` 仅在 smoke 中使用，强制采样两链原长均至少 512 的正对，并使用界面裁剪。
观察连续 100 步（不是只看前两步）的 loss、显存、速度和错误；history 每步记录。
如需减小微批，通过 `--microbatch 4`，并让相关基线使用相同微批。
启用 gLM2 LoRA 或改用 7 卡时，对相应配置和 GPU 列表重新执行上述 smoke、压力和恢复检查。
正式训练前另运行同 world size 的连续/中断/恢复比较：

```bash
bash tools/run_server_training.sh configs_train/server_glm_esmc300_frozen.json runs/server_acceptance/replay_full 0,1,2,3 --smoke-steps 4 --smoke-global-batch 32 --smoke-length 64
bash tools/run_server_training.sh configs_train/server_glm_esmc300_frozen.json runs/server_acceptance/replay_stop 0,1,2,3 --smoke-steps 4 --smoke-global-batch 32 --smoke-length 64 --stop-after 2
bash tools/run_server_training.sh configs_train/server_glm_esmc300_frozen.json runs/server_acceptance/replay_resume 0,1,2,3 --smoke-steps 4 --smoke-global-batch 32 --smoke-length 64 --resume runs/server_acceptance/replay_stop/checkpoints/step_000002
```

比较两个最终 checkpoint 的张量和优化器状态：

```bash
.venv/server/bin/python tools/check_research_replay.py --reference runs/server_acceptance/replay_full/checkpoints/step_000004 --resumed runs/server_acceptance/replay_resume/checkpoints/step_000004 --output runs/server_acceptance/replay_checks.json
```

跨硬件/软件的结果不承诺逐位一致。
对 1/2/4/7 卡记录相同配置下的 pairs/s，再按整机 ¥13.09/小时计算费用，决定正式任务使用几张卡。

## 正式训练与 checkpoint

后续随机序列 mask 增强已接入此入口，起始方案为 15% 直接遮蔽；配置字段、可复现规则及当前接入范围见 [SEQUENCE_MASKING.md](SEQUENCE_MASKING.md)。现有基线配置默认关闭，新增实验可传 `--sequence-mask-probability 0.15`，val/test 不做 mask。

验收通过后启动选定实验，例如：

```bash
bash tools/run_server_training.sh configs_train/server_glm_esmc300_frozen.json runs/server_esmc_frozen_v1 0,1,2,3
```

`batch_size` 始终是全局 batch，不再乘 GPU 数。CPU 线程默认每进程 4；训练表以 Arrow 格式
加载，避免一次转换整表为 Python 对象。验证由 rank 0 执行，其他 rank 等待；残基缓存放在 CPU。
当前版本没有自动后台验证调度。

每次验证、epoch 结束和恢复保存点保留独立 `checkpoints/step_XXXXXX/`，不滚动覆盖旧 step。
checkpoint 保存所有可训练参数、AdamW 状态、各 rank RNG、配置及基模型/数据哈希。
固定基模型在 `artifacts/` 只保存一份；checkpoint 不能脱离这些基模型直接用 AutoModel 加载。

- `best_clip.json`：按 val CLIP AP 选择。
- `best_contact.json`：按 val 接触 P@K 选择。
- `best_interaction.json`：按 val 互作 AP 选择，保留观察。
- `latest.json`、`epoch_XX.json`、`final.json`：指向完整 checkpoint，不复制大权重。

新入口仅接受自身格式的 resume，且必须保持配置、world size、数据和基模型一致。
不能直接恢复原 `flashppi.finetune` 的旧 checkpoint。每个实验记录源码快照、输入哈希和环境版本。

## 官方 test

训练不会自动频繁访问 test；固定选择后显式执行：

```bash
.venv/server/bin/python tools/evaluate_research_checkpoint.py --checkpoint runs/server_esmc_frozen_v1/checkpoints/step_002150 --output runs/server_esmc_frozen_v1/official_ecoli_final
```

完整评分全部 650 正对和 65,000 负对，使用全长与原配对方向；报告 CLIP AP、互作 AP/AUROC、
12 Å/8 Å 接触 P@K。它不是检索 Recall@K 实验；完整蛋白组检索评估属于后续工作。

## 依据

- [ESM C 官方实现](https://github.com/evolutionaryscale/esm/tree/v3.2.1/esm)
- [PyTorch CUDA 12.8 安装版本](https://pytorch.org/get-started/previous-versions/)
- [NCCL 通信检查](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2276/user-guide/docs/troubleshooting.html)
