# DDI 预训练准备

本实验只用距离发行包中 `kind=ddi` 的样本训练。加载固定版本的官方
ESMC-300M 权重，重新初始化所有任务头，不继承 FlashPPI 的 GLM2、LoRA 或任务权重。
数据准备与训练为独立命令；准备脚本本身不会启动训练。
2026-10-12 用户已授权本轮准备完成并通过检查后自动启动两个 epoch 的正式训练。

## 固定协议

- 数据参数：`configs_data/ddi_pretrain_v1.json`。
- 模型及优化参数：`configs_train/ddi_esmc300_v1.json`。
- val 保留原来的 500 对结构正例和 4,200 对互作评估样本（81 正、4,119 负）。
- test 为原始 RF2-PPI 33,000 对（3,000 正、30,000 负）。样本、标签与序列不改动。
- 冻结集合之间存在精确蛋白序列重叠，不能称为严格 protein-disjoint；准备清单记录实际重叠数量。
- best 按 val 的新互作头 AP 选择，同时报告 CLIP AP、接触 P@K 和距离交叉熵。
  RF2 test 由独立命令显式运行，不参与选模型。

## 数据准备与同源排除

通过 `seq1_id/seq2_id` 关联已核验的 `ddi_sequence_clusters.tsv`；原数据中的 `-1`
字段不直接使用。簇对以 `max_cluster_id + 1` 为基数进行 int64 编码，并检查溢出。

MMseqs2 比较全部 DDI 域序列与全部 val/test 序列；门槛为一致性 **≥30%**、
DDI 域覆盖 **≥80%**，不要求覆盖完整测试蛋白。实现以 holdout 为 query、DDI 为
target，故 `--cov-mode 1`。参数包含 `-s 7.5 -e 0.001 --max-seqs 200000`。
搜索属于启发式同源检测，不能据此证明不存在所有远缘同源。
新搜索导出 `fident`，不使用依赖回溯的 `nident`。2026-10-12 已完成的旧搜索
保留原始导出与校验值；仅在输入、MMseqs 版本、搜索阈值等全部一致时复用，
身份阈值由 MMseqs 内部过滤保证，所有通过搜索的目标序列均进入同源簇排除。
独立审计重新读取全部命中目标及原始序列簇映射，验证排除簇集合完全一致。
`tests/test_ddi_homology.py` 通过真实 MMseqs 搜索验证身份阈值、域覆盖方向及旧格式兼容。

保守排除命中域所在 70% 序列簇；任一链被排除则整对排除。无有效接触的记录不采样。
最终保存源 Parquet 的行组索引、正例簇对集合、来源统计和校验值，不复制结构坐标。
原始数据库只读。MMseqs 临时数据库在成功后清理，命令、命中表与日志作为核验记录保留。

采样先均匀抽无序序列簇对，再抽该簇对下的结构。一个 epoch 定义为 N 次有放回抽样，
N 是过滤后的结构数；不代表每条结构恰好遍历一次。结构聚类不参与这个首轮方案。

## 模型与损失

ESMC 输出 `[B,L,960]` 残基特征；去除 BOS/EOS，保留真实残基 MASK 的 attention。
两条链加 segment embedding，沿序列轴拼接，经过两层 960 维 Transformer；
8 组 Q/K 得到 `[B,L1,L2,8]` 的配对特征，分为三个分支：

1. Linear → 单个接触 logit；仅真实结构正例计算按样本归一化的有效位置 BCE。
2. Linear → 32 个距离区间；边界为 2、3、…、32 Å，使用有效位置交叉熵。
   这是本实验的分箱方案，不是照搬 AF3 的 64 区间配置。
3. 对有效配对特征做 mean 与 log-mean-exp 池化、拼接，再 Linear → 互作 logit。
   推理不使用最大接触概率。另保留 1,024 维 CLIP 检索分支。

初始损失权重为 CLIP 0.1、contact 1、distogram 0.1、interaction 1；这些是待验证的
初始设置，不表示已经证明最优。Cβ（GLY 为 Cα）距离在 float64 中计算并分箱，
接触严格采用 `<12 Å`。未知坐标和 padding 不参与结构损失。

训练进行接触锚定的连续界面裁剪，每条链最多 512 残基；序列、坐标和标签同步切片、
随机交换链。随后对标准氨基酸独立以 15% 概率替换为原生 MASK，保留至少一个可见残基。
没有 MLM 目标；验证和测试不裁剪、不 mask。

## 共用负例

对比学习与互作分类共用候选排除规则：排除已知 DDI 正例簇对、同簇自配对和同母蛋白
的域。CongLab 与 TeddyMer 的母蛋白统一用 UniProt accession 标识。
合并来源 `conglab+teddymer` 保留来源标签，并按实际 `original_id` 格式解析同一母蛋白。
全部已知 DDI 正例在过滤前建立排除集合；不使用 val/test 的标签训练。

CLIP 使用有效 batch 内全部合格错配；互作头在各微批内从同一候选规则抽样，
目标正负约 1:1，候选不足时减少数量。随机错配是弱负例，首轮不使用困难负例。
弱负例只计算 CLIP 与互作分类损失，不伪造全零接触或最远距离标签。

## 验证和执行

在仓库根目录运行；当前实测环境为 `.venv/server/bin/python`（ESM 3.2.3）。
一条命令串行完成数据准备、真模型预检和独立核验：

```bash
.venv/server/bin/python tools/finish_ddi_preparation.py
```

进度见 `data/ddi_preflight_v1/pipeline_status.json`；只有状态为 `complete` 且
`readiness.json` 为 `passed` 才表示整条准备链完成。该脚本不调用训练入口。
若仅独立审计失败，修正后可用 `--start-at independent_data_audit` 继续；
审计仍会重新校验完整数据清单、全部命中目标、训练索引和匹配的模型预检报告。
下面是各步骤的独立命令：

```bash
# CPU 回归检查
CUDA_VISIBLE_DEVICES='' .venv/server/bin/python tests/test_ddi_pretraining.py

# 全量准备；成功的同源搜索可复用，输入变化则拒绝复用
.venv/server/bin/python tools/prepare_ddi_pretrain.py --config configs_data/ddi_pretrain_v1.json

# 真模型前向、反向、512×512 显存压力和保存重载；不执行 optimizer.step
.venv/server/bin/python tools/preflight_ddi.py --output data/ddi_preflight_v1/report.json

# 准备完成后独立扫描全部训练索引，并验证一个真实 1,024 正例 batch
.venv/server/bin/python tools/audit_ddi_preparation.py
```

等待已运行的准备流程完成，并在核验通过后启动正式训练：

```bash
.venv/server/bin/python tools/start_ddi_when_ready.py \
  --output runs/ddi_esmc300_distogram_mask15_pairhead_v1_20261012
```

该启动器通过进程锁避免重复启动，固定当前代码和配置校验值；准备失败则不启动。
状态与实际优化器步数见 `data/ddi_preflight_v1/training_launch_status.json`，
训练标准输出见同目录下的 `ddi_training.log`。启动器不自动重启失败训练。

以下是直接训练与独立测试入口：

```bash
.venv/server/bin/python -m flashppi.ddi_train \
  --config configs_train/ddi_esmc300_v1.json \
  --preflight data/ddi_preflight_v1/report.json --output runs/ddi_esmc300_v1

.venv/server/bin/python tools/evaluate_ddi_checkpoint.py \
  --checkpoint runs/ddi_esmc300_v1/best --split rf2 --output runs/ddi_esmc300_v1/test_best
```

当前入口支持单 GPU，global batch=1,024、microbatch=40，通过两遍梯度缓存实现。
完整有效 batch 分为 25 个 40 对微批和最后一个 24 对微批；损失按实际样本数加权。
2026-10-12 在 RTX 5090 32 GB 上实测，每条链物理长度 512、正负比 1:1，
microbatch=32/40 的峰值 allocated 显存分别为 15.48/17.80 GiB，均通过前向、反向和保存重载检查。
压力检查包含常驻 AdamW 一、二阶状态，不执行优化器更新；报告保存在
`data/ddi_preflight_v1/report_micro32.json` 和 `report_micro40.json`。
这些结果不包含 `optimizer.step` 的瞬时分配，也不代表未经裁剪的长序列验证显存。
ESMC 全参数更新（LR=1e-5），新任务头 LR=1e-4；BF16、梯度检查点、初始预算两个 epoch。
多 GPU DDP 尚未实现，不能直接用 `torchrun` 扩展此入口。

保存 `best`、恢复用 `last` 和完成时的 `final`，包含优化器与 RNG 状态。
恢复需 `--resume`，并要求代码、配置、数据和 PyTorch 版本一致。
训练入口检查完整的数据准备清单与匹配的预检报告，任何一项未完成则停止。

这批 DDI 包含预测结构域接触；坐标有效不等于高结构置信度。本轮没有额外
pLDDT/PAE 过滤，亦不能将 DDI 正例解释为已实验证实的独立蛋白互作。
同源排查也不包含 ESMC 的预训练语料或远缘结构相似性审计。
