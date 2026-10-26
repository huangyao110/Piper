# Piper

Piper 是基于 FlashPPI 的蛋白质互作与残基接触预测研究项目，融合 **gLM2、ESM C 300M 和 ProtT5 XL**，增加 CLIP 残基门控池化与接触头的残差通道门控。当前核心指标为 **12 Å / 8 Å 的残基 Contact P@K**。

完整安装、训练、恢复、评估及预测命令见 [主 README](README.md)。本次发布为科研代码整理，训练尚在进行，没有发布最终模型权重，也不声称已经证明优于 FlashPPI。

## 当前方案

- gLM2 从原始预训练权重初始化，后 16 层全量微调；ESM C 和 ProtT5 冻结。
- 三路表示按残基对齐后融合。CLIP 使用残基门控加权平均；contact 使用逐残基残差通道门控，再进入 concat head。
- **contact 训练保持残基级，不做 patch 池化，不扩大训练接触标签。** window=3、stride=2 只用于附加 patch 评估。
- 训练数据为 CongLab DDI、CongLab PPI 和八个蒸馏 PPI 来源，采样比例 50% / 25% / 25%；各池按提供的序列簇均衡抽样，簇内使用去重权重。
- 固定 E. coli test；无 val。4 个采样 epoch，共 4,216 step；每 200 step 及末步测试，每 100 step 保存恢复点。

## 数据隔离必须如何表述

当前协议为 `conglab_distilled_ecoli_rf2ppi_v3`。两端分别与固定 test 的所有正负例端点比对。单端命中要求同时满足：identity **>30%**、报告的 E-value **<1e-5**、query coverage **>50% 或** target coverage **>50%**。仅当训练对的**两个端点均命中**时排除，不要求命中同一个 test 对。

因此允许单端命中，**不能称为 C3 隔离**。参数、二进制、输入文件和判定边界已锁定；规则依据 RF2-PPI 补充材料，MMseqs 的完整命令是我们明确记录的本地实现选择，不能声称是作者公开的原始命令。详见 [数据协议](docs/DATA_PROTOCOL.md)。

`tools/train_test.py --mode build` 构建并验收；`--mode check` 独立复查。检查失败返回非零，不会自动修复成员或更新锁。v1/v2 历史协议不改写。

## 指标含义

每个有结构标签的正例，在指定距离阈值下有 K 个真实接触。把有效残基对按预测分数排序，取前 K 个，其中真实接触的比例就是该结构的 P@K，再对结构取宏平均。**这里 K 是真实接触数，不是固定值，也不是链长。**

Interaction AP 衡量整对蛋白能否判为互作；Contact P@K 衡量接触定位；二者不能互相替代。test 被反复观察，应如实报告，不应再作为从未使用过的最终盲测集。

## 整理后的入口

| 路径 | 用途 |
|---|---|
| `piper/` | `python -m piper train / evaluate / split / predict` |
| `flashppi/` | 保持 checkpoint 兼容的核心实现；没有为改名改动训练源码 |
| `configs_train/piper_rf2ppi_v3.example.json` | 相对路径的示例配置；原配置另行保留 |
| `configs_data/` | 原版本配置与不可自动刷新的锁 |
| `configs_models/` | 三个编码器的固定版本与哈希；权重不入 Git |
| `tools/` | 保留的维护工具及锁定依赖 |
| `resercher/` | 按原要求独立保存的 CLIP 门控生物学研究 |
| `docs/history/` | 旧方案与诊断记录，不作为当前运行说明 |

清理详情见 [清单](docs/PIPER_CLEANUP_20261026.json)，测试结果见 [验证报告](docs/RELEASE_VERIFICATION.md)。大型数据、checkpoint、训练日志、旧运行快照与服务器副本不上传 GitHub。

沿用上游 CC BY-NC 4.0 许可证及署名。Piper 为独立衍生研究项目，不是 FlashPPI 官方版本。
