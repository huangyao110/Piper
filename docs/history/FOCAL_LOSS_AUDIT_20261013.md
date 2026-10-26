# Focal 接触损失端到端检查（2026-10-13）

检查时间：2026-10-13T16:39:57.965692+08:00。检查时后台已完成 step 1636。
本次未改变训练源码、配置或进程；所有诊断在 CPU 运行，未进行 optimizer.step。

## 结论

未发现 Focal 逐元素公式、掩码、反传累积或 checkpoint 迁移错误。
确认当前目标只恢复了旧 LoRA 的 Focal 核函数，并未复现旧版完整归一化和负样本监督。
当前归一化本身是合法的目标定义，不应称为数学错误；但它改变了接触监督的尺度和结构之间的相对权重。
此前通过的实现/显存预检不能证明多任务梯度平衡合理，也不能证明 Focal 提升验证性能。

## 检查链路

| 环节 | 结果 |
|---|---|
| 模型输出 | 接触头输出原始 logits，没有重复 sigmoid；无独立互作分类头 |
| 标签几何 | float64 坐标严格距离 <12 Å；1024 对重放样本均与原始接触标签一致 |
| 裁剪与链交换 | 序列、坐标、有效掩码及接触索引同步；1024 对裁剪后均保留接触 |
| MASK | 只改变输入 token，不删除对应结构监督；40637/269867 与真实 step 1601 日志相同 |
| Focal 核函数 | γ=2、α=0.25；正位置 α=0.25，负位置 α=0.75；exp(-BCE) 对二值标签等于 p_t |
| 掩码 | padding 与未知坐标无直接接触 logit 梯度；仅真实正例结构计算接触/距离损失 |
| 精度 | 训练 logits 为 BF16，loss 核函数显式转 FP32；独立数值核验用 FP64 |
| 梯度累积 | 每微批按实际正例数/全局正例数加权；没有重复除以 1024 |
| 任务权重 | 总 loss = 0.1×CLIP + contact + 0.1×distance；逐条日志加和误差 0.0 |
| 续训 | 权重文件、全部 Adam 状态、随机状态和 step1600 均原样保留，已存迁移逐项核验报告 |
| 源码身份 | 活动源码与 protocol/preflight/checkpoint 对应哈希一致 |
| 选模 | best 仍按固定 val 的 CLIP AP；接触 Precision@K 被记录但不直接参与选模 |

## 真实 batch 与归一化

从 checkpoint1600 的 NumPy RNG 重放 step1601 的全局 1024 对采样，并使用原始种子重放完整裁剪及 MASK。
接触位置占全部有效位置 **1.0233%**；逐结构接触密度中位数 **1.0741%**。
每个结构的有效位置数/接触正位置数中位数为 **93.10**，范围 **12.86–1947.56**。

记第 i 个结构的有效位置数为 N_i、接触正位置数为 K_i、逐元素 Focal 和为 S_i：

- 当前目标：mean_i(S_i / N_i)。
- 逐结构正接触归一化目标：mean_i(S_i / K_i)。
- 旧 LoRA 的微批归一化：sum_i(S_i) / sum_i(K_i)，另含旧协议的合成负对。

逐结构比较时 S_i/N_i = (K_i/N_i)×(S_i/K_i)。因此两种归一化不只是换一个固定常数，接触密度不同的结构相对权重也不同。
本次比较旧版分母时使用相同真实正例，没有重新引入旧协议的合成负对，不能把诊断值当作历史 LoRA loss。

## 相同预测下的实测

使用 checkpoint1600、真实 step1601 的完整第一个 40 对微批；保持原始 padding/MASK。
模型在 CPU FP32 eval 模式计算。梯度范数是损失对两条链共享 ESMC 输出嵌入的 L2 范数，**不是 ESMC 参数梯度范数或实际 Adam 更新量**。

| 目标 | Loss | 共享嵌入梯度范数 |
|---|---:|---:|
| BCE，当前有效位置归一化 | 0.055179 | 0.014610 |
| 当前 Focal，有效位置归一化 | 0.009410 | 0.004270 |
| Focal，旧版微批接触正位置数归一化 | 0.792430 | 0.240288 |
| Focal，逐结构接触正位置数归一化 | 0.793728 | 0.272320 |
| 0.1 × 距离分箱交叉熵 | 0.173693 | 0.011415 |

当前 Focal 嵌入梯度为 BCE 的 **29.2%**、加权距离损失的 **37.4%**。
直接使用旧版微批正接触数分母、同时保持 contact_weight=1，会使本微批接触嵌入梯度变为当前的 **56.3 倍**。
这些是尺度检查结果，不能仅凭梯度大/小断言哪个目标一定更好。
Focal 的正位置贡献占其未归一化分子 **97.99%**，因此本微批没有出现 Focal 分子仍被容易负位置支配的现象。
独立 log-sigmoid FP64 公式核验：loss 绝对误差 4.19e-10，logit 梯度最大绝对误差 8.51e-13。

## 判断与下一步

- loss 从约0.05降至约0.009主要包含目标定义变化，不能称为性能改善。
- 恢复 Focal 核函数后，contact_weight=1 并不保证保持此前 BCE 或 LoRA 的接触监督强度。
- 若目标是恢复旧 LoRA 的监督尺度，应显式选择归一化，再检查与其他任务的梯度比例；不能直接改分母同时默认权重1仍合适。
- 采用同一 checkpoint、相同样本顺序的短程对照，用固定 val 的接触 Precision@K 与 CLIP AP 判断；不自动调参、不用 test 选方案。
- 本次只有1个完整微批的模型梯度诊断，完整全局 batch 只核验标签/密度/采样；未测全局1024 CLIP梯度，且CPU FP32不同于生产BF16。
- Focal 阶段尚未到第一次 step1800 验证，不能据此判断其泛化提升或下降。

## 复现与记录

脚本：`tools/diagnose_ddi_focal.py`。运行目录：`experiments/ddi_focal/`。
完整微批结果：`checks/focal_audit_full_micro40_20261013/`；初步8对结果：`checks/focal_audit_20261013/`。
原始迁移核验：`checks/migration_verification.json`。

```bash
CUDA_VISIBLE_DEVICES='' ../../.venv/server/bin/python -u ../../tools/diagnose_ddi_focal.py --source-root . --checkpoint /media/zlab/lexar4t/FlashPPI/runs/ddi_esmc300_focal_resume1600_20261013/resume_initial --output checks/focal_audit_reproduction --prediction-pairs 40
```

标准公式参照：[Torchvision 官方 sigmoid_focal_loss 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html)。
旧实现：`flashppi/loss.py` 的 `focal_loss` 和 `PPIGradientCache`；当前实现：`experiments/ddi_focal/flashppi/ddi_objectives.py` 与 `ddi_engine.py`。
