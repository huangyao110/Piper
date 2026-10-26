# 后续训练：随机序列 mask

2026-10-12 按用户要求加入输入增强。当前后台 `flashppi.finetune` 训练及其固定源码、配置保持不变；新功能接入 `flashppi.research_train`，没有启动新训练。它是待评估的正则化方案，不预先假定能提高指标。

## 固定起始方案

- 参数片段：`configs_train/sequence_masking_v1.json`，`sequence_masking.probability = 0.15`。这是待合并到完整训练配置的片段，不能单独启动训练；旧配置缺少该字段时保持关闭。
- 在链交换、界面裁剪、tokenization 和标签构建后，两条链的每个标准氨基酸位置分别以 15% 概率独立入选，选中位置全部替换为 PLM 的原生 mask token，不使用 80/10/10 替换策略。每次采样重新选择位置，短链至少保留一个未遮蔽的标准残基；不保证每条链恰好遮蔽 15%。
- 不删除残基、不改变序列长度或坐标，不遮蔽 BOS/EOS/PAD 和未知残基。不根据接触标签挑选 mask 位置。
- masked 位置的 attention mask 仍为 1。互作标签、接触标签和结构有效性掩码保持原样；未来距离监督也保留原坐标及有效性。mask 表示隐藏输入氨基酸，不表示结构缺失或不参与 loss。
- 仅做监督训练的输入增强，没有新增 MLM 预测头或 MLM loss。val/test 始终使用完整、未 mask 的序列。

## 实现与复现

`flashppi/sequence_masking.py` 的 `SequenceMaskingCollator` 包装现有训练 collator；`ResidueMasker` 可用于 GLM2 或原生 ESMC tokenizer。已在本地 tokenizer 核实 GLM2 mask ID 为 35，ESMC 为 32；实现读取 tokenizer 的 ID，不硬编码。

双 PLM 分支使用相同遮蔽位置，并映射到各自原生 mask token。两次 gradient-cache 前向重用同一批已遮蔽 token，避免表示缓存与反向重算不一致。随机种子沿用全局 step/microbatch 种子，不依赖分配到哪张卡；配置和新模块源码进入 checkpoint 审计。开启时，history 记录实际遮蔽数量、可遮蔽残基数与比例，按所有 rank 汇总。

现有 research 入口通过配置字段或 `--sequence-mask-probability 0.15` 开启，`0` 关闭。CLI 值会写入本次完整配置；恢复时仍须保持配置与源码一致，不能在同一条实验轨迹中悄悄改变 mask 比例。

后续 DDI／ESMC-only 预训练入口尚未在本次实现；届时复用本模块并显式开启 15%，不把当前 GLM2 融合配置当作该实验。对照保持数据划分、初始化、训练预算和其他超参数一致，仅比较 0% 与 15%。是否采用由现有 val 的 CLIP AP、接触 P@K 判断；test 不用于选择比例。

## 检查

```bash
.venv/server/bin/python tests/test_sequence_masking.py
.venv/server/bin/python tests/test_research_multigpu.py
```

检查仅使用 CPU 小模型，无权重下载；覆盖特殊 token、标签/裁剪对齐、关闭时随机数不变、比例与种子复现、多 rank 分配、原生 mask 映射及缓存重算梯度一致性。真实 GPU 吞吐和最终指标收益留待下一次正式实验前验证。

本次结果：新增 8 项 masking 检查、原有 6 项 research 回归检查均通过；本地 GLM2/ESMC 原生 tokenizer 的遮蔽与跨词表映射检查通过。后台训练所固定的全部源码和配置哈希保持一致。
