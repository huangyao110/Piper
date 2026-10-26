#!/usr/bin/env python3
"""Exploratory gate/conservation/SASA associations with proteins as observations."""
import sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
import math

from clip_gate.common import ROOT, local_path, read_json, setup_environment, sha256, stable_seed, write_json
setup_environment()
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata
from clip_gate.analysis import annotation_vector, load_annotations, load_result
from clip_gate.statistics import bootstrap_mean, safe_spearman, top_enrichment

MIN_SITES = 30
MIN_FRACTION = .5
MIN_ADJUSTED_SITES = 50
BOOTSTRAPS = 2000
FINDINGS = """这批 val 的结果支持 gate 偏好表面暴露残基。step 2182 的 gate–RSA 相关为
q=+0.229、k=+0.245，且控制氨基酸种类后仍为正。gate–保守性的原始弱负相关
（约 −0.123）在控制氨基酸种类和 RSA 后接近零，95% 区间跨零。因此目前未发现
稳定的独立保守性偏好，不能把 gate 等同于功能重要性或突变热点。

相比 step 1616，表面偏好更明显；界面富集没有同步增强。控制 AA+RSA 后仍有
小幅界面超额富集（q=+0.062、k=+0.076），提示表面偏好不是界面富集的唯一来源，
但这仍是探索性关联，不能据此作因果解释。"""


def rank_partial_correlation(gate, outcome, aa, rsa=None):
    """Correlation of rank residuals, adjusting AA indicators and optional RSA rank.

    This is a descriptive within-protein linear adjustment, not a causal estimate.
    At least 20 residual degrees of freedom and 50 valid sites are required.
    """
    gate, outcome, aa = np.asarray(gate), np.asarray(outcome), np.asarray(aa)
    valid = np.isfinite(gate) & np.isfinite(outcome)
    if rsa is not None:
        rsa = np.asarray(rsa)
        valid &= np.isfinite(rsa)
    if valid.sum() < MIN_ADJUSTED_SITES:
        return None
    g, y = rankdata(gate[valid]), rankdata(outcome[valid])
    letters = aa[valid]
    columns = [np.ones(len(g))] + [(letters == a).astype(float) for a in np.unique(letters)[1:]]
    if rsa is not None:
        columns.append(rankdata(rsa[valid]) / len(g))
    design = np.column_stack(columns)
    if len(g) - np.linalg.matrix_rank(design) < 20:
        return None
    g = g - design @ np.linalg.lstsq(design, g, rcond=None)[0]
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(g) < 1e-8 or np.std(y) < 1e-8:
        return None
    return float(np.corrcoef(g, y)[0, 1])


def eligible_vector(vector, length):
    n = int(np.isfinite(vector).sum())
    return n >= MIN_SITES and n / length >= MIN_FRACTION


def aggregate(rows, metric, branch, checkpoint):
    selected = [r for r in rows if r["branch"] == branch and r["checkpoint"] == checkpoint
                and r.get(metric) is not None and np.isfinite(r[metric])]
    by_block = defaultdict(list)
    for row in selected:
        by_block[row["block_id"]].append(row[metric])
    values = [float(np.mean(v)) for _, v in sorted(by_block.items())]
    return dict(sequences=len(selected), positive_sequences=sum(r[metric] > 0 for r in selected),
                sequence_mean=float(np.mean([r[metric] for r in selected])) if selected else None,
                block=bootstrap_mean(values, BOOTSTRAPS, stable_seed(20261023, metric, branch, checkpoint)))


def build_rows():
    config = read_json(ROOT / "study.json")
    interfaces = pq.read_table(ROOT / "data/interfaces.parquet").to_pylist()
    sasa = read_json(ROOT / "manifests/annotations_sasa.json")["results"]
    completeness = {r["target"]["sequence_id"]: np.mean([x["complete_heavy_atoms"] for x in r["residues"]])
                    for r in sasa}
    rows = []
    for checkpoint in ("before_1616", "after_2182"):
        _, _, sequences, gates = load_result(config, checkpoint)
        annotations, _ = load_annotations("data/annotations.csv", sequences)
        for item in interfaces:
            sid = item["sequence_id"]
            sequence = sequences[sid]
            length = len(sequence)
            aa = np.array(list(sequence))
            rsa = annotation_vector(annotations, sid, length, "relative_sasa")
            cons = annotation_vector(annotations, sid, length, "conservation")
            neff = annotation_vector(annotations, sid, length, "msa_neff")
            joint = np.isfinite(rsa) & np.isfinite(cons)
            cons_joint = np.where(joint, cons, np.nan)
            cons_neff30 = np.where(neff >= 30, cons, np.nan)
            labels = np.array(item["labels"])
            for branch in ("q", "k"):
                gate = gates[(sid, branch)]["relative"]
                row = dict(sequence_id=sid, block_id=item["block_id"], checkpoint=checkpoint,
                           branch=branch, length=length, rsa_n=int(np.isfinite(rsa).sum()),
                           conservation_n=int(np.isfinite(cons).sum()), joint_n=int(joint.sum()),
                           heavy_atom_complete_fraction=float(completeness[sid]))
                row["gate_rsa_rho"] = safe_spearman(gate, rsa) if eligible_vector(rsa, length) else None
                row["gate_rsa_rho_complete_chains"] = row["gate_rsa_rho"] if completeness[sid] >= .95 else None
                row["gate_rsa_aa_adjusted"] = rank_partial_correlation(gate, rsa, aa) if eligible_vector(rsa, length) else None
                row["gate_conservation_rho"] = safe_spearman(gate, cons) if eligible_vector(cons, length) else None
                row["gate_conservation_rho_neff30"] = safe_spearman(gate, cons_neff30) if eligible_vector(cons_neff30, length) else None
                row["gate_conservation_rho_joint"] = safe_spearman(gate, cons_joint) if eligible_vector(cons_joint, length) and joint.sum() >= MIN_ADJUSTED_SITES else None
                row["gate_conservation_aa_rsa_adjusted"] = rank_partial_correlation(gate, cons, aa, rsa) if eligible_vector(cons_joint, length) else None
                row["conservation_rsa_rho"] = safe_spearman(cons, rsa) if eligible_vector(cons_joint, length) else None
                for threshold, name in ((0., "sasa_annotated"), (.25, "surface_only")):
                    valid = np.isfinite(rsa) & (rsa >= threshold) & (labels >= 0)
                    row[f"interface_enrichment_{name}"] = (top_enrichment(labels[valid], gate[valid], .1)
                                                           if valid.sum() >= 30 and len(np.unique(labels[valid])) == 2 else None)
                rows.append(row)
    return rows


def annotation_quality():
    cons = read_json(ROOT / "manifests/annotations_conservation.json")["results"]
    sasa = read_json(ROOT / "manifests/annotations_sasa.json")["results"]
    site_counts = Counter()
    for result in cons:
        for r in result["residues"]:
            site_counts.update(r["exclusion"].split(";") if r["exclusion"] else ["accepted"])
    return dict(
        conservation_status=dict(Counter(r["status"] for r in cons)),
        conservation_rejection_reasons=dict(Counter(r.get("error", "") for r in cons if r["status"] != "mapped")),
        conservation_site_exclusions=dict(site_counts),
        conservation_chains_with_any=sum(any(r["conservation"] is not None for r in s["residues"]) for s in cons),
        conservation_positions=sum(r["conservation"] is not None for s in cons for r in s["residues"]),
        msa_neff_median=float(np.median([s["msa"]["neff"] for s in cons if s["status"] == "mapped"])),
        msa_neff_range=[float(f([s["msa"]["neff"] for s in cons if s["status"] == "mapped"])) for f in (min, max)],
        total_positions=sum(len(s["residues"]) for s in sasa),
        sasa_complete_positions=sum(r["complete_heavy_atoms"] for s in sasa for r in s["residues"]),
        sasa_rsa_above_one=sum((r["rsa_unclipped"] or 0) > 1 for s in sasa for r in s["residues"]),
        sasa_chains_at_least_95_percent_complete=int(sum(np.mean([r["complete_heavy_atoms"] for r in s["residues"]]) >= .95 for s in sasa)))


def fmt(metric):
    block = metric["block"]
    if block["mean"] is None:
        return "NA"
    ci = block["ci95"]
    return f"{block['mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]；n={metric['sequences']}" if ci else f"{block['mean']:+.3f}"


def plot(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    metrics = [("gate_rsa_rho", "Surface accessibility (RSA)"),
               ("gate_conservation_rho", "Conservation (ConSurf)"),
               ("gate_conservation_aa_rsa_adjusted", "Conservation, adjusted for AA + RSA")]
    for ax, (metric, title) in zip(axes, metrics):
        for i, (checkpoint, branch, label, color) in enumerate([
            ("before_1616", "q", "1616 / q", "#9e9e9e"), ("before_1616", "k", "1616 / k", "#6b6b6b"),
            ("after_2182", "q", "2182 / q", "#168e9d"), ("after_2182", "k", "2182 / k", "#225ea8")]):
            data = summary[checkpoint][branch][metric]["block"]
            mean, ci = data["mean"], data["ci95"]
            if mean is not None:
                ax.errorbar(mean, 3 - i, xerr=[[mean - ci[0]], [ci[1] - mean]], fmt="o", capsize=4, color=color)
        ax.axvline(0, color="#888888", ls="--", lw=1)
        ax.set_yticks(range(4), ["2182 / k", "2182 / q", "1616 / k", "1616 / q"])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mean within-chain rank correlation")
        ax.set_ylim(-.5, 3.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CLIP residue gates: associations with 95% block-bootstrap intervals", fontsize=12)
    fig.savefig(directory / "annotation_associations.png", dpi=200)
    fig.savefig(directory / "annotation_associations.pdf")
    plt.close(fig)


def main():
    directory = local_path("results/annotation_study")
    directory.mkdir(exist_ok=False)
    rows = build_rows()
    metrics = [k for k in rows[0] if k.startswith("gate_") or k.startswith("interface_enrichment") or k == "conservation_rsa_rho"]
    summary = {checkpoint: {branch: {metric: aggregate(rows, metric, branch, checkpoint) for metric in metrics}
                           for branch in ("q", "k")} for checkpoint in ("before_1616", "after_2182")}
    quality = annotation_quality()
    controls = {checkpoint: read_json(ROOT / f"results/{checkpoint}/analysis_annotated/summary.json")["interface"]
                for checkpoint in summary}
    movement = {}
    for checkpoint in summary:
        a = pq.read_table(ROOT / f"results/{checkpoint}/analysis_annotated/interface_per_structure_side.parquet").to_pylist()
        movement[checkpoint] = {branch: float(np.mean([r["aa_rsa_movable_fraction"] for r in a if r["branch"] == branch
                                and r.get("aa_rsa_movable_fraction") is not None])) for branch in ("q", "k")}
    pq.write_table(pa.Table.from_pylist(rows), directory / "per_chain_associations.parquet")
    provenance = {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT / "data/annotations.csv", ROOT / "data/annotations.json",
                  ROOT / "data/annotation_details.parquet", ROOT / "annotation_report.py", ROOT / "clip_gate/annotations.py"]}
    write_json(directory / "summary.json", dict(quality=quality, correlations=summary, interface_controls=controls,
               aa_rsa_permutation_movable_fraction=movement, provenance=provenance,
               analysis_policy=dict(min_sites=MIN_SITES, min_chain_annotation_fraction=MIN_FRACTION,
                                    min_adjusted_sites=MIN_ADJUSTED_SITES, bootstrap_replicates=BOOTSTRAPS,
                                    aggregation="within chain, then mean within fixed dependence block; bootstrap blocks",
                                    adjustment="Pearson correlation of rank residuals after within-chain AA indicators + RSA rank")))
    lines = ["# CLIP 残基门控：保守性与表面暴露度", "", "2026-10-23；固定 val 100 个结构、200 条链；checkpoint 1616 与 2182。",
             "只使用已提取的 CLIP residue gates。后台训练、数据成员、同源过滤规则及接触标签均不改变。", "", FINDINGS, "",
             "## 注释质量", "",
             f"- 总计 {quality['total_positions']:,} 个残基；{quality['sasa_complete_positions']:,} 个重原子完整，允许计算 RSA 关联。",
             f"- {quality['sasa_chains_at_least_95_percent_complete']} 条链至少 95% 残基重原子完整，用于结构完整性敏感性检查。",
             f"- 保守性下载/映射状态：{quality['conservation_status']}；{quality['conservation_chains_with_any']} 条链有合格位点，共 {quality['conservation_positions']:,} 个位点。",
             f"- 成功映射 MSA 的整体 Neff 中位数 {quality['msa_neff_median']:.1f}，范围 {quality['msa_neff_range']}。Neff 作为质量诊断，不替代 Rate4Site 的系统发育模型。",
             "- 每条链至少 30 个且至少 50% 位点有注释才计入原始相关；调整后相关至少需要 50 个联合位点及 20 个残差自由度。表内 n 是实际可分析链数。", "",
             "## 与 gate 的关系", "", "数值为链内相关先按固定依赖块等权平均；方括号为 2,000 次块 bootstrap 的 95% 区间。正号表示 gate 越大，对应注释值越大。",
             "q/k 为同一模型的两条分支，不能作为两次独立实验。", "",
             "| 指标 | 1616 q | 2182 q | 2182 k |", "|---|---|---|---|"]
    labels = {
        "gate_rsa_rho": "表面暴露度 Spearman ρ",
        "gate_rsa_rho_complete_chains": "表面暴露度：≥95% 完整链",
        "gate_rsa_aa_adjusted": "表面暴露度：控制氨基酸种类",
        "gate_conservation_rho": "保守性 Spearman ρ",
        "gate_conservation_rho_neff30": "保守性：位点 Neff≥30",
        "gate_conservation_rho_joint": "保守性：联合注释子集原始 ρ",
        "gate_conservation_aa_rsa_adjusted": "保守性：控制氨基酸种类 + RSA",
        "conservation_rsa_rho": "保守性与 RSA 自身的关联",
    }
    for metric, label in labels.items():
        values = [summary[c][b][metric] for c, b in (("before_1616", "q"), ("after_2182", "q"), ("after_2182", "k"))]
        lines.append("| " + label + " | " + " | ".join(fmt(x) for x in values) + " |")
    lines.extend(["", "## 界面富集在控制暴露度之后", "",
                  "已有 Cβ<12 Å（Gly 使用 CA）标签保持原样；Top 10% gate 的界面富集 = 所选位点的界面比例 / 可用位点的界面比例。1 为无富集。",
                  "AA+RSA 控制为在每条链的氨基酸种类 × 10 个 RSA 分箱内打乱 gate；超额富集是观察富集减去 100 次打乱平均值。0 表示未超出这一对照。", "",
                  "| Checkpoint/分支 | 有 RSA 的位点富集 | 仅表面 RSA≥0.25 的富集 | AA+RSA 控制后超额富集 | 可置换位点比例 |", "|---|---|---|---|---|"])
    for c in summary:
        for b in ("q", "k"):
            values = summary[c][b]
            adjusted = controls[c][b]["metrics"]["aa_rsa_excess_enrichment"]["block_macro"]
            ci = adjusted["ci95"]
            lines.append(f"| {c}/{b} | {fmt(values['interface_enrichment_sasa_annotated'])} | {fmt(values['interface_enrichment_surface_only'])} | "
                         f"{adjusted['mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {movement[c][b]:.1%} |")
    lines.extend(["", "## 方法与解释边界", "",
                  "- 链对应通过只读 CongLab 原始 component catalog 的序列哈希确认；不能把 original_id 的排列顺序直接当作整理后的 A/B。然后将实际坐标序列与 ConSurf 参考序列作唯一全局对齐，只转移完全相同的氨基酸位置；identity≥99%、目标序列覆盖≥95%，多解对齐拒绝。",
                  "- SASA 使用 [Biopython Shrake–Rupley](https://biopython.org/docs/latest/api/Bio.PDB.SASA.html)，探针 1.4 Å、每原子 960 点；移除伙伴链、配体、水和氢，保留实验复合物中的构象。用 [Tien et al. 2013](https://doi.org/10.1371/journal.pone.0080635) 最大 ASA 归一化。自身重原子不完整的残基不用于关联；邻近残基缺失原子仍可能影响 SASA。少数 RSA>1 在分箱时截至 1，原始 ASA/RSA 完整保存在明细中。",
                  "- 保守性来自 [ConSurfDB](https://consurfdb.tau.ac.il/overview/) 的既有 Rate4Site 分数，取负号使越高越保守；未重建同源检索或修改数据集去重。下载的真实 MSA 用于质量检查：只保留覆盖≥50% query 列的序列，80% identity 重加权，邻居须覆盖较长序列至少 50%，位点 Neff≥10、非 gap 比例≥50%，并排除数据库标 * 的不足数据位置。更严格的 Neff≥30 结果另列。",
                  "- 原始 MSA、分数、请求 URL、时间和哈希在 cache/annotations/consurf；逐残基映射、丢弃原因和质量在 cache/annotations/conservation 与 data/annotation_details.parquet。不可用或不合格位置保持缺失，不填 0。",
                  "- 分数是相关性，不能把 gate 当作接触概率、突变效应或功能热点。表面暴露度来自孤立链的结合态构象，不等于实测游离态；ConSurf 同源家族可能包含旁系同源蛋白。",
                  "- 氨基酸和 RSA 控制只是描述性调整；小分箱限制置换空间，未排除二级结构、柔性、家族及更多偏差。结构簇块不是穷尽的同源独立性保证。",
                  "- 这是同一 val 上的探索性后续分析，没有使用独立确认集。编码器训练同时接受接触监督，因此观察到的生物学关联不能称作完全无监督的涌现。", "",
                  "![关联图](annotation_associations.png)", "",
                  "原始统计：summary.json、per_chain_associations.parquet；逐残基注释：../../data/annotations.csv 和 annotation_details.parquet。"])
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    plot(summary, directory)
    print(directory / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
