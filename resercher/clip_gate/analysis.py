"""Exploratory interface association and CLIP gate-intervention analysis."""
import csv
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import (ROOT, check_hash, local_path, read_json, sha256, stable_seed,
                     verify_prepared, write_json)
from .panel import write_table
from .statistics import (bootstrap_mean, pair_scores, residue_metrics, safe_spearman,
                         shuffled_within_groups, top_enrichment)


def load_annotations(path, sequences):
    """Strict exact-sequence mapping; an absent annotation is never zero-filled."""
    if path is None:
        return {}, {"status": "not_supplied", "conservation": False, "relative_sasa": False}
    path = local_path(path)
    metadata_path = path.with_suffix(".json")
    metadata = read_json(metadata_path)
    if metadata.get("conservation_direction") != "higher_is_more_conserved":
        raise ValueError("Annotation metadata must declare higher_is_more_conserved")
    if not metadata.get("method") or not metadata.get("source"):
        raise ValueError("Annotation method and source are required")
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sid, position = row["sequence_id"], int(row["position"])
            if sid not in sequences or not 1 <= position <= len(sequences[sid]):
                raise ValueError("Annotation refers to an unknown sequence or position")
            if row["aa"] != sequences[sid][position - 1]:
                raise ValueError("Annotation amino acid does not match the exact sequence")
            key = (sid, position)
            if key in result:
                raise ValueError("Duplicate residue annotation")
            values = {field: float(row[field]) if row.get(field, "").strip() else np.nan
                      for field in ("conservation", "relative_sasa", "msa_neff", "msa_coverage")}
            for field, value in values.items():
                if np.isinf(value):
                    raise ValueError(f"Infinite annotation: {field}")
                if field in ("relative_sasa", "msa_coverage") and np.isfinite(value) and not 0 <= value <= 1:
                    raise ValueError(f"{field} must be a fraction in [0,1]")
                if field == "msa_neff" and np.isfinite(value) and value < 0:
                    raise ValueError("MSA Neff cannot be negative")
            result[key] = values
    return result, dict(status="supplied", rows=len(result), metadata=metadata, sha256=sha256(path),
                        metadata_sha256=sha256(metadata_path),
                        conservation=any(np.isfinite(x["conservation"]) for x in result.values()),
                        relative_sasa=any(np.isfinite(x["relative_sasa"]) for x in result.values()))


def annotation_vector(annotations, sid, length, field):
    return np.array([annotations.get((sid, i + 1), {}).get(field, np.nan) for i in range(length)])


def load_result(config, name):
    manifest = verify_prepared(config)
    directory = local_path(f"results/{name}")
    status = read_json(directory / "status.json")
    if status["status"] != "complete":
        raise ValueError("Extraction is incomplete; do not report partial panels as full results")
    provenance = read_json(directory / "provenance.json")
    if provenance["prepared_sha256"] != sha256(ROOT / "manifests/prepared.json"):
        raise ValueError("Prepared panel changed since extraction")
    for filename, digest in provenance["files"].items():
        check_hash(directory / filename, digest)
    sequences = {r["sequence_id"]: r["sequence"] for r in
                 pq.read_table(local_path("data/sequences.parquet")).to_pylist()}
    grouped = defaultdict(list)
    for row in pq.read_table(directory / "gates.parquet").to_pylist():
        grouped[(row["sequence_id"], row["branch"])].append(row)
    if set(grouped) != {(sid, branch) for sid in sequences for branch in ("q", "k")}:
        raise ValueError("Missing or unexpected sequences/branches in extracted gate table")
    gates = {}
    for (sid, branch), rows in grouped.items():
        rows.sort(key=lambda r: r["position"])
        sequence = sequences[sid]
        if [r["position"] for r in rows] != list(range(1, len(sequence) + 1)):
            raise ValueError("Duplicate or missing residue position")
        if "".join(r["aa"] for r in rows) != sequence:
            raise ValueError("Gate/sequence alignment mismatch")
        raw = np.array([r["raw_gate"] for r in rows])
        alpha = np.array([r["normalized_weight"] for r in rows])
        relative = np.array([r["relative_weight"] for r in rows])
        if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any() or raw.sum() <= 0:
            raise ValueError("Invalid gate weights")
        np.testing.assert_allclose(alpha, raw / raw.sum(), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(relative, len(sequence) * alpha, rtol=1e-8, atol=1e-10)
        gates[(sid, branch)] = dict(raw=raw, alpha=alpha, relative=relative)
    return manifest, directory, sequences, gates


def association_rows(config, sequences, gates, interfaces, annotations):
    result = []
    for item in interfaces:
        sid = item["sequence_id"]
        labels = np.asarray(item["labels"])
        aa = np.array(list(sequences[sid]))
        rsa = annotation_vector(annotations, sid, len(labels), "relative_sasa")
        for branch in ("q", "k"):
            weights = gates[(sid, branch)]["relative"]
            metrics = residue_metrics(labels, weights, config["top_fraction"])
            if metrics is None:
                result.append({**{k: v for k, v in item.items() if k != "labels"}, "branch": branch,
                               "eligible": False, "reason": "fewer_than_two_observed_classes"})
                continue
            row = {**{k: v for k, v in item.items() if k != "labels"}, "branch": branch,
                   "eligible": True, **metrics}
            known = labels >= 0
            rng = np.random.default_rng(stable_seed(config["seed"], sid, branch, "aa_control"))
            null = [top_enrichment(labels[known], shuffled_within_groups(weights[known], aa[known], rng),
                                   config["top_fraction"]) for _ in range(config["annotation_permutations"])]
            row["aa_null_enrichment"] = float(np.mean(null))
            row["aa_excess_enrichment"] = row["top_enrichment"] - row["aa_null_enrichment"]
            row["rsa_observed_residues"] = int(np.sum(known & np.isfinite(rsa)))
            valid_rsa = known & np.isfinite(rsa)
            rsa_metrics = residue_metrics(labels[valid_rsa], weights[valid_rsa], config["top_fraction"])
            row["aa_rsa_excess_enrichment"] = None
            row["aa_rsa_movable_fraction"] = None
            if rsa_metrics is not None:
                # Ten predeclared relative-SASA bins, stratified jointly by AA.
                bins = np.minimum((rsa[valid_rsa] * 10).astype(int), 9)
                groups = np.array([f"{residue}:{bin_id}" for residue, bin_id in zip(aa[valid_rsa], bins)])
                _, counts = np.unique(groups, return_counts=True)
                row["aa_rsa_movable_fraction"] = float(counts[counts > 1].sum() / len(groups))
                rng = np.random.default_rng(stable_seed(config["seed"], sid, branch, "aa_rsa_control"))
                null = [top_enrichment(labels[valid_rsa], shuffled_within_groups(weights[valid_rsa], groups, rng),
                                       config["top_fraction"]) for _ in range(config["annotation_permutations"])]
                row["aa_rsa_excess_enrichment"] = rsa_metrics["top_enrichment"] - float(np.mean(null))
            result.append(row)
    return result


def aggregate_associations(config, rows):
    result = {}
    for branch in ("q", "k"):
        selected = [r for r in rows if r["branch"] == branch and r["eligible"]]
        summary = dict(eligible_structure_sides=len(selected),
                       excluded_structure_sides=sum(r["branch"] == branch and not r["eligible"] for r in rows),
                       metrics={})
        for metric in ("ap", "auroc", "interface_prevalence", "top_enrichment",
                       "aa_excess_enrichment", "aa_rsa_excess_enrichment"):
            per_sequence = defaultdict(list)
            blocks = {}
            for row in selected:
                if row.get(metric) is not None:
                    per_sequence[row["sequence_id"]].append(row[metric])
                    blocks[row["sequence_id"]] = row["block_id"]
            # First average repeated structures/partners for a sequence, then
            # average sequences in a dependence block. Residues are not replicates.
            per_block = defaultdict(list)
            for sid, values in per_sequence.items():
                per_block[blocks[sid]].append(float(np.mean(values)))
            values = [float(np.mean(v)) for _, v in sorted(per_block.items())]
            summary["metrics"][metric] = dict(
                sequence_macro=float(np.mean([np.mean(v) for v in per_sequence.values()])) if per_sequence else None,
                unique_sequences=len(per_sequence), block_macro=bootstrap_mean(
                    values, config["bootstrap_replicates"], stable_seed(config["seed"], branch, metric)))
        result[branch] = summary
    return result


def interaction_report(scores):
    groups = defaultdict(list)
    for row in scores:
        groups[row["variant"]].append(row)
    result = {}
    for variant, rows in groups.items():
        labels = np.array([r["label"] for r in rows])
        predictions = np.array([r["clip_score"] for r in rows])
        if len(np.unique(labels)) != 2:
            raise ValueError("Interaction panel needs both labeled classes")
        queries = defaultdict(dict)
        for row in rows:
            # Ranking treats repeated labeled candidate pairs only once.
            prior = queries[row["sequence1"]].get(row["sequence2"])
            if prior is not None and prior["label"] != row["label"]:
                raise ValueError("Conflicting labels for the same oriented candidate pair")
            queries[row["sequence1"]][row["sequence2"]] = row
        ranks, margins = [], []
        for candidates in queries.values():
            positive = [r["clip_score"] for r in candidates.values() if r["label"] == 1]
            negative = [r["clip_score"] for r in candidates.values() if r["label"] == 0]
            if positive and negative:
                scores_here = np.array([r["clip_score"] for r in candidates.values()])
                # Best positive rank with midranks for equal scores.
                best = max(positive)
                rank = 1 + np.sum(scores_here > best) + .5 * (np.sum(scores_here == best) - 1)
                ranks.append(float(1 / rank))
                margins.append(float(np.mean(positive) - np.mean(negative)))
        result[variant] = dict(ap=float(average_precision_score(labels, predictions)),
                               auroc=float(roc_auc_score(labels, predictions)),
                               rows=len(rows), positives=int(labels.sum()),
                               eligible_labeled_candidate_queries=len(ranks),
                               labeled_candidate_mrr=float(np.mean(ranks)) if ranks else None,
                               positive_negative_margin=float(np.mean(margins)) if margins else None)
    baseline = result["learned"]
    for values in result.values():
        values["ap_delta_from_learned"] = values["ap"] - baseline["ap"]
        values["auroc_delta_from_learned"] = values["auroc"] - baseline["auroc"]
    return result


def analyze(config, name, annotation_path=None):
    manifest, directory, sequences, gates = load_result(config, name)
    annotations, annotation_info = load_annotations(annotation_path, sequences)
    output = local_path(directory / ("analysis" if annotation_path is None else "analysis_annotated"))
    output.mkdir(exist_ok=False)
    interfaces = pq.read_table(local_path("data/interfaces.parquet")).to_pylist()
    associations = association_rows(config, sequences, gates, interfaces, annotations)
    distributions = []
    for (sid, branch), values in gates.items():
        alpha, raw = values["alpha"], values["raw"]
        conservation = annotation_vector(annotations, sid, len(raw), "conservation")
        rsa = annotation_vector(annotations, sid, len(raw), "relative_sasa")
        positive = alpha > 0
        distributions.append(dict(sequence_id=sid, branch=branch, length=len(raw), raw_mean=float(raw.mean()),
                                  raw_min=float(raw.min()), raw_max=float(raw.max()),
                                  relative_q10=float(np.quantile(values["relative"], .1)),
                                  relative_q90=float(np.quantile(values["relative"], .9)),
                                  effective_residue_fraction=float(1 / np.sum(alpha**2) / len(alpha)),
                                  entropy_fraction=float(-np.sum(alpha[positive] * np.log(alpha[positive])) / np.log(len(alpha)))
                                  if len(alpha) > 1 else 1.,
                                  conservation_n=int(np.isfinite(conservation).sum()),
                                  conservation_spearman=safe_spearman(values["relative"], conservation),
                                  relative_sasa_n=int(np.isfinite(rsa).sum()),
                                  relative_sasa_spearman=safe_spearman(values["relative"], rsa),
                                  q_k_spearman=safe_spearman(gates[(sid, "q")]["relative"], gates[(sid, "k")]["relative"])))
    vector_rows = pq.read_table(directory / "vectors.parquet").to_pylist()
    vectors = {(r["sequence_id"], r["branch"], r["variant"]): r["vector"] for r in vector_rows}
    if len(vectors) != len(vector_rows):
        raise ValueError("Duplicate CLIP vectors")
    pairs = pq.read_table(local_path("data/pairs.parquet")).to_pylist()
    scores = pair_scores(pairs, vectors, config["gate_shuffles"])
    binary = interaction_report(scores)
    aggregate = aggregate_associations(config, associations)
    for filename, rows in (("interface_per_structure_side", associations), ("gate_distributions", distributions),
                           ("intervention_pair_scores", scores)):
        write_table(output / (filename + ".parquet"), rows)
    report = dict(checkpoint=name, panel=manifest["summary"], top_fraction=config["top_fraction"],
                  annotations=annotation_info, interface=aggregate, clip_interventions=binary,
                  extraction_sha256=sha256(directory / "provenance.json"),
                  analysis_source_sha256=sha256(__file__),
                  caveats=["Exploratory validation panel, not independent confirmatory test",
                           "Gate is partner-independent for a given full sequence and q/k role",
                           "Gate weight is not a contact probability, hotspot score or biological causal effect",
                           "Shared encoders also receive contact-loss supervision",
                           "Inference interventions do not estimate a retrained ungated model's performance",
                           "No RSA or conservation control unless explicit annotations supplied",
                           "Existing structural clusters/dependence blocks are not exhaustive homology exclusion"])
    write_json(output / "summary.json", report)
    lines = [f"# CLIP residue gate: {name}", "", "探索性固定 val 分析；尚不能据此声称门控具有因果生物学意义。", "",
             "界面沿用源数据的 Cβ <12 Å（Gly 使用 CA）；不重新计算接触阈值。", "",
             "| Branch | 蛋白等权 AP | 蛋白等权 Top 10% 富集 | AA 控制后超额富集 |", "|---|---:|---:|---:|"]
    for branch in ("q", "k"):
        metrics = aggregate[branch]["metrics"]
        vals = [metrics[key]["sequence_macro"] for key in ("ap", "top_enrichment", "aa_excess_enrichment")]
        lines.append(f"| {branch} | " + " | ".join("NA" if v is None else f"{v:.4f}" for v in vals) + " |")
    lines += ["", "summary.json 另含结构/同源依赖块等权均值与块 bootstrap 区间；残基不作为独立重复。", "",
              "| 池化干预 | CLIP AP | AUROC | 相对原门控 ΔAP |", "|---|---:|---:|---:|"]
    for key in ("learned", "uniform_q", "uniform_k", "uniform_both"):
        values = binary[key]
        lines.append(f"| {key} | {values['ap']:.4f} | {values['auroc']:.4f} | {values['ap_delta_from_learned']:+.4f} |")
    shuffled = [v["ap"] for k, v in binary.items() if k.startswith("shuffle_") and k.endswith("_both")]
    lines += ["", f"双分支打乱门控：{len(shuffled)} 次，AP 均值 {np.mean(shuffled):.4f}；各次结果在 summary.json。",
              "打乱间变异仅反映干预随机性，不是生物样本置信区间。",
              f"可用的有正有负候选查询：{binary['learned']['eligible_labeled_candidate_queries']}；不构造未标注负例。", "",
              "保守性/SASA 注释：" + annotation_info["status"] + "。未提供的注释不能解读为没有关联。", "",
              "完整指标、排除计数、注释覆盖和限制见 summary.json 及逐序列/逐结构 parquet。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    make_plot(output, distributions, aggregate, binary)
    print(str(output / "REPORT.md"))


def make_plot(output, distributions, aggregate, binary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for branch in ("q", "k"):
        axes[0].hist([r["effective_residue_fraction"] for r in distributions if r["branch"] == branch],
                     bins=np.linspace(0, 1, 21), alpha=.5, label=branch)
    axes[0].set(xlabel="Effective residue fraction (1 = uniform)", ylabel="Unique sequences")
    axes[0].legend()
    keys = ["learned", "uniform_q", "uniform_k", "uniform_both"]
    axes[1].bar(range(len(keys)), [binary[k]["ap"] for k in keys])
    axes[1].set_xticks(range(len(keys)), keys, rotation=25, ha="right")
    axes[1].set(ylabel="CLIP AP on fixed labeled validation pairs", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(local_path(output / "overview.png"), dpi=180)
    plt.close(fig)


def compare(config):
    before = read_json(local_path("results/before_1616/analysis/summary.json"))
    after = read_json(local_path("results/after_2182/analysis/summary.json"))
    if before["panel"] != after["panel"] or before["annotations"] != after["annotations"]:
        raise ValueError("Comparison requires the same panel and annotations")
    _, _, sequences, gates_before = load_result(config, "before_1616")
    _, _, _, gates_after = load_result(config, "after_2182")
    rank_rows = [dict(sequence_id=sid, branch=branch,
                      gate_spearman=safe_spearman(gates_before[(sid, branch)]["relative"],
                                                 gates_after[(sid, branch)]["relative"]))
                 for sid in sequences for branch in ("q", "k")]
    deltas = {branch: {metric: after["interface"][branch]["metrics"][metric]["sequence_macro"] -
                             before["interface"][branch]["metrics"][metric]["sequence_macro"]
                      for metric in ("ap", "top_enrichment", "aa_excess_enrichment")}
              for branch in ("q", "k")}
    write_table("results/comparison/gate_stability.parquet", rank_rows)
    write_json("results/comparison/summary.json", dict(
        before="before_1616", after="after_2182", interface_sequence_macro_deltas=deltas,
        learned_clip_ap_delta=after["clip_interventions"]["learned"]["ap"] - before["clip_interventions"]["learned"]["ap"],
        interpretation="Descriptive checkpoint comparison; added data, sampling, training steps and numerical changes are confounded"))
    print("Checkpoint comparison written inside resercher/results/comparison")
