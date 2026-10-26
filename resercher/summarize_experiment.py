#!/usr/bin/env python3
"""Audit fixed results and summarize the completed exploratory gate experiment."""
import csv
from collections import defaultdict
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from clip_gate.common import ROOT, REPOSITORY, read_json, setup_environment, sha256, stable_seed, write_json
setup_environment()

import numpy as np
import pyarrow.parquet as pq
from clip_gate.analysis import load_result
from clip_gate.statistics import bootstrap_mean


def block_values(rows, branch, metric):
    sequences, membership = defaultdict(list), {}
    for row in rows:
        if row["branch"] == branch and row["eligible"]:
            sequences[row["sequence_id"]].append(row[metric])
            membership[row["sequence_id"]] = row["block_id"]
    blocks = defaultdict(list)
    for sid, values in sequences.items():
        blocks[membership[sid]].append(float(np.mean(values)))
    return {key: float(np.mean(values)) for key, values in blocks.items()}


def main():
    config = read_json(ROOT / "study.json")
    summaries, association, distributions, shuffle_summary = {}, {}, {}, {}
    source_hashes = {}
    for alias in ("before_1616", "after_2182"):
        load_result(config, alias)
        directory = ROOT / "results" / alias / "analysis"
        summaries[alias] = read_json(directory / "summary.json")
        association[alias] = pq.read_table(directory / "interface_per_structure_side.parquet").to_pylist()
        table = pq.read_table(directory / "gate_distributions.parquet").to_pylist()
        distributions[alias] = {branch: {metric: float(np.mean([row[metric] for row in table if row["branch"] == branch]))
            for metric in ("effective_residue_fraction", "entropy_fraction", "relative_q10", "relative_q90", "q_k_spearman")}
            for branch in ("q", "k")}
        shuffled = summaries[alias]["clip_interventions"]
        shuffle_summary[alias] = {}
        for branch in ("q", "k", "both"):
            rows = [v for k, v in shuffled.items() if k.startswith("shuffle_") and k.endswith("_" + branch)]
            shuffle_summary[alias][branch] = {metric: dict(mean=float(np.mean([r[metric] for r in rows])),
                min=float(min(r[metric] for r in rows)), max=float(max(r[metric] for r in rows))) for metric in ("ap", "auroc")}
        for filename in ("summary.json", "interface_per_structure_side.parquet", "gate_distributions.parquet",
                         "intervention_pair_scores.parquet"):
            path = directory / filename
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)

    old_run = REPOSITORY / config["checkpoints"]["before_1616"]
    reference = old_run.parent.parent / "val_step_001616/predictions.csv"
    with reference.open() as handle:
        old_scores = list(csv.DictReader(handle))[:228]
    table = pq.read_table(ROOT / "results/before_1616/analysis/intervention_pair_scores.parquet").to_pylist()
    current = sorted([row for row in table if row["variant"] == "learned"], key=lambda row: row["row_index"])
    if len(old_scores) != len(current) or any(a["pair_id"] != b["pair_id"] or int(a["label"]) != b["label"]
                                               for a, b in zip(old_scores, current)):
        raise ValueError("Historical reference row mapping differs")
    differences = [abs(float(a["clip_score"]) - b["clip_score"]) for a, b in zip(old_scores, current)]
    if max(differences) > 1e-6:
        raise ValueError("Baseline score replay differs from historical validation")
    audit = dict(reference=str(reference), sha256=sha256(reference), rows=len(current),
                 max_absolute_score_difference=max(differences), tolerance=1e-6,
                 learned_ap_equal=summaries["before_1616"]["clip_interventions"]["learned"]["ap"] ==
                 read_json(reference.parent / "metrics.json")["clip_ap"])

    paired = {}
    for branch in ("q", "k"):
        paired[branch] = {}
        for metric in ("ap", "auroc", "top_enrichment", "aa_excess_enrichment"):
            before = block_values(association["before_1616"], branch, metric)
            after = block_values(association["after_2182"], branch, metric)
            if set(before) != set(after):
                raise ValueError("Paired dependence blocks differ")
            delta = [after[key] - before[key] for key in sorted(before)]
            paired[branch][metric] = bootstrap_mean(delta, config["bootstrap_replicates"],
                                                    stable_seed(config["seed"], branch, metric, "paired_checkpoint_delta"))

    control = ROOT / "control/experiment_20261023"
    plan = read_json(control / "resume_plan.json")
    output, original = Path(plan["output"]), Path(plan["checkpoint"]).parent.parent
    resume = read_json(output / "resume.json")
    invariant_files = ("config.json", "input_hashes.json", "source_sha256.json", "sample_schedule.json", "trainable_parameters.json")
    invariants = {name: read_json(original / name) == read_json(output / name) for name in invariant_files}
    if not all(invariants.values()) or not resume["optimizer_restored"] or not resume["rng_restored"] or resume["warmup_restarted"]:
        raise ValueError("Training continuation audit failed")
    state = read_json(output / "status.json") if (output / "status.json").exists() else {}
    continuation = dict(saved_step=plan["step"], source=resume["source"], current_step=state.get("step"),
                        first_new_step_observed=state.get("step", 0) > plan["step"], config_changes=resume["config_changes"],
                        optimizer_restored=True, rng_restored=True, warmup_restarted=False, invariant_files=invariants)
    combined = dict(baseline_replay=audit, distributions=distributions, gate_shuffle_summary=shuffle_summary,
                    paired_checkpoint_change=paired, training_continuation=continuation, source_hashes=source_hashes,
                    note="Paired checkpoint intervals are exploratory descriptive comparisons; no multiple-testing correction or causal data attribution")
    write_json("results/comparison/experiment_audit.json", combined)
    print(__import__("json").dumps(combined, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    aliases = ("before_1616", "after_2182")
    colors = ("#7488a5", "#246c92")
    for offset, (alias, color) in enumerate(zip(aliases, colors)):
        values = summaries[alias]["clip_interventions"]
        x = np.arange(3) + (offset - .5) * .35
        ys = [values["learned"]["ap"], values["uniform_both"]["ap"], shuffle_summary[alias]["both"]["ap"]["mean"]]
        axes[0].bar(x, ys, width=.35, color=color, label=alias.split("_")[1])
        for xx, yy in zip(x, ys):
            axes[0].text(xx, yy + .012, f"{yy:.3f}", ha="center", fontsize=9)
        for branch_id, branch in enumerate(("q", "k")):
            value = summaries[alias]["interface"][branch]["metrics"]["top_enrichment"]["block_macro"]
            mean, (low, high) = value["mean"], value["ci95"]
            axes[1].errorbar(branch_id + (offset - .5) * .18, mean,
                yerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=5, label=alias.split("_")[1] if branch_id == 0 else None)
    axes[0].set_xticks(range(3), ["Learned gate", "Uniform q+k", "Shuffle q+k\n20 repeats"])
    axes[0].set(ylabel="CLIP AP on fixed validation pairs", ylim=(0, .85))
    axes[0].legend(title="Checkpoint step", frameon=False)
    axes[1].axhline(1, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xticks([0, 1], ["q gate", "k gate"])
    axes[1].set(xlim=(-.5, 1.5), ylim=(.9, 1.45), ylabel="Top 10% interface enrichment (1 = chance)")
    axes[1].legend(title="Checkpoint step", frameon=False)
    fig.suptitle("Exploratory CLIP gate study: 100 structures, 228 labeled interactions", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "results/comparison/experiment_overview.png", dpi=180)
    fig.savefig(ROOT / "results/comparison/experiment_overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
