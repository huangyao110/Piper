"""Within-protein metrics; tied ranks and dependent residues handled explicitly."""
import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def top_enrichment(labels, weights, fraction):
    labels, weights = np.asarray(labels), np.asarray(weights)
    if not 0 < fraction <= 1 or len(labels) == 0 or labels.sum() == 0:
        raise ValueError("Enrichment requires observations, positives and a valid top fraction")
    k = max(1, math.ceil(len(labels) * fraction))
    threshold = np.partition(weights, len(weights) - k)[len(weights) - k]
    high, tied = weights > threshold, weights == threshold
    # Expected number of positives under random ordering of cutoff ties.
    positives = labels[high].sum() + (k - high.sum()) * labels[tied].mean()
    return float((positives / k) / labels.mean())


def residue_metrics(labels, weights, fraction):
    labels, weights = np.asarray(labels), np.asarray(weights)
    keep = labels >= 0
    labels, weights = labels[keep], weights[keep]
    if len(np.unique(labels)) != 2:
        return None
    return dict(n_observed=int(len(labels)), interface_prevalence=float(labels.mean()),
                ap=float(average_precision_score(labels, weights)),
                auroc=float(roc_auc_score(labels, weights)),
                top_enrichment=top_enrichment(labels, weights, fraction))


def shuffled_within_groups(values, groups, rng):
    values, groups = np.asarray(values), np.asarray(groups)
    shuffled = values.copy()
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        shuffled[positions] = rng.permutation(values[positions])
    return shuffled


def safe_spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return None
    return float(spearmanr(a, b).statistic)


def bootstrap_mean(values, replicates, seed):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, ci95=None)
    mean = float(values.mean())
    if len(values) < 2:
        return dict(n=int(len(values)), mean=mean, ci95=None)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(replicates)])
    return dict(n=int(len(values)), mean=mean, ci95=np.quantile(means, [.025, .975]).tolist())


def pair_scores(pairs, vectors, shuffles):
    """Use original labeled row orientation. Never invent missing-pair negatives."""
    variants = [("learned", "learned", "learned"), ("uniform_q", "uniform", "learned"),
                ("uniform_k", "learned", "uniform"), ("uniform_both", "uniform", "uniform")]
    for replicate in range(shuffles):
        key = f"shuffle_{replicate:03d}"
        variants.extend([(key + "_q", key, "learned"), (key + "_k", "learned", key),
                         (key + "_both", key, key)])
    result = []
    for variant, q_variant, k_variant in variants:
        for row in pairs:
            q = vectors[(row["sequence1"], "q", q_variant)]
            k = vectors[(row["sequence2"], "k", k_variant)]
            score = float(np.sum(np.asarray(q, dtype=np.float32) * np.asarray(k, dtype=np.float32), dtype=np.float32))
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite pair score")
            result.append({**row, "variant": variant, "clip_score": score})
    return result
