"""Full-sequence evaluation for single/fused PLMs with CPU residue caching."""
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

from .training_utils import clear_rotary_cache, sha256
from .lora import merged_for_inference
from .patch_contact import PatchContactHead, PatchSpec, structural_patch_metrics


def classification_metrics(y, score, threshold):
    precision, recall, _ = precision_recall_curve(y, score)
    tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, score)),
        "auprc_trapezoidal": float(auc(recall, precision)),
        "auroc": float(roc_auc_score(y, score)),
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def score_rows(model, tokenizer, rows, output, encode_batch=1, pair_batch=16, clip_row_count=None,
               patch_spec=None, patch_on_labeled_positives=False, coverage_small_threshold=0.25):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    started = time.monotonic()
    sequences = sorted({r[k] for r in rows for k in ("seq1", "seq2")}, key=lambda s: (len(s), s))
    # Structural-only rows need contact predictions, not retrieval vectors.
    # In particular, very short structural peptides can overflow the FP16
    # retrieval MLP despite finite contact predictions. Do not fabricate a
    # retrieval score or change precision for an unused validation quantity.
    clip_row_count = len(rows) if clip_row_count is None else clip_row_count
    if not 0 <= clip_row_count <= len(rows):
        raise ValueError("Invalid number of rows requiring CLIP scores")
    clip_sequences = {r[k] for r in rows[:clip_row_count] for k in ("seq1", "seq2")}
    cache, vectors = {}, {}
    model.eval()
    clear_rotary_cache(model)
    result = []
    prediction_is_patch = isinstance(getattr(model, "contact_head", None), PatchContactHead)
    has_coverage = prediction_is_patch and model.contact_head.coverage_mix is not None
    if prediction_is_patch and (patch_spec is None or patch_spec != model.contact_head.spec):
        raise ValueError("Patch predictions require an identical evaluation window/stride")
    try:
        with merged_for_inference(model), context:
            for start in range(0, len(sequences), encode_batch):
                seqs = sequences[start:start + encode_batch]
                tokens = tokenizer(seqs, padding=True, truncation=False, return_tensors="pt").to(device)
                if tokens.attention_mask.sum(1).tolist() != list(map(len, seqs)):
                    raise ValueError("Residue/token alignment mismatch")
                encoded = model.encode_protein(tokens.input_ids, tokens.attention_mask)
                needed = [j for j, sequence in enumerate(seqs) if sequence in clip_sequences]
                if needed:
                    q = model.head_q(encoded[needed], tokens.attention_mask[needed]).float()
                    k = model.head_k(encoded[needed], tokens.attention_mask[needed]).float()
                    for j, source_index in enumerate(needed):
                        vectors[seqs[source_index]] = (q[j].cpu().clone(), k[j].cpu().clone())
                    del q, k
                for j, sequence in enumerate(seqs):
                    cache[sequence] = encoded[j, :len(sequence)].cpu().clone()
                del encoded
            order = sorted(range(len(rows)), key=lambda i: (len(rows[i]["seq1"]) // 64,
                                                          len(rows[i]["seq2"]) // 64))
            cursor = 0
            while cursor < len(order):
                indices, max_a, max_b = [], 0, 0
                for i in order[cursor:cursor + pair_batch]:
                    na, nb = max(max_a, len(rows[i]["seq1"])), max(max_b, len(rows[i]["seq2"]))
                    if indices and (len(indices) + 1) * (na + nb)**2 > 4_000_000:
                        break
                    indices.append(i)
                    max_a, max_b = na, nb
                left = [cache[rows[i]["seq1"]] for i in indices]
                right = [cache[rows[i]["seq2"]] for i in indices]
                a, b = pad_sequence(left, batch_first=True).to(device), pad_sequence(right, batch_first=True).to(device)
                ma = torch.arange(a.shape[1], device=device)[None] < torch.tensor([len(x) for x in left], device=device)[:, None]
                mb = torch.arange(b.shape[1], device=device)[None] < torch.tensor([len(x) for x in right], device=device)[:, None]
                coverage_logits = None
                if has_coverage:
                    logits, valid, coverage_logits = model.contact_head.forward_with_coverage(a, b, ma, mb)
                else:
                    logits, valid = model.predict_contacts(a, b, ma, mb)
                logits = logits.float().masked_fill(~valid, -torch.inf)
                maximum = logits.flatten(1).max(1).values
                scores = maximum.sigmoid().tolist()
                clip = {}
                needed = [i for i in indices if i < clip_row_count]
                if needed:
                    q = torch.stack([vectors[rows[i]["seq1"]][0] for i in needed]).to(device)
                    k = torch.stack([vectors[rows[i]["seq2"]][1] for i in needed]).to(device)
                    clip = dict(zip(needed, (q * k).sum(-1).tolist()))
                for j, i in enumerate(indices):
                    row = rows[i]
                    n, m = len(row["seq1"]), len(row["seq2"])
                    item = dict(row_index=i, label=int(row.get("label", 1)),
                                pair_id=row.get("pair_id", ""), taxid=row.get("taxid", ""),
                                length1=n, length2=m, contact_score=scores[j], clip_score=clip.get(i),
                                max_contact_logit=float(maximum[j]),
                                contacts_precision_at_k=None, contacts_8A_precision_at_k=None)
                    if patch_spec is not None:
                        item.update(patch_precision_at_k=None, patch_average_precision=None,
                                    patch_positive_count=None, patch_valid_count=None,
                                    patch_total_count=None, patch_random_ap=None)
                        if has_coverage:
                            item.update(coverage_positive_patches=None, coverage_mae=None, coverage_mae_a=None,
                                        coverage_mae_b=None, coverage_small_positive_patches=None,
                                        coverage_small_mae=None, small_patch_existence_recall_at_half=None,
                                        coverage_small_threshold=coverage_small_threshold)
                        if (i >= clip_row_count or patch_on_labeled_positives) and item["label"] == 1:
                            item.update(structural_patch_metrics(logits[j], row, patch_spec, prediction_is_patch,
                                        coverage_logits[j] if coverage_logits is not None else None,
                                        coverage_small_threshold=coverage_small_threshold))
                    va = row.get("valid1", [True] * n)
                    vb = row.get("valid2", [True] * m)
                    if item["label"] == 1 and not prediction_is_patch:
                        structural = torch.tensor(va, device=device)[:, None] & torch.tensor(vb, device=device)[None, :]
                        for column in ("contacts", "contacts_8A"):
                            coords = {(int(x), int(y)) for x, y in row.get(column, [])
                                      if 0 <= x < n and 0 <= y < m and va[x] and vb[y]}
                            if coords:
                                flat = logits[j, :n, :m].masked_fill(~structural, -torch.inf).flatten()
                                top = flat.topk(len(coords)).indices.tolist()
                                item[column + "_precision_at_k"] = sum((x // m, x % m) in coords for x in top) / len(coords)
                    result.append(item)
                cursor += len(indices)
                atomic_json(output / "status.json", dict(status="running", completed=cursor, total=len(rows)))
                del logits, coverage_logits, a, b
        result.sort(key=lambda x: x["row_index"])
        required = [value for r in result for value in
                    (r["contact_score"], r["clip_score"], r["max_contact_logit"]) if value is not None]
        if not np.isfinite(required).all():
            raise FloatingPointError("Nonfinite evaluation scores")
        with (output / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
        atomic_json(output / "status.json", dict(status="complete", completed=len(rows), total=len(rows),
                                                  seconds=time.monotonic() - started))
    finally:
        clear_rotary_cache(model)
    return result


def interaction_metrics(rows):
    labels = np.array([r["label"] for r in rows])
    scores = np.array([r["contact_score"] for r in rows])
    clip = np.array([r["clip_score"] for r in rows])
    return dict(interaction_rows=len(rows), positive=int(labels.sum()),
                negative=int((labels == 0).sum()),
                contact_score_ap=float(average_precision_score(labels, scores)),
                contact_score_auroc=float(roc_auc_score(labels, scores)),
                clip_ap=float(average_precision_score(labels, clip)))


def coverage_metrics(rows):
    """Macro averages over structures with fully observed positive patches."""
    eligible = [r for r in rows if r.get("coverage_mae") is not None]
    result = dict(coverage_evaluable_structures=len(eligible),
                  coverage_positive_patches=sum(r["coverage_positive_patches"] for r in eligible),
                  coverage_small_positive_patches=sum(r["coverage_small_positive_patches"] for r in eligible))
    for key in ("coverage_mae", "coverage_mae_a", "coverage_mae_b", "coverage_small_mae",
                "small_patch_existence_recall_at_half"):
        values = [r[key] for r in eligible if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def evaluate_validation(model, tokenizer, config, output):
    binary = pq.read_table(config["validation_interactions"]).to_pylist()
    structures = pq.read_table(config["validation_structures"]).to_pylist()
    settings = config.get("patch_evaluation")
    spec = PatchSpec(**settings) if settings else None
    results = score_rows(model, tokenizer, binary + structures, output, clip_row_count=len(binary),
                        pair_batch=config.get("eval_pair_batch", 16), patch_spec=spec,
                        coverage_small_threshold=config.get('coverage_evaluation', {}).get('small_threshold', 0.25))
    metrics = interaction_metrics(results[:len(binary)])
    values = [r["contacts_precision_at_k"] for r in results[len(binary):]
              if r["contacts_precision_at_k"] is not None]
    metrics.update(contact_structures=len(structures), contact_evaluable=len(values),
                   contact_precision_at_k=float(np.mean(values)) if values else None,
                   interaction_sha256=sha256(config["validation_interactions"]),
                   structures_sha256=sha256(config["validation_structures"]),
                   evaluation="full length; FP32 weights/FP16 autocast; original orientation; sigmoid(max valid logit)")
    if spec is not None:
        evaluated = [r for r in results[len(binary):] if r["patch_average_precision"] is not None]
        for key in ("patch_average_precision", "patch_precision_at_k", "patch_random_ap"):
            metrics[key] = float(np.mean([r[key] for r in evaluated])) if evaluated else None
        metrics.update(patch_evaluable=len(evaluated), patch_window=spec.window, patch_stride=spec.stride,
                       contact_resolution="patch" if config.get("contact_patch") else "residue",
                       patch_label="any observed contact; zero supervised only if all real residue pairs observed",
                       patch_aggregation="per-structure macro; overlapping patches are not independent samples")
    if config.get("contact_coverage"):
        metrics.update(coverage_metrics(results[len(binary):]))
        metrics['coverage_small_threshold'] = config.get('coverage_evaluation', {}).get('small_threshold', 0.25)
    atomic_json(Path(output) / "metrics.json", metrics)
    return metrics


def evaluate_test(model, tokenizer, config, output):
    """Full fixed E. coli benchmark, shared by CLI and scheduled monitoring.

    Uses the historical scoring convention without cropping, symmetrization,
    retrieval prefiltering, or using test labels to build training negatives.
    """
    rows = pq.read_table(config['test']).to_pylist()
    settings = config.get('patch_evaluation')
    spec = PatchSpec(**settings) if settings else None
    results = score_rows(model, tokenizer, rows, output, clip_row_count=len(rows),
                         pair_batch=config.get('eval_pair_batch', 16), patch_spec=spec,
                         patch_on_labeled_positives=True,
                         coverage_small_threshold=config.get('coverage_evaluation', {}).get('small_threshold', 0.25))
    metrics = interaction_metrics(results)
    labels = np.array([r['label'] for r in results])
    metrics.update(rows=len(results), random_ap_baseline=float(labels.mean()),
                   contact=classification_metrics(labels, np.array([r['contact_score'] for r in results]), 0.5),
                   clip=classification_metrics(labels, np.array([r['clip_score'] for r in results]), 0.0))
    for key in ('contacts', 'contacts_8A'):
        values = [r[key + '_precision_at_k'] for r in results if r[key + '_precision_at_k'] is not None]
        metrics[key] = dict(evaluable_positives=len(values),
                            mean_precision_at_k=float(np.mean(values)) if values else None)
    if spec:
        patch = [r for r in results if r['patch_average_precision'] is not None]
        metrics['patch'] = dict(window=spec.window, stride=spec.stride, evaluable_positives=len(patch),
            average_precision=float(np.mean([r['patch_average_precision'] for r in patch])) if patch else None,
            precision_at_k=float(np.mean([r['patch_precision_at_k'] for r in patch])) if patch else None,
            resolution='patch' if config.get('contact_patch') else 'residue pooled to patch grid')
    if config.get('contact_coverage'):
        metrics.update(coverage_metrics(results))
        metrics['coverage_small_threshold'] = config.get('coverage_evaluation', {}).get('small_threshold', 0.25)
    atomic_json(Path(output) / 'metrics.json', metrics)
    return metrics
