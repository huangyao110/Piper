"""Evaluate the released E. coli pairs using the official held-out checkpoint.

Run from the repository root. Downloaded model code and weights remain unchanged.
All pairs are scored in dataset orientation, without retrieval filtering.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, auc, confusion_matrix)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/flashppi_official/flashppi_noecoli")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/flashppi_official/ecoli_pdb_benchmark/data/train-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/official_ecoli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=0, help="0 keeps complete sequences")
    parser.add_argument("--limit", type=int, default=0, help="Smoke check only; 0 evaluates all rows")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires a CUDA GPU")
    started = time.time()
    rows = pq.read_table(args.dataset).to_pylist()
    if args.limit:
        # Retain both classes in smoke runs; official data are positive-first.
        half = args.limit // 2
        selected = [i for i, r in enumerate(rows) if r["label"] == 1][:half]
        selected += [i for i, r in enumerate(rows) if r["label"] == 0][:args.limit - half]
    else:
        selected = list(range(len(rows)))
    sequences = sorted({rows[i][key] for i in selected for key in ("seq1", "seq2")}, key=lambda s: (len(s), s))
    seq_index = {s: i for i, s in enumerate(sequences)}
    manifest = {
        "model": str(args.model.resolve()), "model_sha256": sha256(args.model / "model.safetensors"),
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "rows": len(selected), "positive": sum(rows[i]["label"] for i in selected),
        "negative": sum(1 - rows[i]["label"] for i in selected),
        "unique_sequences": len(sequences), "max_length": args.max_length or "full",
        "sequences_truncated": sum(len(s) > args.max_length for s in sequences) if args.max_length else 0,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "autocast": "float16", "weights": "float32",
        "orientation": "seq1 -> seq2 as published; no symmetrization",
        "contact_score": "sigmoid(max(valid residue-pair logits)); float32 sigmoid",
        "filtering": "none; every provided positive and negative pair is scored",
        "attention": "official PyTorch SDPA fallback; no flash_attn package",
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Output directory contains a different evaluation; choose a new output")
    write_json(manifest_path, manifest)
    print("MANIFEST", json.dumps(manifest), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True).cuda().eval()
    embeddings, query, key = [], [], []
    cache_path = args.output / "embeddings.pt"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if cache_path.exists():
            saved = torch.load(cache_path, map_location="cuda", weights_only=True)
            embeddings, query, key = saved["embeddings"], saved["query"], saved["key"]
            assert saved["sequences"] == sequences
            print("Loaded cached sequence embeddings", flush=True)
        else:
            for start in range(0, len(sequences), 8):
                seqs = sequences[start:start + 8]
                options = {"truncation": True, "max_length": args.max_length} if args.max_length else {"truncation": False}
                tokens = tokenizer(seqs, return_tensors="pt", padding=True, **options).to("cuda")
                lengths = tokens["attention_mask"].sum(1).tolist()
                assert lengths == [min(len(s), args.max_length) if args.max_length else len(s) for s in seqs], "Unexpected special tokens or sequence/token mismatch"
                encoded = model.encode_protein(tokens["input_ids"], tokens["attention_mask"])
                q = model.head_q(encoded, tokens["attention_mask"]).float()
                k = model.head_k(encoded, tokens["attention_mask"]).float()
                for j, length in enumerate(lengths):
                    embeddings.append(encoded[j, :int(length)].clone())
                    query.append(q[j].clone())
                    key.append(k[j].clone())
                if start % 80 == 0:
                    print(f"ENCODE {min(start + 8, len(sequences))}/{len(sequences)} elapsed={time.time()-started:.1f}s", flush=True)
            query, key = torch.stack(query), torch.stack(key)
            torch.save({"sequences": sequences, "embeddings": [e.cpu() for e in embeddings],
                        "query": query.cpu(), "key": key.cpu()}, cache_path)

        def score_batch(indices):
            left = [embeddings[seq_index[rows[i]["seq1"]]] for i in indices]
            right = [embeddings[seq_index[rows[i]["seq2"]]] for i in indices]
            a, b = pad_sequence(left, batch_first=True), pad_sequence(right, batch_first=True)
            ma = torch.arange(a.shape[1], device="cuda")[None, :] < torch.tensor([len(x) for x in left], device="cuda")[:, None]
            mb = torch.arange(b.shape[1], device="cuda")[None, :] < torch.tensor([len(x) for x in right], device="cuda")[:, None]
            logits, valid = model.predict_contacts(a, b, ma.long(), mb.long())
            return logits.float().masked_fill(~valid, -torch.inf)

        first = selected[0]
        longest = max(selected, key=lambda i: len(rows[i]["seq1"]) + len(rows[i]["seq2"]))
        alone = score_batch([first])[0]
        mixed = score_batch([first, longest])[0, :alone.shape[0], :alone.shape[1]]
        delta = abs(torch.sigmoid(alone.max()).item() - torch.sigmoid(mixed.max()).item())
        options = {"truncation": True, "max_length": args.max_length} if args.max_length else {"truncation": False}
        ta = tokenizer(rows[first]["seq1"], return_tensors="pt", **options).to("cuda")
        tb = tokenizer(rows[first]["seq2"], return_tensors="pt", **options).to("cuda")
        direct = model(input_ids1=ta.input_ids, input_ids2=tb.input_ids,
                       attention_mask1=ta.attention_mask, attention_mask2=tb.attention_mask).contact_score.item()
        direct_delta = abs(direct - torch.sigmoid(alone.max()).item())
        checks = {"batch_score_delta": delta, "cached_vs_forward_delta": direct_delta, "tolerance": 0.002}
        write_json(args.output / "sanity_checks.json", checks)
        print("SANITY", checks, flush=True)
        assert delta < 0.002 and direct_delta < 0.002, "Scoring consistency check failed"

        csv_path = args.output / "predictions.csv"
        done = set()
        if csv_path.exists():
            with csv_path.open() as f:
                done = {int(r["row_index"]) for r in csv.DictReader(f)}
        todo = [i for i in selected if i not in done]
        todo.sort(key=lambda i: (len(rows[i]["seq1"]) // 64, len(rows[i]["seq2"]) // 64))
        fields = ["row_index", "id1", "id2", "label", "length1", "length2", "clip_score", "max_contact_logit", "contact_score", "contacts_precision_at_k", "contacts_8A_precision_at_k"]
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not done:
                writer.writeheader()
            cursor, last_log = 0, 0
            while cursor < len(todo):
                indices = []
                max_a = max_b = 0
                for i in todo[cursor:cursor + args.batch_size]:
                    la = len(embeddings[seq_index[rows[i]["seq1"]]])
                    lb = len(embeddings[seq_index[rows[i]["seq2"]]])
                    na, nb = max(max_a, la), max(max_b, lb)
                    count = len(indices) + 1
                    if indices and (count * (na + nb) ** 2 > 16_000_000 or count * (na + nb) > 16000):
                        break
                    indices.append(i)
                    max_a, max_b = na, nb
                logits = score_batch(indices)
                maximum = logits.flatten(1).max(1).values
                scores = torch.sigmoid(maximum).cpu().tolist()
                max_values = maximum.cpu().tolist()
                for j, i in enumerate(indices):
                    row = rows[i]
                    a_idx, b_idx = seq_index[row["seq1"]], seq_index[row["seq2"]]
                    la, lb = len(embeddings[a_idx]), len(embeddings[b_idx])
                    result = dict(row_index=i, id1=row["id1"], id2=row["id2"], label=row["label"],
                                  length1=la, length2=lb, clip_score=float((query[a_idx] * key[b_idx]).sum()),
                                  max_contact_logit=max_values[j], contact_score=scores[j])
                    for column in ("contacts", "contacts_8A"):
                        precision = ""
                        if row["label"] == 1:
                            coordinates = {(int(a), int(b)) for a, b in row[column] if 0 <= a < la and 0 <= b < lb}
                            if coordinates:
                                flat = logits[j, :la, :lb].reshape(-1)
                                top = flat.topk(len(coordinates)).indices.cpu().tolist()
                                precision = sum((index // lb, index % lb) in coordinates for index in top) / len(coordinates)
                        result[column + "_precision_at_k"] = precision
                    writer.writerow(result)
                cursor += len(indices)
                if time.time() - last_log > 20 or cursor == len(todo):
                    handle.flush()
                    progress = {"status": "running", "completed": len(done) + cursor, "total": len(selected), "elapsed_seconds": time.time() - started}
                    write_json(args.output / "status.json", progress)
                    print("PROGRESS", json.dumps(progress), flush=True)
                    last_log = time.time()
                del logits

    import pandas as pd
    df = pd.read_csv(csv_path).sort_values("row_index")
    assert len(df) == len(selected) and df.row_index.nunique() == len(selected)
    assert np.isfinite(df[["contact_score", "clip_score", "max_contact_logit"]].to_numpy()).all()
    df.to_csv(csv_path, index=False)
    y = df.label.to_numpy()
    results = {"rows": len(df), "positive": int(y.sum()), "negative": int((y == 0).sum()),
               "random_ap_baseline": float(y.mean()), "contact": classification_metrics(y, df.contact_score, 0.5),
               "clip": classification_metrics(y, df.clip_score, 0.0),
               "contact_logit_ap": float(average_precision_score(y, df.max_contact_logit)),
               "elapsed_seconds_this_invocation": time.time() - started,
               "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
    for column in ("contacts", "contacts_8A"):
        values = df[column + "_precision_at_k"].dropna()
        results[column] = {"evaluable_positives": len(values), "mean_precision_at_k": float(values.mean()), "median_precision_at_k": float(values.median())}
    write_json(args.output / "metrics.json", results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for col, label in [("contact_score", "Contact head"), ("clip_score", "Contrastive head")]:
        precision, recall, _ = precision_recall_curve(y, df[col])
        ap = average_precision_score(y, df[col])
        ax.plot(recall, precision, label=f"{label} (AP={ap:.4f})")
    ax.axhline(y.mean(), color="gray", linestyle="--", label=f"Random ({y.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="FlashPPI: official held-out E. coli benchmark", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "precision_recall.png", dpi=180)
    write_json(args.output / "status.json", {"status": "complete", "completed": len(df), "total": len(selected)})
    print("RESULTS", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
