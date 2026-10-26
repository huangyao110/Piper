"""Small, dependency-light helpers for the independent DDI experiment."""
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def rf2_rows(config):
    path = Path(config["rf2_pairs"])
    if sha256(path) != "36596120a89a6fb714cbee272c2f85f87aca52d6e3cca54504bf3abd8b59a1c0":
        raise ValueError("RF2-PPI original test identity changed")
    mapping = {}
    with open(config["rf2_sequences"]) as handle:
        for row in csv.DictReader(handle):
            seq = row["sequence"]
            if (not seq or len(seq) != int(row["length"])
                    or hashlib.md5(seq.encode()).hexdigest() != row["sequence_md5"]
                    or row["accession"] in mapping):
                raise ValueError("Invalid RF2 sequence cache")
            mapping[row["accession"]] = seq
    with path.open() as handle:
        rows = []
        for row in csv.DictReader(handle, delimiter="\t"):
            a, b = row["Protein pairs"].split("_")
            rows.append(dict(pair_id=row["Protein pairs"], seq1=mapping[a], seq2=mapping[b],
                             label={"positive": 1, "negative": 0}[row["Category"]]))
    if len(rows) != 33000 or sum(r["label"] for r in rows) != 3000:
        raise ValueError("RF2 test counts changed")
    return rows
