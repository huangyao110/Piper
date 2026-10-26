"""Prepare only actual CongLab structures; retain frozen validation/test isolation.

Reuses the verified PPI val/E.coli split and DDI val/RF2 cluster exclusions.
Adds PPI->RF2 and DDI->E.coli homology filtering. Source data is read-only.
Temporary FASTA/MMseqs files are removed; final training parquet and a manifest
pin provenance, filtering, clusters and checksums for both comparison arms.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from flashppi.cropping import valid_contact_counts
from flashppi.ddi_common import rf2_rows, sha256, write_json


def sequence_dictionary(table):
    result = {}
    for side in (1, 2):
        for sid, sequence in zip(table[f"seq{side}_id"].to_pylist(), table[f"seq{side}"].to_pylist()):
            if sid in result and result[sid] != sequence:
                raise ValueError("Sequence ID maps to different residues")
            result[sid] = sequence
    return result


def check_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError("Frozen input checksum mismatch: " + str(path))
    return actual


def lookup(sequence_ids, sorted_ids, clusters):
    query = np.asarray(sequence_ids, dtype="S68")
    positions = np.searchsorted(sorted_ids, query)
    if (positions >= len(sorted_ids)).any() or not np.array_equal(sorted_ids[positions], query):
        raise ValueError("DDI sequence is absent from the verified cluster map")
    return clusters[positions]


def search(config, sequences, holdout_sequences, kind, scratch, output):
    query, target = scratch / f"{kind}.fasta", scratch / f"{kind}_holdout.fasta"
    with query.open("w") as handle:
        for sid, sequence in sorted(sequences.items()):
            handle.write(f">{sid}\n{sequence}\n")
    with target.open("w") as handle:
        for sequence in sorted(holdout_sequences):
            handle.write(f">holdout_{hashlib.sha256(sequence.encode()).hexdigest()}\n{sequence}\n")
    coverage = config["ppi_rf2_coverage"] if kind == "ppi_rf2" else config["ddi_ecoli_domain_coverage"]
    hits = scratch / f"{kind}.tsv"
    reverse = kind == "ddi_ecoli"
    # Hundreds of E.coli references querying an indexed domain library is much
    # faster than expanding k-mers for nearly a million domain queries. Export
    # the columns back into candidate-domain / held-out-reference order.
    search_query, search_target = (target, query) if reverse else (query, target)
    command = [config["mmseqs"], "easy-search", str(search_query), str(search_target), str(hits),
               str(scratch / f"{kind}_tmp"), "--min-seq-id", str(config["identity"]),
               "-c", str(coverage), "--cov-mode", "1" if reverse else "0",
               "--alignment-mode", "3", "--seq-id-mode", "0", "--max-accept", "2147483647",
               "--max-seqs", str(config["max_seqs"]), "-s", str(config["sensitivity"]),
               "-e", str(config[f"{kind}_evalue"]), "--threads", str(config["threads"]),
               "--split-memory-limit", "16G", "--format-output",
               "target,query,fident,tcov,qcov" if reverse else "query,target,fident,qcov,tcov"]
    started = time.monotonic()
    cached = output / f"{kind}_excluded_ids.json"
    cached_record = output / f"{kind}_homology.json"
    if cached.exists() and cached_record.exists():
        record = json.loads(cached_record.read_text())
        excluded = set(json.loads(cached.read_text()))
        if (record["query_sha256"] != sha256(query) or record["holdout_sha256"] != sha256(target)
                or record["command"][6:] != command[6:]
                or record["excluded_sequence_ids_sha256"] != hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()):
            raise ValueError("Homology cache inputs, parameters or contents changed")
        print("REUSED_VERIFIED_HOMOLOGY", kind, len(excluded), flush=True)
        return excluded, record
    print(f"HOMOLOGY {kind}: {len(sequences)} candidate sequences vs {len(holdout_sequences)} held-out sequences", flush=True)
    with (output / f"{kind}_mmseqs.log").open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    # Short exact matches can be missed by the k-mer prefilter. Exclude them
    # explicitly regardless of the heuristic alignment search outcome.
    excluded = {sid for sid, sequence in sequences.items() if sequence in holdout_sequences}
    exact_count, rows = len(excluded), 0
    with hits.open() as handle:
        for line in handle:
            sid, _, identity, qcov, tcov = line.rstrip().split("\t")
            # MMseqs applies exact thresholds internally; exported values are
            # rounded. Do not re-test a boundary using those rounded values.
            if float(identity) < config["identity"] - 0.002 or float(qcov) < coverage - 0.002:
                raise ValueError("Hit disagrees with configured identity/coverage")
            if kind == "ppi_rf2" and float(tcov) < coverage - 0.002:
                raise ValueError("PPI hit lacks bidirectional coverage")
            if sid not in sequences:
                raise ValueError("Unexpected MMseqs query ID")
            excluded.add(sid); rows += 1
    record = dict(command=command, query_sha256=sha256(query), holdout_sha256=sha256(target),
                  hits_sha256=sha256(hits), hit_rows=rows, hit_sequences=len(excluded),
                  exact_match_sequences=exact_count,
                  query_sequences=len(sequences), holdout_sequences=len(holdout_sequences),
                  seconds=time.monotonic() - started,
                  threshold_boundary="inclusive; enforced by MMseqs before rounded export",
                  excluded_sequence_ids_sha256=hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest())
    write_json(output / f"{kind}_homology.json", record)
    write_json(cached, sorted(excluded))
    print("HOMOLOGY_DONE", kind, rows, len(excluded), flush=True)
    return excluded, record


def joint_clusters(ppi, ddi, dc1, dc2):
    """Separate source cluster namespaces, merging shared exact sequence IDs.

    DDI clusters are used for known-positive masking, not PPI/DDI branch choice.
    A -1 sentinel for all DDI sequences would silently break false-negative masks.
    """
    pc1, pc2 = (ppi[f"seq{s}_cluster"].to_numpy() for s in (1, 2))
    pids = np.unique(np.r_[pc1, pc2]); dids = np.unique(np.r_[dc1, dc2])
    if (pids < 0).any() or (dids < 0).any():
        raise ValueError("Missing sequence cluster")
    pc1, pc2 = np.searchsorted(pids, pc1), np.searchsorted(pids, pc2)
    dc1, dc2 = np.searchsorted(dids, dc1) + len(pids), np.searchsorted(dids, dc2) + len(pids)
    parent = np.arange(len(pids) + len(dids), dtype=np.int64)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pmap = {}
    for side, clusters in ((1, pc1), (2, pc2)):
        for sid, cluster in zip(ppi[f"seq{side}_id"].to_pylist(), clusters):
            if sid in pmap and pmap[sid] != cluster:
                raise ValueError("PPI sequence has inconsistent cluster IDs")
            pmap[sid] = int(cluster)
    shared = set()
    for side, clusters in ((1, dc1), (2, dc2)):
        for sid, cluster in zip(ddi[f"seq{side}_id"].to_pylist(), clusters):
            if sid in pmap:
                shared.add(sid)
                a, b = root(pmap[sid]), root(int(cluster))
                parent[max(a, b)] = min(a, b)
    roots = np.array([root(i) for i in range(len(parent))])
    _, compact = np.unique(roots, return_inverse=True)
    return tuple(compact[x] for x in (pc1, pc2, dc1, dc2)), dict(
        ppi_input_clusters=len(pids), ddi_input_clusters=len(dids), joint_clusters=int(compact.max())+1,
        exact_sequences_shared_across_kinds=len(shared),
        policy="independent 70/70 clusters; union only through identical sequence IDs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs_data/conglab_mixture_v1.json"))
    parser.add_argument("--resume", action="store_true", help="Reuse checksum-verified completed homology audits")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output, release = Path(config["output"]), Path(config["release"])
    output.mkdir(parents=True, exist_ok=args.resume)
    if (output / "train.parquet").exists():
        raise FileExistsError("Refusing to overwrite a prepared training parquet")
    started = time.monotonic(); hashes = {}; stats = {}

    def status(stage):
        write_json(output / "status.json", dict(state="preparing", stage=stage, seconds=time.monotonic()-started))
        print(stage, flush=True)

    status("verify_prior_filters")
    split = json.loads((Path(config["filtered_ppi"]).parent / "manifest.json").read_text())
    prior_dir = Path(config["ddi_prior_filter"])
    prior = json.loads((prior_dir / "manifest.json").read_text())
    if split["state"] != "complete" or prior["state"] != "complete":
        raise ValueError("Prior split or DDI filter is incomplete")
    for key, old_key in (("filtered_ppi", "train.parquet"), ("validation_structures", "validation.parquet")):
        hashes[config[key]] = check_hash(config[key], split["outputs"][old_key]["sha256"])
    for key in ("validation_structures", "validation_interactions", "rf2_pairs", "rf2_sequences", "sequence_clusters"):
        hashes[config[key]] = check_hash(config[key], prior["input_hashes"][str(Path(config[key]).resolve())])
    hashes[config["test"]] = check_hash(config["test"], split["inputs"]["test_sha256"])
    hashes[str(prior_dir / "excluded_clusters.npy")] = check_hash(prior_dir / "excluded_clusters.npy", prior["output_hashes"]["excluded_clusters.npy"])
    for path in (args.config, release / "DATASET.json", prior_dir / "manifest.json", Path(config["filtered_ppi"]).parent / "manifest.json"):
        hashes[str(path)] = sha256(path)
    status("read_actual_conglab_structures")
    field, value = config["source_field"], config["source_value"]
    if field != "structure_source" or value != "conglab":
        raise ValueError("This experiment requires actual CongLab structures")
    ppi = pq.read_table(config["filtered_ppi"], filters=[(field, "=", value)])
    stats["ppi_after_prior_val_ecoli_filter"] = len(ppi)
    dataset = json.loads((release / "DATASET.json").read_text())
    parts = []
    entries = [x for x in dataset["parquet_shards"] if x["kind"] == "ddi"]
    for i, entry in enumerate(entries):
        path = release / entry["file"]
        hashes[str(path)] = check_hash(path, entry["sha256"])
        part = pq.read_table(path, filters=[(field, "=", value)])
        if len(part):
            parts.append(part)
        if (i+1) % 16 == 0:
            print("READ_DDI_SHARDS", i+1, len(entries), flush=True)
    ddi = pa.concat_tables(parts); del parts
    stats["ddi_actual_conglab_before_filter"] = len(ddi)
    status("join_verified_ddi_clusters")
    mapping = pacsv.read_csv(config["sequence_clusters"], parse_options=pacsv.ParseOptions(delimiter="\t"),
                            convert_options=pacsv.ConvertOptions(include_columns=["sequence_id", "cluster_id"]))
    ids = np.asarray(mapping["sequence_id"].to_numpy(), dtype="S68")
    clusters = mapping["cluster_id"].to_numpy(); del mapping
    if not np.all(ids[1:] > ids[:-1]):
        raise ValueError("Verified cluster mapping is not sorted and unique")
    dc1, dc2 = (lookup(ddi[f"seq{s}_id"].to_numpy(), ids, clusters) for s in (1, 2))
    excluded = np.load(prior_dir / "excluded_clusters.npy")
    keep = ~(np.isin(dc1, excluded) | np.isin(dc2, excluded))
    stats["ddi_prior_val_rf2_cluster_excluded"] = int((~keep).sum())
    ddi = ddi.filter(keep); dc1, dc2 = dc1[keep], dc2[keep]
    del ids, clusters, excluded
    status("additional_homology")
    rf2 = {s for r in rf2_rows(config) for s in (r["seq1"], r["seq2"])}
    ecoli_table = pq.read_table(config["test"], columns=["seq1", "seq2"])
    ecoli = set(ecoli_table["seq1"].to_pylist()) | set(ecoli_table["seq2"].to_pylist())
    with tempfile.TemporaryDirectory(prefix="scratch_", dir=output) as temporary:
        scratch = Path(temporary)
        hit_p, hp = search(config, sequence_dictionary(ppi), rf2, "ppi_rf2", scratch, output)
        keep = pc.invert(pc.or_(
            pc.is_in(ppi["seq1_id"], value_set=pa.array(sorted(hit_p), type=pa.string())),
            pc.is_in(ppi["seq2_id"], value_set=pa.array(sorted(hit_p), type=pa.string()))))
        stats["ppi_rf2_homolog_excluded"] = len(ppi) - int(pc.sum(keep).as_py())
        ppi = ppi.filter(keep)
        hit_d, hd = search(config, sequence_dictionary(ddi), ecoli, "ddi_ecoli", scratch, output)
        bad1 = pc.is_in(ddi["seq1_id"], value_set=pa.array(sorted(hit_d), type=pa.string())).to_numpy()
        bad2 = pc.is_in(ddi["seq2_id"], value_set=pa.array(sorted(hit_d), type=pa.string())).to_numpy()
        # Expand new domain matches to their existing 70/70 clusters.
        bad_clusters = np.unique(np.r_[dc1[bad1], dc2[bad2]])
        keep = ~(np.isin(dc1, bad_clusters) | np.isin(dc2, bad_clusters))
        stats["ddi_ecoli_homolog_cluster_excluded"] = int((~keep).sum())
        ddi = ddi.filter(keep); dc1, dc2 = dc1[keep], dc2[keep]
    status("valid_contacts_and_joint_cluster_ids")
    keep_p, keep_d = valid_contact_counts(ppi) > 0, valid_contact_counts(ddi) > 0
    stats["ppi_no_valid_contacts"] = int((~keep_p).sum()); stats["ddi_no_valid_contacts"] = int((~keep_d).sum())
    ppi, ddi = ppi.filter(keep_p), ddi.filter(keep_d)
    dc1, dc2 = dc1[keep_d], dc2[keep_d]
    (pc1, pc2, dc1, dc2), cluster_summary = joint_clusters(ppi, ddi, dc1, dc2)
    for name, left, right in (("ppi", pc1, pc2), ("ddi", dc1, dc2)):
        table = ppi if name == "ppi" else ddi
        for side, values in ((1, left), (2, right)):
            field_name = f"seq{side}_cluster"
            table = table.set_column(table.schema.get_field_index(field_name), field_name, pa.array(values, type=pa.int64()))
        if name == "ppi": ppi = table
        else: ddi = table
    stats.update(ppi_eligible=len(ppi), ddi_eligible=len(ddi), eligible=len(ppi)+len(ddi))
    train = pa.concat_tables([ppi, ddi])
    assert set(train["structure_source"].to_pylist()) == {"conglab"}
    assert len(set(train["pair_id"].to_pylist())) == len(train)
    combined_sequences = sequence_dictionary(train)
    heldout_sequences = set(rf2) | set(ecoli)
    for key in ("validation_structures", "validation_interactions"):
        t = pq.read_table(config[key], columns=["seq1", "seq2"])
        heldout_sequences.update(t["seq1"].to_pylist()); heldout_sequences.update(t["seq2"].to_pylist())
    exact_overlap = set(combined_sequences.values()) & heldout_sequences
    if exact_overlap:
        raise ValueError(f"Unexpected exact held-out sequence overlap: {len(exact_overlap)}")
    status("write_final_training_parquet")
    path = output / "train.parquet"
    pq.write_table(train, path, compression="zstd", row_group_size=1024)
    multiplier = max(1_000_000, cluster_summary["joint_clusters"])
    if multiplier**2 >= np.iinfo(np.int64).max:
        raise ValueError("Cluster key overflow")
    manifest = dict(state="complete", config=config, input_hashes=hashes,
                    output_hashes={str(path): sha256(path)},
                    outputs={"train.parquet": {"sha256": sha256(path), "rows": len(train)},
                             "validation.parquet": {"sha256": hashes[config["validation_structures"]],
                                                    "external_frozen_reference": config["validation_structures"]}},
                    stats=stats, clusters=cluster_summary, cluster_multiplier=multiplier,
                    source_selection="structure_source == conglab (not source contains conglab)",
                    exact_heldout_sequence_overlap=0,
                    ppi_existing_policy=split["policy"], ddi_existing_policy=prior["config"]["homology"],
                    additional_homology={"ppi_rf2": hp, "ddi_ecoli": hd},
                    elapsed_seconds=time.monotonic()-started,
                    script_sha256=sha256(__file__),
                    limitations=["CongLab provenance, not the exact published FlashPPI split or corpus",
                                 "Filtering fine-tuning data cannot remove pretrained checkpoint exposure",
                                 "MMseqs heuristic searches do not prove absence of remote homology",
                                 "Frozen val and RF2 benchmark share some proteins as previously audited"])
    write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", dict(state="complete", stats=stats, seconds=time.monotonic()-started))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
