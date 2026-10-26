"""Read frozen validation members without redefining any data split or contact rule."""
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import (ROOT, REPOSITORY, check_hash, local_path, read_json,
                     sequence_id, sha256, write_json)


def interface_labels(row):
    """Return -1 unknown / 0 observed noncontact / 1 observed interface.

    Missing partner coordinates cannot establish a negative. Positive evidence
    remains valid if the contacting residue pair is observed. Indices are source
    zero-based; reported residue positions elsewhere are one-based.
    """
    lengths = [len(row["seq1"]), len(row["seq2"])]
    masks = []
    for side, length in enumerate(lengths, 1):
        valid = np.asarray(row[f"valid{side}"], dtype=bool)
        if valid.shape != (length,):
            raise ValueError("Invalid source residue mask")
        masks.append(valid)
    labels = [np.full(n, -1, dtype=np.int8) for n in lengths]
    for side in (0, 1):
        if masks[1 - side].all():
            labels[side][masks[side]] = 0
    for contact in row["contacts"]:
        if len(contact) != 2 or any(not isinstance(x, (int, np.integer)) for x in contact):
            raise ValueError("Contact indices must be integer residue pairs")
        i, j = contact
        if not (0 <= i < lengths[0] and 0 <= j < lengths[1]):
            raise ValueError("Source contact index is out of bounds")
        if not (masks[0][i] and masks[1][j]):
            raise ValueError("Source contact refers to an unobserved residue")
        labels[0][i] = labels[1][j] = 1
    return [x.tolist() for x in labels]


def make_panel(structures, interactions):
    sequences, interfaces, pairs = {}, [], []
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    def record(sequence, role):
        if not sequence:
            raise ValueError("Empty sequence")
        sid = sequence_id(sequence)
        item = sequences.setdefault(sid, dict(sequence_id=sid, sequence=sequence, length=len(sequence),
                                              for_interface=False, for_interaction=False))
        item[role] = True
        find(sid)
        return sid

    family_members = {}
    for index, row in enumerate(structures):
        labels = interface_labels(row)
        ids = [record(row[f"seq{s}"], "for_interface") for s in (1, 2)]
        # A dependence block joins exact sequences, recorded homology clusters,
        # and both partners from a structure. No new clustering/search is run.
        union(*ids)
        for side, sid in enumerate(ids, 1):
            cluster = row[f"seq{side}_cluster"]
            if cluster is None:
                raise ValueError("The fixed structural panel must retain cluster IDs")
            family = str(cluster)
            if family in family_members:
                union(sid, family_members[family])
            family_members[family] = sid
            interfaces.append(dict(structure_index=index, pair_id=row["pair_id"], side=side,
                                   sequence_id=sid, partner_id=ids[2 - side],
                                   cluster_id=family, labels=labels[side - 1]))
    for index, row in enumerate(interactions):
        if row["label"] not in (0, 1):
            raise ValueError("Interaction label must be 0 or 1")
        ids = [record(row[f"seq{s}"], "for_interaction") for s in (1, 2)]
        pairs.append(dict(row_index=index, pair_id=row["pair_id"], sequence1=ids[0], sequence2=ids[1],
                          label=int(row["label"])))
    for item in interfaces:
        item["block_id"] = find(item["sequence_id"])
    return sorted(sequences.values(), key=lambda x: (x["length"], x["sequence_id"])), interfaces, pairs


def write_table(relative, rows):
    path = local_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Arrow otherwise takes field names only from the first dictionary, which
    # could be an excluded record lacking optional metrics.
    names = sorted({key for row in rows for key in row})
    table = pa.Table.from_pydict({key: [row.get(key) for row in rows] for key in names})
    pq.write_table(table, path, compression="zstd")
    return path


def prepare(config):
    if local_path("manifests/prepared.json").exists():
        from .common import verify_prepared
        verify_prepared(config)
        print("Existing research panel and metadata verified; nothing overwritten.")
        return
    inputs, tables, metadata = {}, {}, {}
    for name, info in config["inputs"].items():
        source = REPOSITORY / info["path"]
        check_hash(source, info["sha256"])
        target = local_path(f"data/{name}.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[name] = {**info, "local_path": str(target.relative_to(ROOT))}
        table = pq.read_table(target)
        tables[name] = table.to_pylist()
        metadata[name] = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    sequences, interfaces, pairs = make_panel(tables["structures"], tables["interactions"])
    generated = [local_path(x["local_path"]) for x in inputs.values()]
    for name, rows in (("sequences", sequences), ("interfaces", interfaces), ("pairs", pairs)):
        generated.append(write_table(f"data/{name}.parquet", rows))
    snapshots = {}
    fallback = REPOSITORY / config["snapshot_fallback"]
    for name, relative in config["checkpoints"].items():
        checkpoint = REPOSITORY / relative
        run = checkpoint.parent.parent
        recorded = read_json(run / "source_sha256.json")
        checkpoint_manifest = read_json(checkpoint / "manifest.json")
        check_hash(checkpoint / "trainable.safetensors", checkpoint_manifest["trainable_sha256"])
        checkpoint_config = read_json(checkpoint / "config.json")
        if checkpoint_config.get("clip_pooling", {}).get("type") != "residue_gated":
            raise ValueError("Checkpoint does not use CLIP residue gated pooling")
        for info in inputs.values():
            source = str(REPOSITORY / info["path"])
            if checkpoint_manifest["input_hashes"].get(source) != info["sha256"]:
                raise ValueError("Checkpoint validation panel differs from the pinned research panel")
        sources = {}
        # Use recorded files whenever possible, and explicitly mark dependencies
        # not covered by historical source manifests as fallback copies.
        names = {p.name for p in fallback.glob("*.py")} | {
            p.split("/")[-1] for p in recorded if p.startswith("flashppi/")}
        for filename in sorted(names):
            target = local_path(f"vendor/{name}/flashppi/{filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            key = f"flashppi/{filename}"
            if filename == "__init__.py":
                target.write_text('"""Isolated research model snapshot; see manifests/prepared.json."""\n')
                source, kind = None, "research_import_shim"
            else:
                pinned = key in recorded
                source = run / "code" / key if pinned else fallback / filename
                if pinned:
                    check_hash(source, recorded[key])
                shutil.copyfile(source, target)
                kind = "checkpoint_recorded" if pinned else "fallback_not_checkpoint_recorded"
            sources[str(target.relative_to(ROOT))] = dict(source=str(source) if source else None,
                                                          kind=kind, sha256=sha256(target))
        snapshots[name] = dict(path=str(checkpoint), step=checkpoint_manifest["step"],
                               trainable_sha256=checkpoint_manifest["trainable_sha256"],
                               metadata_hashes={f: sha256(checkpoint / f) for f in ("config.json", "manifest.json")},
                               recorded_sources_sha256=sha256(run / "source_sha256.json"), sources=sources)
    summary = dict(structures=len(tables["structures"]), interfaces=len(interfaces),
                   interaction_rows=len(pairs), interaction_positive=sum(p["label"] for p in pairs),
                   unique_sequences=len(sequences), structural_sequences=sum(x["for_interface"] for x in sequences),
                   interaction_sequences=sum(x["for_interaction"] for x in sequences),
                   structural_dependence_blocks=len({x["block_id"] for x in interfaces}),
                   max_sequence_length=max(x["length"] for x in sequences),
                   annotation_status="Conservation and relative SASA not supplied")
    manifest = dict(study_sha256=sha256(ROOT / "study.json"), inputs=inputs,
                    source_metadata=metadata, summary=summary, checkpoints=snapshots,
                    generated_hashes={str(p.relative_to(ROOT)): sha256(p) for p in generated})
    write_json("manifests/prepared.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
