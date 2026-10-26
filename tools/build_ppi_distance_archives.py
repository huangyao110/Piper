#!/usr/bin/env python3
"""Create additive, verified PPI distance NPZ/TAR shards from a pinned release.

Original archives, structures and contact labels are opened read-only. Samples
are matched against both the published Parquet and the existing contact TAR.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import gemmi
import numpy as np
import pyarrow.parquet as pq
from Bio.Data.PDBData import protein_letters_3to1_extended
from scipy.spatial.distance import cdist

BASE = Path("/media/zlab/lexar4t/ppidb")
SNAPSHOT = "20261009_101259"
RULES = dict(
    schema_version=1, kind="ppi", sample_unit="structure", matrix="L1 x L2",
    distance_dtype="float32", unit="angstrom", method="cb", glycine_atom="CA",
    structure_model=0, conformer="first_conformer", occupancy="strictly positive",
    unknown_positions="NaN; valid1[:, None] & valid2[None, :]",
    coordinates_dtype="float64", contact_cutoff_angstrom=12.0, comparison="lt",
    rounding="Nearest float32; below-12 values rounding to 12 use nextafter(12, -inf).",
    truncation="none", diagonal_policy="retain cross-chain homomer contacts",
    compression="numpy.savez_compressed (ZIP DEFLATE), uncompressed outer TAR",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def extract_chain(model, name):
    """Match dataset_build/build_contact_labels.py's representative-atom rules."""
    chain = model.find_chain(name)
    if chain is None:
        raise ValueError(f"Missing chain {name}")
    residues = list(chain.get_polymer().first_conformer())
    sequence = "".join(protein_letters_3to1_extended.get(r.name.strip().upper(), "X")
                       for r in residues)
    if not sequence:
        raise ValueError(f"Empty polymer chain {name}")
    coordinates = np.full((len(residues), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(residues), dtype=np.bool_)
    for i, residue in enumerate(residues):
        selected = "CA" if residue.name.strip().upper() == "GLY" else "CB"
        points = [np.array([a.pos.x, a.pos.y, a.pos.z], dtype=np.float64)
                  for a in residue.first_conformer()
                  if a.name.strip() == selected and not a.element.is_hydrogen and a.occ > 0]
        points = [point for point in points if np.isfinite(point).all()]
        if len(points) > 1:
            raise ValueError(f"Multiple representative atoms: chain {name}, index {i}")
        if points:
            coordinates[i] = points[0]
            valid[i] = True
    return sequence, coordinates, valid


def distance_matrix(x, y, valid1, valid2, contacts):
    """Full cross-chain distances; preserve the source's strict contact boundary.

    Coordinates are retained in float64 in the NPZ for exact recomputation.
    The special float32 rounding case is counted and changes at most one ULP.
    It avoids losing a true <12 contact to a float32 value rounded to 12.0.
    """
    d64 = cdist(x, y)
    valid = valid1[:, None] & valid2[None, :]
    if not np.isfinite(d64[valid]).all() or np.any(d64[valid] < 0):
        raise ValueError("Invalid finite distance")
    d64[~valid] = np.nan
    contacts = np.asarray(contacts, dtype=np.int64).reshape(-1, 2)
    if len(contacts) and (np.any(contacts < 0) or np.any(contacts >= d64.shape)):
        raise ValueError("Contact index outside matrix")
    expected = np.unique(contacts[:, 0] * len(y) + contacts[:, 1])
    if not np.array_equal(np.flatnonzero(d64 < 12.0), expected):
        raise ValueError("PDB float64 distances disagree with published contact labels")
    d32 = d64.astype(np.float32)
    rounded_up = (d64 < 12.0) & (d32 >= np.float32(12.0))
    d32[rounded_up] = np.nextafter(np.float32(12.0), np.float32(-np.inf))
    if not np.array_equal(np.flatnonzero(d32 < 12.0), expected):
        raise ValueError("Stored float32 distances disagree with contact labels")
    if not np.array_equal(np.isfinite(d32), valid):
        raise ValueError("Distance mask mismatch or float32 overflow")
    maximum_error = float(np.max(np.abs(d32[valid].astype(np.float64) - d64[valid]))) if valid.any() else 0.
    return d32, dict(cells=int(d32.size), valid_cells=int(valid.sum()),
                     contacts=int(len(expected)), rounding_adjustments=int(rounded_up.sum()),
                     max_abs_rounding_error_angstrom=maximum_error,
                     max_distance_angstrom=float(d64[valid].max()) if valid.any() else None)


def calculate(task):
    prepared, row = task
    pid = row["pair_id"]
    try:
        root = Path(prepared).resolve()
        path = (root / row["pdb_file"]).resolve(strict=True)
        if not path.is_relative_to(root / "pdbs"):
            raise ValueError("PDB path outside prepared/pdbs")
        raw = path.read_bytes()
        body = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        structure = gemmi.read_pdb_string(body.decode("ascii"))
        structure.setup_entities()
        a, x, va = extract_chain(structure[0], row["chain1"])
        b, y, vb = extract_chain(structure[0], row["chain2"])
        if a != row["seq1"] or b != row["seq2"]:
            raise ValueError("PDB sequence differs from pinned Parquet")
        if [len(a), len(b)] != [row["len1"], row["len2"]]:
            raise ValueError("Sequence lengths differ from pinned Parquet")
        for sequence, side in ((a, 1), (b, 2)):
            if "seq_" + hashlib.sha256(sequence.encode("ascii")).hexdigest() != row[f"seq{side}_id"]:
                raise ValueError("Sequence hash mismatch")
        if not np.array_equal(va, row["valid1"]) or not np.array_equal(vb, row["valid2"]):
            raise ValueError("PDB representative-atom masks disagree with published labels")
        distances, stats = distance_matrix(x, y, va, vb, row["contacts"])
        output = BytesIO()
        np.savez_compressed(
            output, distances=distances, shape=np.array(distances.shape, dtype=np.int32),
            valid1=va, valid2=vb, coordinates1=x, coordinates2=y,
            pair_id=pid, complex_id=row["complex_id"], sequence_pair_id=row["sequence_pair_id"],
            seq1_id=row["seq1_id"], seq2_id=row["seq2_id"], seq1=a, seq2=b,
            chain1=row["chain1"], chain2=row["chain2"], pdb_file=row["pdb_file"],
            pdb_sha256=hashlib.sha256(raw).hexdigest(), structure_source=row["structure_source"],
            schema_version=np.int32(RULES["schema_version"]), method="cb", unit="angstrom",
            glycine_atom="CA", cutoff_angstrom=12., comparison="lt",
            rounding_adjustments=np.int64(stats["rounding_adjustments"]),
        )
        return pid, output.getvalue(), stats, row["structure_source"]
    except Exception as exc:
        raise RuntimeError(f"{pid}: {type(exc).__name__}: {exc}") from exc


def records(training, manifest, prefix, fingerprints=None):
    """Read only row groups overlapping this hash prefix, preserving ID order."""
    lower, upper = f"ppi_{prefix}", f"ppi_{prefix}g"
    previous = ""
    for shard in manifest["parquet_shards"]:
        if shard["kind"] != "ppi":
            continue
        pf = pq.ParquetFile(training / shard["file"])
        col = pf.schema_arrow.get_field_index("pair_id")
        checked = False
        for number in range(pf.num_row_groups):
            stats = pf.metadata.row_group(number).column(col).statistics
            if stats is not None and (stats.max < lower or stats.min >= upper):
                continue
            if not checked and fingerprints is not None:
                source_path = training / shard["file"]
                actual_hash = sha256(source_path)
                if actual_hash != shard["sha256"]:
                    raise ValueError(f"Published Parquet checksum mismatch: {source_path}")
                fingerprints[str(source_path)] = actual_hash
                checked = True
            for row in pf.read_row_group(number, use_threads=False).to_pylist():
                if not row["pair_id"].startswith(lower):
                    continue
                if row["pair_id"] <= previous:
                    raise ValueError("PPI rows must have unique, increasing IDs")
                previous = row["pair_id"]
                if row["kind"] != "ppi" or not re.fullmatch(r"ppi_[0-9a-f]{64}", previous):
                    raise ValueError("Invalid PPI sample identity")
                yield row


def contact_members(archive):
    with tarfile.open(archive, "r|") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".npz"):
                raise ValueError(f"Unexpected source TAR member: {member.name}")
            yield member.name, tar.extractfile(member).read()


def checked_tasks(prepared, rows, archive):
    """Require exact row/member agreement; never silently skip a missing sample."""
    originals = iter(contact_members(archive))
    for row in rows:
        name, body = next(originals, (None, None))
        expected_name = "concat/" + row["label_file"]
        if name != expected_name:
            raise ValueError(f"Source TAR/Parquet membership mismatch: {name} vs {expected_name}")
        with np.load(BytesIO(body), allow_pickle=False) as original:
            for key in ("pair_id", "complex_id", "sequence_pair_id", "seq1_id", "seq2_id"):
                if str(original[key]) != row[key]:
                    raise ValueError(f"Original NPZ identity mismatch: {key}")
            if (str(original["method"]) != "cb" or float(original["cutoff_angstrom"]) != 12.
                    or str(original["comparison"]) != "lt"):
                raise ValueError("Original contact rules differ")
            for key in ("contacts", "valid1", "valid2"):
                target = np.asarray(row[key])
                if key == "contacts":
                    target = target.reshape(-1, 2)
                if not np.array_equal(original[key], target):
                    raise ValueError(f"Original TAR/Parquet {key} mismatch: {row['pair_id']}")
            if original["shape"].tolist() != [row["len1"], row["len2"]]:
                raise ValueError("Original shape mismatch")
        yield str(prepared), row
    if next(originals, None) is not None:
        raise ValueError("Source TAR contains unmatched samples")


def ordered_results(executor, tasks, pending_limit):
    pending = deque()
    iterator = iter(tasks)
    for _ in range(pending_limit):
        task = next(iterator, None)
        if task is None:
            break
        pending.append(executor.submit(calculate, task))
    while pending:
        yield pending.popleft().result()
        task = next(iterator, None)
        if task is not None:
            pending.append(executor.submit(calculate, task))


def verify_archive(path, original_path, expected_count):
    """Independently reload every output member and compare original contacts."""
    originals = iter(contact_members(original_path))
    count = cells = 0
    for name, body in contact_members(path):
        old_name, old_body = next(originals, (None, None))
        if name.replace("distances/", "concat/", 1) != old_name:
            raise ValueError("Verification member mismatch")
        with np.load(BytesIO(body), allow_pickle=False) as f, np.load(BytesIO(old_body), allow_pickle=False) as old:
            if str(f["pair_id"]) != str(old["pair_id"]):
                raise ValueError("Verification sample ID mismatch")
            d = f["distances"]
            if d.dtype != np.float32 or list(d.shape) != old["shape"].tolist():
                raise ValueError("Verification dtype/shape mismatch")
            for key in ("valid1", "valid2"):
                if not np.array_equal(f[key], old[key]):
                    raise ValueError("Verification validity mask mismatch")
            mask = old["valid1"][:, None] & old["valid2"][None, :]
            if not np.array_equal(np.isfinite(d), mask) or not np.isnan(d[~mask]).all():
                raise ValueError("Verification distance mask mismatch")
            ij = old["contacts"]
            if not np.array_equal(np.flatnonzero(d < 12.), np.unique(ij[:, 0].astype(np.int64) * d.shape[1] + ij[:, 1])):
                raise ValueError("Verification contact labels mismatch")
            count += 1
            cells += d.size
    if next(originals, None) is not None or count != expected_count:
        raise ValueError("Verification sample count mismatch")
    return dict(records=count, cells=cells, all_members_reloaded=True,
                contacts_exact=True, masks_exact=True, errors=0)


def readme():
    return """# PPI distance matrices

These new files complement the original contact release; its files are unchanged.
Only prefixes listed in `DISTANCES.json` have been generated. This is not a new
train/validation/test split. Continue using the project's frozen split by pair_id.

`archives/npz-distance-ppi-XX.tar` contains
`distances/ppi/XX/YY/<pair_id>.npz`, one full, untruncated matrix per structure.
Each prefix has exactly the members of the matching `npz-ppi-XX.tar`.

NPZ fields:
- `distances`: float32 [L1,L2], Angstrom, cross-chain C-beta distance (glycine C-alpha).
- `valid1`, `valid2`: boolean residue validity; unknown distances are NaN.
- `coordinates1`, `coordinates2`: float64 [L,3] representative-atom coordinates;
  unavailable atoms are NaN. These support higher-precision recomputation.
- `shape`, sequence strings/IDs, pair/structure IDs, chain names, PDB path and
  SHA256, structure source, schema and geometry metadata.

Both matrix axes match seq1/seq2 exactly. First model and first conformer are used;
nonpositive occupancy and missing non-glycine C-beta atoms are masked. Same-sequence
chains remain distinct; the cross-chain diagonal is retained. No distance clipping
or binning is performed. Source structure quality is not upgraded by this export.

Float64 distances are converted to float32. If a value strictly below 12 rounds to
12, it is stored as the preceding float32 value; `rounding_adjustments` counts these
cases. Thus `<12` exactly reproduces the source binary labels. Full float64
coordinates are retained to reproduce distances without this rounding convention.

```python
import io, tarfile, numpy as np
with tarfile.open('archives/npz-distance-ppi-00.tar') as archive:
    member = next(iter(archive))
    with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as f:
        distance = f['distances']
        valid = f['valid1'][:, None] & f['valid2'][None, :]
        contacts = valid & (distance < 12.0)
        # For distograms: bin distances only where valid is True.
```

`DISTANCES_SHA256SUMS` checks the new archives only. Per-shard receipts record
source fingerprints, rules, counts, checks and the exact generation command.
The original release README and source attribution/terms continue to apply.
"""


def run(args):
    output, training, prepared = args.output.resolve(), args.training_release.resolve(), args.prepared.resolve()
    metadata_path = output / "DATASET.json"
    metadata = json.loads(metadata_path.read_text())
    training_path = training / "DATASET.json"
    manifest = json.loads(training_path.read_text())
    if metadata["snapshot"] != SNAPSHOT or manifest["snapshot"] != SNAPSHOT:
        raise ValueError("Unexpected source snapshot")
    prefix = args.prefix
    source_entry = next(s for s in metadata["shards"] if s["kind"] == "ppi" and s["prefix"] == prefix)
    original = output / source_entry["file"]
    source_hash = sha256(original)
    if source_hash != source_entry["sha256"]:
        raise ValueError("Original contact TAR fails its published checksum")
    # Only additive names may be mutated below this directory.
    receipts = output / "distance_receipts"
    receipts.mkdir(exist_ok=True)
    archive = output / "archives" / f"npz-distance-ppi-{prefix}.tar"
    receipt_path = receipts / f"ppi-{prefix}.json"
    status_path = receipts / f"ppi-{prefix}.status.json"
    source_fingerprints = {str(metadata_path): sha256(metadata_path), str(training_path): sha256(training_path)}
    for name in ("README.md", "SHA256SUMS", "PIN.json", "reader.py"):
        path = output / name
        source_fingerprints[str(path)] = sha256(path)
    with (receipts / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if archive.exists() or receipt_path.exists():
            if not (archive.exists() and receipt_path.exists()):
                raise FileExistsError("Unreceipted archive or missing archive; refuse overwrite")
            previous = json.loads(receipt_path.read_text())
            if previous["rules"] != RULES or sha256(archive) != previous["sha256"] or previous["source_archive_sha256"] != source_hash:
                raise ValueError("Existing output does not match receipt/rules/source")
            checked = verify_archive(archive, original, source_entry["files"])
            print(json.dumps(dict(status="already_complete_verified", archive=str(archive), verification=checked)), flush=True)
            return
        partial = archive.with_name(archive.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"Partial archive retained for inspection: {partial}")
        if shutil.disk_usage(output).free < args.reserve_gib * 2**30:
            raise OSError("Insufficient free space above reserved floor")
        started = time.monotonic()
        totals = Counter(records=0, cells=0, valid_cells=0, contacts=0, rounding_adjustments=0, npz_bytes=0)
        sources = Counter()
        maxima = dict(max_abs_rounding_error_angstrom=0., max_distance_angstrom=0.)
        def status(state, **extra):
            value = dict(state=state, prefix=prefix, expected_records=source_entry["files"],
                         pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                         elapsed_seconds=round(time.monotonic()-started, 2), **totals, **extra)
            atomic_json(status_path, value)
            print(json.dumps(value), flush=True)
        status("generating")
        try:
            tasks = checked_tasks(prepared, records(training, manifest, prefix, source_fingerprints), original)
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
                with tarfile.open(partial, "w", format=tarfile.USTAR_FORMAT) as tar:
                    for pid, body, stats, source in ordered_results(executor, tasks, 2*args.workers):
                        if shutil.disk_usage(output).free - len(body) < args.reserve_gib * 2**30:
                            raise OSError("Stopped before exhausting reserved disk space")
                        digest = pid.split("_", 1)[1]
                        member = tarfile.TarInfo(f"distances/ppi/{digest[:2]}/{digest[2:4]}/{pid}.npz")
                        member.size, member.mode, member.mtime = len(body), 0o644, 0
                        tar.addfile(member, BytesIO(body))
                        totals["records"] += 1
                        totals["npz_bytes"] += len(body)
                        sources[source] += 1
                        for key in ("cells", "valid_cells", "contacts", "rounding_adjustments"):
                            totals[key] += stats[key]
                        for key in maxima:
                            maxima[key] = max(maxima[key], stats[key] or 0.)
                        if totals["records"] % 250 == 0:
                            status("generating")
            if totals["records"] != source_entry["files"]:
                raise ValueError("Generated sample count differs from source shard")
            status("verifying_all_members")
            verified = verify_archive(partial, original, source_entry["files"])
            if sha256(original) != source_hash or any(sha256(p) != h for p, h in source_fingerprints.items()):
                raise ValueError("Source metadata/archive changed during generation")
            archive_hash = sha256(partial)
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            # Exclusive publication protects pre-existing files even on a naming race.
            os.link(partial, archive)
            partial.unlink()
            receipt = dict(file=str(archive.relative_to(output)), sha256=archive_hash, bytes=archive.stat().st_size,
                           prefix=prefix, rules=RULES, **totals, **maxima, source_counts=dict(sources),
                           verification=verified, source_archive=str(original), source_archive_sha256=source_hash,
                           source_fingerprints=source_fingerprints, source_snapshot=SNAPSHOT,
                           dependencies={name: importlib.metadata.version(name) for name in
                                         ("numpy", "scipy", "gemmi", "pyarrow", "biopython")},
                           code_sha256=sha256(__file__), command=[sys.executable, *sys.argv],
                           elapsed_seconds=round(time.monotonic()-started, 2))
            atomic_json(receipt_path, receipt)
            all_receipts = [json.loads(p.read_text()) for p in sorted(receipts.glob("ppi-??.json"))]
            atomic_json(output / "DISTANCES.json", dict(schema_version=1, source_snapshot=SNAPSHOT, rules=RULES,
                        scope="Only listed PPI prefixes; all DDI and unlisted PPI prefixes excluded",
                        complete_prefixes=[r["prefix"] for r in all_receipts],
                        records=sum(r["records"] for r in all_receipts), shards=all_receipts))
            (output / "DISTANCES_README.md").write_text(readme())
            (output / "DISTANCES_SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in all_receipts))
            status("complete", archive=str(archive), sha256=archive_hash, verification=verified)
        except BaseException as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}", partial_archive=str(partial))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / f"distributions/ppidb-npz-{SNAPSHOT}")
    parser.add_argument("--training-release", type=Path, default=BASE / f"distributions/ppidb-training-{SNAPSHOT}")
    parser.add_argument("--prepared", type=Path, default=BASE / "prepared")
    parser.add_argument("--prefix", default="00", help="One two-digit PPI hash prefix (default: 00)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reserve-gib", type=float, default=32.)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{2}", args.prefix) or args.workers < 1 or args.reserve_gib < 0:
        parser.error("Invalid prefix, workers or disk reserve")
    run(args)


if __name__ == "__main__":
    main()
