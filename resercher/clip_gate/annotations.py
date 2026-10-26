"""Independent, cached structure/conservation annotation of the frozen gate panel.

These are annotation-quality rules, never training-data filtering rules. No split,
contact label, production code, model parameter or gate value is changed.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import io
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlparse

from Bio import Align, SeqIO
from Bio.Data.PDBData import residue_sasa_scales
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1
import numpy as np
import pyarrow.parquet as pq
import requests

from .common import ROOT, local_path, read_json, sequence_id, sha256, write_json

BASE = "https://consurfdb.tau.ac.il"
PREPARED = Path("/media/zlab/lexar4t/ppidb/prepared")
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
BACKBONE = {"N", "CA", "C", "O"}
SIDECHAINS = {
    "A": "CB", "R": "CB CG CD NE CZ NH1 NH2", "N": "CB CG OD1 ND2",
    "D": "CB CG OD1 OD2", "C": "CB SG", "Q": "CB CG CD OE1 NE2",
    "E": "CB CG CD OE1 OE2", "G": "", "H": "CB CG ND1 CD2 CE1 NE2",
    "I": "CB CG1 CG2 CD1", "L": "CB CG CD1 CD2", "K": "CB CG CD CE NZ",
    "M": "CB CG SD CE", "F": "CB CG CD1 CD2 CE1 CE2 CZ", "P": "CB CG CD",
    "S": "CB OG", "T": "CB OG1 CG2", "W": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "Y": "CB CG CD1 CD2 CE1 CE2 CZ OH", "V": "CB CG1 CG2",
}
POLICY = {
    "sasa_probe_radius_angstrom": 1.4, "sasa_sphere_points": 960,
    "sasa_normalization": "Tien et al. 2013 theoretical maxima (Biopython Wilke)",
    "sasa_context": "isolated observed chain in bound coordinates; protein heavy atoms only",
    "sasa_quality": "canonical residue with every expected heavy atom present at positive occupancy",
    "mapping_min_identity": .99, "mapping_min_target_coverage": .95,
    "msa_min_sequence_query_coverage": .5,
    "msa_identity_reweighting_threshold": .8,
    "msa_neighbor_min_overlap_fraction_of_longer_sequence": .5,
    "conservation_min_site_neff": 10., "conservation_min_site_coverage": .5,
    "conservation_exclude_consurf_star": True,
    "conservation_score": "negative normalized ConSurf Rate4Site SCORE; higher is more conserved",
}


def targets():
    rows = pq.read_table(ROOT / "data/structures.parquet").to_pylist()
    result = []
    catalog = PREPARED.parent / "catalog/source_indexes/conglab.sqlite"
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    for row in rows:
        original = row["original_id"].split(":")
        if len(original) != 2:
            raise ValueError(f"Cannot map original chain identifier: {row['original_id']}")
        # Prepared A/B follow sorted sequence hashes, NOT original_id order.
        # Read the original pair-specific component catalog; do not infer a chain
        # identity from an index, or alter any production dataset.
        a, b = original
        for prefix in (a + "-" + b, b + "-" + a, None):
            names = [prefix + "__" + x if prefix else x for x in original]
            components = [connection.execute(
                "SELECT sid,chain,file FROM components WHERE kind=? AND name=?",
                (row["kind"], name)).fetchone() for name in names]
            if all(components):
                break
        if not all(components) or {x[0] for x in components} != {row["seq1_id"], row["seq2_id"]}:
            raise ValueError(f"Cannot validate component-to-prepared sequence mapping: {row['original_id']}")
        for side in (1, 2):
            candidates = [i for i, component in enumerate(components) if component[0] == row[f"seq{side}_id"]]
            if len(candidates) != 1:
                raise ValueError("Ambiguous original component mapping")
            idx = candidates[0]
            pdb, chain = original[idx].rsplit("_", 1)
            result.append(dict(sequence_id=sequence_id(row[f"seq{side}"]), sequence=row[f"seq{side}"],
                               pdb_id=pdb.upper(), original_chain=chain,
                               chain=row[f"chain{side}"], pdb_file=row["pdb_file"],
                               component_catalog=str(catalog), component_name=names[idx],
                               component_sid=components[idx][0], component_chain=components[idx][1],
                               component_file=components[idx][2]))
    connection.close()
    if len({r["sequence_id"] for r in result}) != len(result):
        raise ValueError("This annotation version requires one structural context per unique sequence")
    return result


def missing_atoms(aa, atoms):
    if aa not in SIDECHAINS:
        return ["noncanonical_residue"]
    return sorted((BACKBONE | set(SIDECHAINS[aa].split())) - set(atoms))


def structure_annotation(item):
    cache = local_path(f"cache/annotations/sasa/{item['sequence_id']}.json")
    source = (PREPARED / item["pdb_file"]).resolve()
    if not source.is_relative_to(PREPARED) or not source.is_file():
        raise ValueError(f"Missing local structure: {source}")
    source_sha = sha256(source)
    if cache.exists():
        saved = read_json(cache)
        if saved["source_sha256"] != source_sha or saved["policy"] != POLICY:
            raise ValueError("SASA cache source/policy changed; create an explicit new study version")
        return saved
    # The initial coordinate-only pass used catalog IDs with a seq_ prefix.
    # Its coordinate/sequence equality check was already exact; reuse only its
    # expensive SASA values and replace the audited chain-source metadata.
    pilot = local_path(f"cache/annotations/sasa/seq_{item['sequence_id']}.json")
    if pilot.exists():
        saved = read_json(pilot)
        if saved["source_sha256"] == source_sha and saved["policy"] == POLICY and saved["target"]["sequence"] == item["sequence"]:
            saved.update(target=item, coordinate_calculation_cache_reused_sha256=sha256(pilot))
            write_json(cache, saved)
            return saved
    with gzip.open(source, "rt") as handle:
        model = PDBParser(QUIET=True).get_structure("observed", handle)[0]
    chain = model[item["chain"]].copy()
    # Keep observed polymer residues, remove solvent/ligands and hydrogens.
    for residue in list(chain):
        if residue.id[0] != " ":
            chain.detach_child(residue.id)
            continue
        for atom in list(residue):
            if atom.element in ("H", "D"):
                residue.detach_child(atom.id)
    residues = list(chain)
    sequence = "".join(seq1(r.resname, custom_map={"MSE": "M", "UNK": "X"}) for r in residues)
    if sequence != item["sequence"]:
        raise ValueError(f"Coordinate sequence mismatch: {item['pdb_id']} {item['original_chain']}")
    ShrakeRupley(probe_radius=POLICY["sasa_probe_radius_angstrom"],
                n_points=POLICY["sasa_sphere_points"]).compute(chain, level="R")
    result = []
    for position, (aa, residue) in enumerate(zip(sequence, residues), 1):
        present = {a.name for a in residue if (a.get_occupancy() or 0) > 0}
        missing = missing_atoms(aa, present)
        maximum = residue_sasa_scales["Wilke"].get(residue.resname)
        raw = float(residue.sasa / maximum) if maximum else None
        result.append(dict(position=position, aa=aa, pdb_resnum=residue.id[1],
                           pdb_icode=residue.id[2].strip(), asa=float(residue.sasa),
                           rsa_unclipped=raw,
                           relative_sasa=float(np.clip(raw, 0, 1)) if raw is not None and not missing else None,
                           missing_heavy_atoms=missing, complete_heavy_atoms=not missing))
    saved = dict(target=item, source=str(source), source_sha256=source_sha, policy=POLICY,
                 residues=result, altloc_policy="Biopython highest-occupancy selected conformer",
                 caveat="Missing atoms of neighboring residues can still inflate measured exposure")
    write_json(cache, saved)
    return saved


def fetch(session, url, output, params=None):
    output = local_path(output)
    if output.exists():
        return output.read_text()
    if urlparse(url).hostname != "consurfdb.tau.ac.il":
        raise ValueError(f"Unexpected annotation host: {url}")
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=(10, 40))
            response.raise_for_status()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(response.text)
            write_json(str(output) + ".source.json", dict(url=response.url, status=response.status_code,
                       retrieved_utc=datetime.now(timezone.utc).isoformat(), sha256=sha256(output)))
            time.sleep(.15)
            return response.text
        except requests.RequestException:
            if "response" in locals() and response.status_code in (400, 404):
                raise
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def download_consurf(item):
    import json
    directory = local_path(f"cache/annotations/consurf/{item['pdb_id']}_{item['original_chain']}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "download.json").exists():
        result = read_json(directory / "download.json")
        if result["status"] == "downloaded":
            return dict(result, target=item)
    with requests.Session() as session:
        session.headers["User-Agent"] = "FlashPPI-gate-annotation-research/1.0 (cached public PDB annotations)"
        try:
            identity = json.loads(fetch(session, BASE + "/get_identical_chain/", directory / "identity.json",
                                  dict(unique_pdb=item["pdb_id"], unique_chain=item["original_chain"])))
            data = json.loads(fetch(session, BASE + "/get_final_data/", directory / "metadata.json",
                              dict(unique_pdb=item["pdb_id"], **identity)))
            for key, name in (("Score_File", "grades.txt"), ("msa_fasta", "msa.fasta")):
                fetch(session, data[key], directory / name)
            result = dict(status="downloaded", directory=str(directory.relative_to(ROOT)),
                          target=item, representative=identity)
        except (requests.RequestException, ValueError, KeyError) as error:
            result = dict(status="unavailable", target=item, error=str(error))
    write_json(directory / "download.json", result)
    return result


def parse_grades(text):
    result = []
    for line in text.splitlines():
        if not re.match(r"^\s*\d+\s+[A-Z]\s+", line):
            continue
        fields = [s.strip() for s in line.split("\t")]
        if len(fields) < 9:
            raise ValueError(f"Unexpected ConSurf score layout: {line[:80]}")
        count, total = map(int, fields[8].split("/"))
        result.append(dict(position=int(fields[0]), aa=fields[1], atom=fields[2],
                           conservation=-float(fields[3]), grade=int(fields[4].replace("*", "")),
                           insufficient="*" in fields[4], msa_count=count, msa_total=total))
    if not result or [r["position"] for r in result] != list(range(1, len(result) + 1)):
        raise ValueError("ConSurf position table missing, duplicated or out of order")
    return result


def map_positions(target, reference):
    """Map observed sequence to SEQRES; transfer only exact AA matches, not residue numbers."""
    if target == reference:
        return list(range(len(target))), dict(identity=1., coverage=1., mismatches=0, method="exact")
    aligner = Align.PairwiseAligner(mode="global", match_score=3, mismatch_score=-8,
                                    open_gap_score=-10, extend_gap_score=-.5)
    alignments = aligner.align(target, reference)
    first = alignments[0]
    mapping = [-1] * len(target)
    aligned, matches = 0, 0
    for (a, b), (c, d) in zip(*first.aligned):
        if b - a != d - c:
            raise ValueError("Invalid alignment block")
        for i, j in zip(range(a, b), range(c, d)):
            aligned += 1
            if target[i] == reference[j]:
                mapping[i] = j
                matches += 1
    identity, coverage = matches / max(1, aligned), matches / len(target)
    if identity < POLICY["mapping_min_identity"] or coverage < POLICY["mapping_min_target_coverage"]:
        raise ValueError(f"Conservation mapping rejected: identity={identity:.3f}, target coverage={coverage:.3f}")
    # Repetitive sequences can have equally good, differently positioned alignments.
    # Reject ambiguous whole-chain transfer rather than choosing arbitrary indices.
    try:
        second = alignments[1]
    except IndexError:
        second = None
    if second is not None and not np.array_equal(first.coordinates, second.coordinates):
        raise ValueError("Ambiguous optimal sequence-to-ConSurf alignment")
    return mapping, dict(identity=identity, coverage=coverage, mismatches=aligned - matches,
                         method="unique_global_alignment_exact_AA_transfers")


def msa_quality(text, reference):
    records = list(SeqIO.parse(io.StringIO(text), "fasta"))
    if len(records) < 2 or len({len(r.seq) for r in records}) != 1:
        raise ValueError("Invalid or insufficient aligned FASTA")
    strings = [str(r.seq).upper().replace(".", "-") for r in records]
    candidates = [i for i, s in enumerate(strings) if s.replace("-", "") == reference]
    if not candidates:
        raise ValueError("No exact reference sequence in ConSurf MSA")
    query_index = candidates[0]
    query_columns = np.array([i for i, aa in enumerate(strings[query_index]) if aa != "-"])
    matrix = np.array([list(s) for s in strings], dtype="U1")[:, query_columns]
    # Quality weights use only query-aligned, canonical residues. Rate4Site itself
    # remains the externally computed phylogenetic conservation estimator.
    valid = np.isin(matrix, list(CANONICAL))
    selected = valid.mean(axis=1) >= POLICY["msa_min_sequence_query_coverage"]
    matrix, valid = matrix[selected], valid[selected]
    if not len(matrix):
        raise ValueError("No sufficiently covered homologs")
    lengths = valid.sum(axis=1)
    neighbors = np.zeros(len(matrix), dtype=int)
    for i in range(len(matrix)):
        overlap = valid & valid[i]
        counts = overlap.sum(axis=1)
        identity = ((matrix == matrix[i]) & overlap).sum(axis=1) / np.maximum(counts, 1)
        comparable = counts >= POLICY["msa_neighbor_min_overlap_fraction_of_longer_sequence"] * np.maximum(lengths, lengths[i])
        neighbors[i] = np.sum(comparable & (identity >= POLICY["msa_identity_reweighting_threshold"]))
    weights = 1 / np.maximum(neighbors, 1)
    site_neff = weights @ valid
    coverage = valid.mean(axis=0)
    return dict(n_sequences=len(records), n_retained=int(len(matrix)), neff=float(weights.sum()),
                reference_id=records[query_index].id), site_neff, coverage


def conservation_annotation(item):
    cache = local_path(f"cache/annotations/conservation/{item['sequence_id']}.json")
    if cache.exists():
        saved = read_json(cache)
        if saved["policy"] != POLICY:
            raise ValueError("Conservation annotation policy changed")
        return saved
    downloaded = download_consurf(item)
    result = dict(target=item, policy=POLICY, download=downloaded, residues=[])
    if downloaded["status"] != "downloaded":
        result.update(status="unavailable", error=downloaded["error"])
        # Failed downloads are retryable on the next invocation.
        return result
    directory = local_path(downloaded["directory"])
    result["source_hashes"] = {p.name: sha256(p) for p in
                               (directory / "grades.txt", directory / "msa.fasta", directory / "metadata.json")}
    try:
        grades = parse_grades((directory / "grades.txt").read_text())
        reference = "".join(r["aa"] for r in grades)
        mapping, mapping_info = map_positions(item["sequence"], reference)
        quality, neff, coverage = msa_quality((directory / "msa.fasta").read_text(), reference)
        for i, j in enumerate(mapping):
            row = dict(position=i + 1, aa=item["sequence"][i], reference_position=j + 1 if j >= 0 else None,
                       conservation=None, msa_neff=None, msa_coverage=None, exclusion="unmapped")
            if j >= 0:
                grade = grades[j]
                reasons = []
                if grade["insufficient"]:
                    reasons.append("consurf_insufficient")
                if neff[j] < POLICY["conservation_min_site_neff"]:
                    reasons.append("site_neff_below_10")
                if coverage[j] < POLICY["conservation_min_site_coverage"]:
                    reasons.append("site_coverage_below_half")
                row.update(conservation=grade["conservation"] if not reasons else None,
                           conservation_unfiltered=grade["conservation"], consurf_grade=grade["grade"],
                           consurf_insufficient=grade["insufficient"], consurf_msa_count=grade["msa_count"],
                           consurf_msa_total=grade["msa_total"], msa_neff=float(neff[j]),
                           msa_coverage=float(coverage[j]), exclusion=";".join(reasons))
            result["residues"].append(row)
        result.update(status="mapped", mapping=mapping_info, msa=quality)
    except (ValueError, IndexError) as error:
        result.update(status="rejected", error=str(error))
    write_json(cache, result)
    return result


def annotate(kind, workers):
    items = targets()
    operation = structure_annotation if kind == "sasa" else conservation_annotation
    executor_type = ProcessPoolExecutor if kind == "sasa" else ThreadPoolExecutor
    complete = []
    with executor_type(max_workers=workers) as executor:
        futures = {executor.submit(operation, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            result = future.result()
            complete.append(result)
            print(f"{kind} {len(complete)}/{len(items)} {item['pdb_id']}:{item['original_chain']} "
                  f"{result.get('status', 'computed')} {result.get('error', '')}", flush=True)
    write_json(f"manifests/annotations_{kind}.json", dict(policy=POLICY, results=complete))


def assemble():
    structures = read_json(local_path("manifests/annotations_sasa.json"))["results"]
    conservation = read_json(local_path("manifests/annotations_conservation.json"))["results"]
    by_seq = {r["target"]["sequence_id"]: r for r in conservation}
    output = local_path("data/annotations.csv")
    fields = ["sequence_id", "position", "aa", "conservation", "relative_sasa", "msa_neff", "msa_coverage"]
    detailed, rows = [], []
    for structure in sorted(structures, key=lambda r: r["target"]["sequence_id"]):
        sid = structure["target"]["sequence_id"]
        cons = {r["position"]: r for r in by_seq[sid]["residues"]}
        for residue in structure["residues"]:
            c = cons.get(residue["position"], {})
            row = dict(sequence_id=sid, **residue,
                       **{k: c.get(k) for k in ("conservation", "msa_neff", "msa_coverage")})
            rows.append({k: row[k] for k in fields})
            detailed.append(dict(**row, conservation_unfiltered=c.get("conservation_unfiltered"),
                                 consurf_insufficient=c.get("consurf_insufficient"),
                                 conservation_exclusion=c.get("exclusion", by_seq[sid].get("error")),
                                 original_pdb=structure["target"]["pdb_id"],
                                 original_chain=structure["target"]["original_chain"]))
    if output.exists():
        raise FileExistsError("Do not overwrite a previously assembled annotation version")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(detailed), local_path("data/annotation_details.parquet"))
    write_json(output.with_suffix(".json"), dict(
        conservation_direction="higher_is_more_conserved", date=datetime.now(timezone.utc).isoformat(),
        method=POLICY, source="Local original CongLab coordinates; ConSurfDB public PDB-chain profiles/MSAs",
        source_note="ConSurfDB home states last update 2025-02-19; exact response hashes and retrieval dates cached",
        annotation_source_sha256=sha256(__file__),
        manifests={name: sha256(local_path(f"manifests/annotations_{name}.json")) for name in ("sasa", "conservation")},
        source_panel_sha256=sha256(ROOT / "data/structures.parquet"),
        counts=dict(rows=len(rows), rsa=sum(r["relative_sasa"] is not None for r in rows),
                    conservation=sum(r["conservation"] is not None for r in rows)),
        caveats=["Exploratory annotations only; original train/val/test members and filtering remain unchanged",
                 "Isolated bound-conformation SASA is not experimentally unbound SASA",
                 "RSA above 1 is clipped for generic annotation bins; unbounded RSA retained in details",
                 "Neighboring missing atoms can inflate SASA; own incomplete residues excluded",
                 "ConSurf homolog families may include paralogs; correlation is not causal functional importance"]))
    print(read_json(output.with_suffix(".json"))["counts"], flush=True)
