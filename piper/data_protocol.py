"""Versioned, locked data isolation protocols.

Historical v1 reuses the original classifier/decisions. Fresh v2 and RF2-PPI v3
recompute all members under separate immutable locks; never reinterpret one
version's stored hits with another version's predicate.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .indexed_data import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = 'conglab_ecoli_v1_locked'
LOCK_PATH = ROOT / 'configs_data/conglab_ecoli_filter.lock.json'
LOCK_SHA256 = '125e80a50007beeea0bd987dd843c4fc94fdcda81b51456264847e05004b4fea'

# User-authorized fresh train/test protocol (2026-10-24). The legacy protocol
# above, including its cached negative decisions, remains a separate version.
STRICT_PROTOCOL_ID = 'conglab_distilled_ecoli_strict_v2'
STRICT_LOCK_PATH = ROOT / 'configs_data/conglab_distilled_ecoli_strict_v2.lock.json'
STRICT_LOCK_SHA256 = 'b8a3a133238ac99c09a817241d77fade696d91266b67b4c0fab9b373b10b5a4f'
RF2_PROTOCOL_ID = 'conglab_distilled_ecoli_rf2ppi_v3'
RF2_LOCK_PATH = ROOT / 'configs_data/conglab_distilled_ecoli_rf2ppi_v3.lock.json'
RF2_LOCK_SHA256 = 'c7b87860e08c17a911eae7a01f13b973c34dcd60ff36ffd5998ef9c731f7f7b0'
FRESH_PROTOCOL_IDS = (STRICT_PROTOCOL_ID, RF2_PROTOCOL_ID)


def read_strict_lock(protocol_id=STRICT_PROTOCOL_ID):
    if protocol_id not in FRESH_PROTOCOL_IDS:
        raise ValueError('Unknown fresh filtering protocol: ' + str(protocol_id))
    path, digest = ((RF2_LOCK_PATH, RF2_LOCK_SHA256) if protocol_id == RF2_PROTOCOL_ID
                    else (STRICT_LOCK_PATH, STRICT_LOCK_SHA256))
    if file_sha256(path) != digest:
        raise ValueError('Strict filtering protocol lock changed; create an approved new version')
    return json.loads(path.read_text())


def validate_strict_config(config):
    protocol_id = config.get('filter_protocol')
    lock = read_strict_lock(protocol_id)
    for key, value in lock['required_config'].items():
        if config.get(key) != value:
            raise ValueError(f'Strict filtering protocol requires {key}={value!r}')
    allowed = set(lock['required_config']) | {
        'version', 'release', 'legacy_directory', 'test', 'output', 'mmseqs',
        'duckdb_memory_limit', 'seed'}
    if set(config) - allowed:
        raise ValueError('Unrecognized strict protocol configuration: ' + str(set(config) - allowed))
    return dict(id=protocol_id, lock_sha256=RF2_LOCK_SHA256 if protocol_id == RF2_PROTOCOL_ID else STRICT_LOCK_SHA256)


def verify_strict_inputs(config):
    protocol = validate_strict_config(config)
    lock = read_strict_lock(protocol['id'])
    version = subprocess.check_output([config['mmseqs'], 'version'], text=True).strip()
    if version != lock['search']['version'] or file_sha256(config['mmseqs']) != lock['search']['binary_sha256']:
        raise ValueError('Strict protocol MMseqs version/binary mismatch')
    release = Path(config['release'])
    for name, digest in lock['input_files'].items():
        if file_sha256(release / name) != digest:
            raise ValueError('Strict protocol source input changed: ' + name)
    if file_sha256(config['test']) != lock['test']['sha256']:
        raise ValueError('Frozen E. coli test changed')
    if file_sha256(Path(config['legacy_directory']) / 'train.parquet') != lock['legacy_annotation']['train_sha256']:
        raise ValueError('Historical train used for the before/after annotation changed')
    return protocol


def strict_command(executable, query, target, alignment, temporary, target_count, kind, direction,
                   protocol_id=STRICT_PROTOCOL_ID):
    """All search parameters come from the hashed lock, never a mutable config."""
    lock = read_strict_lock(protocol_id)
    if target_count < 1:
        raise ValueError('Cannot search an empty target database')
    cov_mode = lock['search']['coverage_modes'][kind][direction]
    arguments = list(lock['search']['arguments'])
    arguments[arguments.index('--prefilter-mode') + 1] = str(lock['search']['prefilter_modes'][direction])
    return [str(executable), 'easy-search', str(query), str(target), str(alignment), str(temporary),
            *arguments, '--cov-mode', str(cov_mode), '--max-seqs', str(target_count)]


def verify_strict_search_log(path, direction, protocol_id=STRICT_PROTOCOL_ID):
    text = Path(path).read_text()
    mode = read_strict_lock(protocol_id)['search']['prefilter_modes'][direction]
    version = re.search(r'MMseqs Version:\s*(\S+)', text)
    observed = re.search(r'Prefilter mode\s+(\d+)', text)
    if version is None or version.group(1) != '18.8cc5c' or observed is None or int(observed.group(1)) != mode:
        raise ValueError('Strict MMseqs version/prefilter mode drift')
    if mode == 0:
        verify_search_log(path)
    elif mode != 2 or not re.search(r'\d+ alignments calculated', text):
        raise ValueError('Required direct alignment pass is missing')


def strict_hit(fields, kind, direction, protocol_id=STRICT_PROTOCOL_ID):
    """Integer boundaries; coverage uses the entire uncropped FASTA sequence.

    Columns: query,target,nident,alnlen,qstart,qend,qlen,tstart,tend,tlen.
    MMseqs coordinates are one-based inclusive. For a DDI, only the training
    domain must reach 50% coverage, irrespective of search direction.
    """
    if protocol_id == RF2_PROTOCOL_ID:
        return rf2ppi_hit(fields, kind, direction)
    if protocol_id != STRICT_PROTOCOL_ID:
        raise ValueError('Unknown alignment protocol')
    direction = direction.removesuffix('_nofilter')
    if kind not in ('ppi', 'ddi') or direction not in ('test_to_train', 'train_to_test'):
        raise ValueError('Unknown strict alignment scope')
    if len(fields) != 10:
        raise ValueError('Unexpected MMseqs alignment columns')
    ident, aln, qs, qe, qlen, ts, te, tlen = map(int, fields[2:])
    if not (0 <= ident <= aln and aln > 0 and 1 <= qs <= qe <= qlen and 1 <= ts <= te <= tlen):
        raise ValueError('Invalid MMseqs alignment coordinates/counts')
    qcovered, tcovered = 2 * (qe - qs + 1) >= qlen, 2 * (te - ts + 1) >= tlen
    coverage = (qcovered and tcovered) if kind == 'ppi' else (
        tcovered if direction == 'test_to_train' else qcovered)
    return 10 * ident > 3 * aln and coverage


def rf2ppi_hit(fields, kind, direction):
    """RF2-PPI M3.1/M3.3: significant identity AND either-side coverage.

    PPI chains and DDI domains have the SAME rule. Exact sequences receive no
    exemption from significance. The last two columns are MMseqs evalue,bits.
    """
    if kind not in ('ppi', 'ddi') or direction.removesuffix('_nofilter') not in ('test_to_train', 'train_to_test'):
        raise ValueError('Unknown RF2-PPI alignment scope')
    if len(fields) != 12:
        raise ValueError('RF2-PPI evidence requires 12 columns including evalue,bits')
    ident, aln, qs, qe, qlen, ts, te, tlen = map(int, fields[2:10])
    # MMseqs 18.8cc5c can emit a zero-residue result for low-complexity queries.
    # Its Smith-Waterman no-alignment return leaves significance uninitialized;
    # neither a tiny printed E-value nor these zero coordinates constitute a hit.
    # Recognize only the observed empty-result geometry; other malformed rows
    # remain errors. The caller still verifies both sequence IDs and lengths.
    if ident == aln == qs == ts == te == 0 and qe in (0, 1) and qlen > 0 and tlen > 0:
        return False
    evalue, bits = map(float, fields[10:12])
    if not (0 <= ident <= aln and aln > 0 and 1 <= qs <= qe <= qlen and 1 <= ts <= te <= tlen):
        raise ValueError('Invalid MMseqs alignment coordinates/counts')
    if not (math.isfinite(evalue) and evalue >= 0 and math.isfinite(bits)):
        raise ValueError('Invalid MMseqs significance evidence')
    return (10 * ident > 3 * aln and evalue < 1e-5
            and (2 * (qe - qs + 1) > qlen or 2 * (te - ts + 1) > tlen))


def strict_pair_decision(hit1, hit2, protocol_id=STRICT_PROTOCOL_ID):
    if protocol_id == RF2_PROTOCOL_ID:
        return 'test_both_endpoints_homology' if hit1 and hit2 else 'candidate'
    if protocol_id != STRICT_PROTOCOL_ID:
        raise ValueError('Unknown pair filtering protocol')
    return 'test_either_endpoint_homology' if hit1 or hit2 else 'candidate'


def read_lock():
    if file_sha256(LOCK_PATH) != LOCK_SHA256:
        raise ValueError('Frozen filtering protocol changed; explicit user approval is required')
    return json.loads(LOCK_PATH.read_text())


def validate_config(config):
    """Fail before creating output or reusing a cache with a different policy."""
    required = dict(filter_protocol=PROTOCOL_ID, test_exclusion='both_chains',
                    val_exclusion='either_chain', ppi_coverage='both_coverage_50',
                    ddi_coverage='candidate_coverage_80', preserve_all_legacy=True)
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Frozen filtering protocol requires {key}={value!r}')
    if any(key in config for key in ('homology', 'search_parameters', 'kmer_length', 'sensitivity')):
        raise ValueError('Search overrides are not permitted by the frozen filtering protocol')
    return dict(id=PROTOCOL_ID, lock_sha256=LOCK_SHA256)


def verify_baseline(config):
    protocol = validate_config(config)
    lock = read_lock()
    for name, expected in lock['baseline_files'].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError('Frozen filtering baseline changed: ' + name)
    legacy = Path(config['legacy_directory'])
    for name, entry in lock['frozen_outputs'].items():
        if file_sha256(legacy / name) != entry['sha256']:
            raise ValueError('Frozen legacy split changed: ' + name)
    if config['test_sha256'] != lock['test_sha256'] or file_sha256(config['test']) != lock['test_sha256']:
        raise ValueError('Frozen test changed')
    if config['expected_legacy'] != {'ppi': 256596, 'ddi': 570709}:
        raise ValueError('Original train membership/count cannot change')
    version = subprocess.check_output([config['mmseqs'], 'version'], text=True).strip()
    if version != lock['search']['mmseqs_version']:
        raise ValueError('Frozen MMseqs version changed: ' + version)
    if config['threads'] != lock['search']['threads']:
        raise ValueError('Frozen MMseqs thread setting changed')
    return protocol


def original_record(kind, category):
    directory, suffix = ('conglab_ecoli_audit_v1', 'ecoli') if category == 'test' else ('conglab_ecoli_v1', 'val')
    path = ROOT / f'data/{directory}/{kind}_{suffix}_hits.json'
    expected = read_lock()['baseline_files'][str(path.relative_to(ROOT))]
    if file_sha256(path) != expected:
        raise ValueError('Original homology cache changed: ' + str(path))
    return json.loads(path.read_text())


def reference_fasta(sequences):
    return ''.join(f'>r{i}\n{seq}\n' for i, seq in enumerate(sorted(sequences)))


def locked_command(record, executable, reference, candidate, alignment, temporary, count):
    """Use original search command; pin the originally observed k=6/score=96."""
    command = list(record['command'])
    command[:6] = [executable, 'easy-search', str(reference), str(candidate), str(alignment), str(temporary)]
    command[command.index('--max-seqs') + 1] = str(count)
    # These defaults were logged as 6 and 96 by all four original searches.
    command += ['-k', '6', '--k-score', 'seq:96,prof:2147483647']
    return command


def verify_search_log(path):
    text = Path(path).read_text()
    settings = re.findall(r'Index table k-mer threshold:\s*(\d+) at k-mer size (\d+)', text)
    if not settings or any(pair != ('96', '6') for pair in settings):
        raise ValueError('Actual MMseqs prefilter differs from the original k=6/score=96 protocol')
    version = re.search(r'MMseqs Version:\s*(\S+)', text)
    if version is None or version.group(1) != '18.8cc5c':
        raise ValueError('Actual MMseqs version differs from the frozen protocol')


def restore_known_decisions(hits, decisions, category):
    """Old positive AND negative search decisions are authoritative."""
    known = set(decisions.filter(decisions[category + '_known'])['seq_id'].to_pylist())
    old_hits = set(decisions.filter(decisions[category + '_hit'])['seq_id'].to_pylist())
    if not old_hits <= known:
        raise ValueError('Cached hit is outside its original candidate scope')
    return (set(hits) - known) | old_hits


def legacy_decisions(output, kind):
    """Recover exact original candidate scopes, verified against FASTA hashes.

    A missing ID in a hit list only means "no hit" if that ID was actually in
    the original search. Test searched all raw CongLab; val searched the rows
    left after test filtering and valid-contact filtering.
    """
    from .cropping import valid_contact_counts

    output = Path(output)
    destination = output / f'legacy_{kind}_sequence_decisions.parquet'
    receipt = output / f'legacy_{kind}_sequence_decisions.json'
    if destination.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if record['lock_sha256'] != LOCK_SHA256 or file_sha256(destination) != record['sha256']:
            raise ValueError('Legacy sequence decision cache changed')
        return pq.read_table(destination)
    caches = {name: original_record(kind, name) for name in ('test', 'val')}
    rule = 'both_coverage_50' if kind == 'ppi' else 'candidate_coverage_80'
    hits = {name: set(r['hits'][rule]) for name, r in caches.items()}
    release = Path(caches['test']['inputs']['config']['release'])
    manifest = json.loads((ROOT / 'data/conglab_ecoli_v1/manifest.json').read_text())
    metadata_path = release / 'DATASET.json'
    if file_sha256(metadata_path) != manifest['input_hashes'][str(metadata_path)]:
        raise ValueError('Original source manifest changed')
    metadata = json.loads(metadata_path.read_text())
    sequences = {'test': {}, 'val': {}}
    hit_values = pa.array(sorted(hits['test']))
    columns = ['seq1_id', 'seq2_id', 'seq1', 'seq2', 'contacts', 'valid1', 'valid2']
    for entry in metadata['parquet_shards']:
        if entry['kind'] != kind:
            continue
        path = release / entry['file']
        if file_sha256(path) != entry['sha256']:
            raise ValueError('Original source shard changed: ' + str(path))
        table = pq.read_table(path, columns=columns, filters=[('structure_source', '=', 'conglab')])
        if not len(table):
            continue
        a, b = [pc.is_in(table[f'seq{s}_id'], value_set=hit_values).to_numpy() for s in (1, 2)]
        before_val = table.filter(pa.array(~(a & b)))
        before_val = before_val.filter(pa.array(valid_contact_counts(before_val) > 0))
        for category, selected in [('test', table), ('val', before_val)]:
            for side in (1, 2):
                for sid, seq in zip(selected[f'seq{side}_id'].to_pylist(), selected[f'seq{side}'].to_pylist()):
                    if sid in sequences[category] and sequences[category][sid] != seq:
                        raise ValueError('Original sequence identity changed')
                    sequences[category][sid] = seq
    hashes = {}
    for category, seqs in sequences.items():
        digest = hashlib.sha256()
        for sid, seq in sorted(seqs.items()):
            digest.update(f'>{sid}\n{seq}\n'.encode())
        hashes[category] = digest.hexdigest()
        if hashes[category] != caches[category]['inputs']['candidate']:
            raise ValueError('Cannot reproduce original candidate scope: ' + kind + '/' + category)
        if not hits[category] <= seqs.keys():
            raise ValueError('Original cache contains unknown candidate sequences')
    ids = sorted(sequences['test'])
    result = pa.table({'seq_id': ids, 'test_known': [True]*len(ids),
                       'val_known': [s in sequences['val'] for s in ids],
                       'test_hit': [s in hits['test'] for s in ids],
                       'val_hit': [s in hits['val'] for s in ids]})
    pending = destination.with_suffix('.pending')
    pq.write_table(result, pending, compression='zstd')
    pending.replace(destination)
    receipt.write_text(json.dumps(dict(lock_sha256=LOCK_SHA256, sha256=file_sha256(destination),
                                      candidate_fasta_hashes=hashes,
                                      known_counts={k: len(v) for k, v in sequences.items()}), indent=2)+'\n')
    return result
