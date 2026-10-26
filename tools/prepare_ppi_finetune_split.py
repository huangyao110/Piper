#!/usr/bin/env python3
"""Build the pinned FlashPPI PPI-only train/validation split in one invocation.

Run with the existing esmfold2 Python environment from any working directory:
  python tools/prepare_ppi_finetune_split.py
  python tools/prepare_ppi_finetune_split.py --self-test

Scientific parameters are frozen in POLICY, not exposed as CLI overrides.
Only final train.parquet, validation.parquet, assignments.parquet,
sequence_clusters.parquet and manifest.json are retained in --output.
The official test stays at its original path, unchanged. MMseqs2 working files
exist only inside automatically cleaned temporary directories on /dev/shm.
SIGINT/SIGTERM are handled; an uncatchable SIGKILL can leave a temporary folder.

Validation contains POSITIVES ONLY. Construction of a fixed, STRING-filtered
negative validation benchmark is a separate training-preparation task. Do not
calculate binary AP on this positive-only file.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/zlab/lexar4t/ppidb/distributions/ppidb-training-20261009_101259')
TEST = ROOT / 'artifacts/flashppi_official/ecoli_pdb_benchmark/data/train-00000-of-00001.parquet'
SOURCE_MANIFEST_SHA = '74e019800189b2634bfbfc057aa4ff390492bc226a0a2b4a3f2fb7801b925dca'
TEST_SHA = '07dace691cac67561d264fbbcd8deb862ad6356998b129ae57dee7d983de7f6c'
MMSEQS_SHA = 'd6d1bf1b50c095eb0cb66f685a7fb1d9b8b69497e1e88416fd48f28e313cc773'
STATES = ['train', 'excluded_test_homology', 'excluded_validation_homology', 'validation']


@dataclass(frozen=True)
class Policy:
    version: str = 'ppi_split_v1'
    seed: int = 20261010
    validation_positives: int = 500
    exclusion_identity: float = 0.30
    exclusion_coverage: float = 0.50
    exclusion_comparison: str = 'strictly greater than both thresholds; either endpoint excludes pair'
    coverage_mode: int = 0
    coverage_definition: str = 'both sequence alignment spans / respective full lengths'
    validation_identity: float = 0.30
    validation_coverage: float = 0.80
    sampling_identity: float = 0.70
    sampling_coverage: float = 0.70
    cluster_mode: int = 0
    cluster_reassign: int = 1
    sensitivity: float = 7.5
    alignment_mode: int = 3
    seq_id_mode: int = 0
    search_evalue: float = 1000000.0
    clustering_evalue: float = 0.001
    threads: int = 8
    mmseqs_version: str = '18.8cc5c'
    validation_selection: str = 'distinct-sequence positive pairs; unique unordered 30/80 cluster pairs; source-proportional quotas; seeded SHA256 ordering'
    representative_selection: str = 'prefer fully_valid, then lexicographically smallest pair_id'


POLICY = Policy()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def fasta(path):
    records, header, parts = {}, None, []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    if header in records:
                        raise ValueError(f'Duplicate FASTA ID: {header}')
                    records[header] = ''.join(parts)
                header, parts = line[1:].split()[0], []
            else:
                parts.append(line)
    if header is not None:
        if header in records:
            raise ValueError(f'Duplicate FASTA ID: {header}')
        records[header] = ''.join(parts)
    if not records or any(not s for s in records.values()):
        raise ValueError('Empty FASTA or empty sequence')
    return records


def write_fasta(path, records):
    with Path(path).open('w') as handle:
        for identifier, sequence in records:
            handle.write(f'>{identifier}\n{sequence}\n')


def rank_key(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def hits_threshold(nident, alnlen, qs, qe, qlen, ts, te, tlen):
    # Integer comparisons avoid rounded fident/qcov output at 30% and 50%.
    return (nident * 10 > alnlen * 3 and
            (abs(qe - qs) + 1) * 2 > qlen and
            (abs(te - ts) + 1) * 2 > tlen)


def quotas(counts, total):
    denominator = sum(counts.values())
    if denominator < total:
        raise ValueError(f'Only {denominator} nonredundant validation candidates; need {total}')
    result = {key: total * count // denominator for key, count in counts.items()}
    order = sorted(counts, key=lambda key: (-(total * counts[key] % denominator), key))
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    return result


class MMseqs:
    def __init__(self, executable, temporary, policy):
        self.executable, self.temporary, self.policy = str(executable), Path(temporary), policy
        self.commands = []

    def run(self, args):
        command = [self.executable, *map(str, args)]
        self.commands.append([part.replace(str(self.temporary), '<RAM_TEMP>') for part in command])
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        try:
            output, _ = process.communicate()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
        if process.returncode:
            raise RuntimeError(f'MMseqs2 failed ({process.returncode}): {command}\n{output[-12000:]}')

    @contextmanager
    def stage(self, name):
        with tempfile.TemporaryDirectory(prefix=name + '_', dir=self.temporary) as directory:
            yield Path(directory)

    def exclude(self, sequence_indices, sequences, references, name):
        """Union both search directions and exact matches; return excluded sequence IDs."""
        indices = sorted(sequence_indices)
        references = sorted(set(references))
        reference_set = set(references)
        exact = {i for i in indices if sequences[i] in reference_set}
        excluded = set(exact)
        stats = {'candidate_sequences': len(indices), 'reference_sequences': len(references),
                 'exact_matches': len(exact), 'directions': {}}
        if not indices or not references:
            return excluded, stats
        with self.stage(name) as work:
            candidate = work / 'candidate.fasta'
            reference = work / 'reference.fasta'
            write_fasta(candidate, ((f's{i}', sequences[i]) for i in indices))
            write_fasta(reference, ((f'r{i}', s) for i, s in enumerate(references)))
            for direction, query, target, target_count in [
                ('candidate_to_reference', candidate, reference, len(references)),
                ('reference_to_candidate', reference, candidate, len(indices)),
            ]:
                log(f'{name}: {direction}, {len(indices)} candidate sequences')
                hits = work / f'{direction}.tsv'
                self.run(['easy-search', query, target, hits, work / direction,
                          '--min-seq-id', self.policy.exclusion_identity,
                          '-c', self.policy.exclusion_coverage, '--cov-mode', 0,
                          '--alignment-mode', 3, '--seq-id-mode', 0, '-a', 1,
                          '-s', self.policy.sensitivity, '-e', self.policy.search_evalue,
                          '--max-seqs', target_count, '--max-accept', 2147483647,
                          '--max-rejected', 2147483647, '--mask', 0, '--shuffle', 0,
                          '--threads', self.policy.threads, '--split-memory-limit', '24G',
                          '--format-output', 'query,target,nident,alnlen,qstart,qend,qlen,tstart,tend,tlen',
                          '--remove-tmp-files', 1, '-v', 1])
                qualified, seen = 0, set()
                with hits.open() as handle:
                    for line in handle:
                        q, t, *numbers = line.rstrip().split('\t')
                        numbers = list(map(int, numbers))
                        if numbers[0] <= 0:
                            raise ValueError('MMseqs2 did not supply nident; alignment backtrace is required')
                        if hits_threshold(*numbers):
                            candidate_id = q if direction == 'candidate_to_reference' else t
                            seen.add(int(candidate_id[1:]))
                            qualified += 1
                excluded.update(seen)
                stats['directions'][direction] = {'qualifying_alignments': qualified, 'matched_sequences': len(seen)}
        stats['excluded_sequences'] = len(excluded)
        return excluded, stats

    def cluster(self, sequence_indices, sequences, identity, coverage, name):
        indices = sorted(sequence_indices)
        if not indices:
            return {}
        with self.stage(name) as work:
            source, prefix = work / 'sequences.fasta', work / 'clusters'
            write_fasta(source, ((f's{i}', sequences[i]) for i in indices))
            log(f'{name}: {len(indices)} sequences, identity={identity}, coverage={coverage}')
            self.run(['easy-cluster', source, prefix, work / 'work',
                      '--min-seq-id', identity, '-c', coverage, '--cov-mode', 0,
                      '--alignment-mode', 3, '--seq-id-mode', 0,
                      '--cluster-mode', self.policy.cluster_mode,
                      '--cluster-reassign', self.policy.cluster_reassign,
                      '-s', self.policy.sensitivity, '-e', self.policy.clustering_evalue,
                      '--shuffle', 0, '--threads', self.policy.threads,
                      '--split-memory-limit', '24G', '--remove-tmp-files', 1, '-v', 1])
            representatives = {}
            with Path(str(prefix) + '_cluster.tsv').open() as handle:
                for line in handle:
                    rep, member = line.rstrip().split('\t')
                    member = int(member[1:])
                    if member in representatives:
                        raise ValueError('MMseqs2 assigned a sequence more than once')
                    representatives[member] = int(rep[1:])
            if set(representatives) != set(indices):
                raise ValueError('MMseqs2 cluster assignments do not cover input exactly')
            ordered = {rep: i for i, rep in enumerate(sorted(set(representatives.values())))}
            return {member: ordered[rep] for member, rep in representatives.items()}


def prepare(source, test, output, executable, policy=POLICY, production=True):
    source, test, output = Path(source).resolve(), Path(test).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output == test.parent or test.parent in output.parents:
        raise ValueError('Output must be outside the protected input directories')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    version = subprocess.check_output([str(executable), 'version'], text=True).strip()
    if version != policy.mmseqs_version:
        raise ValueError(f'Require MMseqs2 {policy.mmseqs_version}, got {version}')
    inputs = {'dataset_manifest_sha256': digest(source / 'DATASET.json'), 'test_sha256': digest(test),
              'fasta_sha256': digest(source / 'ppi_seqs.fasta'), 'mmseqs_binary_sha256': digest(executable)}
    if production:
        assert inputs['dataset_manifest_sha256'] == SOURCE_MANIFEST_SHA, 'Unexpected source release'
        assert inputs['test_sha256'] == TEST_SHA, 'Official test changed'
        assert inputs['mmseqs_binary_sha256'] == MMSEQS_SHA, 'Unexpected MMseqs2 binary build'
    info = json.loads((source / 'DATASET.json').read_text())
    assert info['state'] == 'complete'
    shards = [s for s in info['parquet_shards'] if s['kind'] == 'ppi']
    for item in shards:
        log(f'Checking immutable input {item["file"]}')
        if digest(source / item['file']) != item['sha256']:
            raise ValueError(f'Input checksum failed: {item["file"]}')
    expected_fasta = next(f['sha256'] for f in info['files'] if f['file'] == 'ppi_seqs.fasta')
    assert inputs['fasta_sha256'] == expected_fasta
    records = fasta(source / 'ppi_seqs.fasta')
    names = sorted(records)
    sequences = [records[n] for n in names]
    assert len(set(sequences)) == len(sequences), 'FASTA is not sequence-deduplicated'
    name_to_index = {n: i for i, n in enumerate(names)}
    total = sum(s['records'] for s in shards)
    endpoint_a = np.empty(total, dtype=np.int32)
    endpoint_b = np.empty(total, dtype=np.int32)
    source_codes = np.empty(total, dtype=np.int16)
    source_names = []
    source_to_code = {}
    representatives = {}
    offset = 0
    # Read only compact metadata for selection; original contacts/masks are copied later.
    columns = ['pair_id', 'seq1_id', 'seq2_id', 'source', 'fully_valid', 'contacts']
    for shard in shards:
        count = 0
        for batch in pq.ParquetFile(source / shard['file']).iter_batches(batch_size=16384, columns=columns):
            metadata = batch.select(['pair_id', 'seq1_id', 'seq2_id', 'source', 'fully_valid']).to_pydict()
            contact_counts = pc.list_value_length(batch.column('contacts')).to_numpy()
            for j, pair_id in enumerate(metadata['pair_id']):
                a, b = name_to_index[metadata['seq1_id'][j]], name_to_index[metadata['seq2_id'][j]]
                endpoint_a[offset], endpoint_b[offset] = a, b
                label = metadata['source'][j]
                if label not in source_to_code:
                    source_to_code[label] = len(source_names)
                    source_names.append(label)
                code = source_to_code[label]
                source_codes[offset] = code
                # Validation mirrors the heteromeric official benchmark. Train retains homomers.
                if a != b and contact_counts[j] > 0:
                    pair = tuple(sorted((a, b)))
                    quality = (not bool(metadata['fully_valid'][j]), pair_id)
                    old = representatives.get(pair)
                    if old is None or quality < old[0]:
                        representatives[pair] = (quality, offset, code)
                offset += 1
            count += len(batch)
        assert count == shard['records']
    assert offset == total
    log(f'Indexed {total} PPI structures, {len(sequences)} sequences, {len(representatives)} eligible distinct validation pairs')
    test_table = pq.read_table(test)
    test_references = set(test_table['seq1'].to_pylist()) | set(test_table['seq2'].to_pylist())
    if production:
        assert len(test_table) == 65650 and Counter(test_table['label'].to_pylist()) == {0: 65000, 1: 650}
    state = np.zeros(total, dtype=np.int8)
    started = time.time()
    temporary_root = Path('/dev/shm')
    if shutil.disk_usage(temporary_root).free < 8 * 1024**3:
        raise RuntimeError('Need at least 8 GiB free RAM-backed /dev/shm; no disk fallback is used')
    with tempfile.TemporaryDirectory(prefix='flashppi_split_', dir=temporary_root) as temporary:
        mmseqs = MMseqs(executable, temporary, policy)
        test_hits, test_stats = mmseqs.exclude(range(len(sequences)), sequences, test_references, 'exclude_test')
        mask = np.zeros(len(sequences), dtype=bool)
        mask[list(test_hits)] = True
        state[mask[endpoint_a] | mask[endpoint_b]] = 1
        remaining = np.flatnonzero(state == 0)
        remaining_sequences = set(endpoint_a[remaining]) | set(endpoint_b[remaining])
        log(f'After test exclusion: {len(remaining)} structures, {len(remaining_sequences)} sequences')
        derep = mmseqs.cluster(remaining_sequences, sequences, policy.validation_identity,
                              policy.validation_coverage, 'validation_dereplication')
        groups = {}
        for pair, (quality, row, code) in representatives.items():
            if state[row] != 0:
                continue
            cluster_pair = tuple(sorted((derep[pair[0]], derep[pair[1]])))
            key = rank_key('|'.join(names[i] for i in pair), policy.seed)
            candidate = (key, row, code, pair)
            if cluster_pair not in groups or candidate < groups[cluster_pair]:
                groups[cluster_pair] = candidate
        by_source = defaultdict(list)
        for candidate in groups.values():
            by_source[source_names[candidate[2]]].append(candidate)
        selected_quotas = quotas({k: len(v) for k, v in by_source.items()}, policy.validation_positives)
        selected = []
        for label in sorted(by_source):
            selected.extend(sorted(by_source[label])[:selected_quotas[label]])
        validation_rows = sorted(candidate[1] for candidate in selected)
        validation_sequences = set(endpoint_a[validation_rows]) | set(endpoint_b[validation_rows])
        log(f'Selected {len(validation_rows)} nonredundant validation pairs: {selected_quotas}')
        val_hits, val_stats = mmseqs.exclude(remaining_sequences, sequences,
                                           [sequences[i] for i in sorted(validation_sequences)], 'exclude_validation')
        mask[:] = False
        mask[list(val_hits)] = True
        state[(state == 0) & (mask[endpoint_a] | mask[endpoint_b])] = 2
        state[validation_rows] = 3
        training_rows = np.flatnonzero(state == 0)
        training_sequences = set(endpoint_a[training_rows]) | set(endpoint_b[training_rows])
        if not training_sequences:
            raise ValueError('No training sequences remain after homology exclusion')
        assert not (training_sequences & validation_sequences)
        assert not ((training_sequences | validation_sequences) & test_hits)
        assert not (training_sequences & val_hits)
        train_clusters = mmseqs.cluster(training_sequences, sequences, policy.sampling_identity,
                                       policy.sampling_coverage, 'train_sampling_clusters')
        val_clusters = mmseqs.cluster(validation_sequences, sequences, policy.sampling_identity,
                                     policy.sampling_coverage, 'validation_sampling_clusters')
        cluster_offset = len(set(train_clusters.values()))
        clusters = np.full(len(sequences), -1, dtype=np.int64)
        for i, cluster in train_clusters.items():
            clusters[i] = cluster
        for i, cluster in val_clusters.items():
            clusters[i] = cluster + cluster_offset
        commands = mmseqs.commands
    # No MMseqs2 work directory survives this point.
    assert not Path(temporary).exists()
    output.mkdir(parents=True, exist_ok=False)
    try:
        writers = {}
        assignment_writer = None
        offset = 0
        try:
            for shard in shards:
                local_offset = 0
                for batch in pq.ParquetFile(source / shard['file']).iter_batches(batch_size=8192):
                    table = pa.Table.from_batches([batch])
                    count = len(table)
                    stop = offset + count
                    values = state[offset:stop]
                    a, b = endpoint_a[offset:stop], endpoint_b[offset:stop]
                    # These are the only two original columns replaced. Masks/labels are untouched.
                    for col, mapped in [('seq1_cluster', clusters[a]), ('seq2_cluster', clusters[b])]:
                        pos = table.schema.get_field_index(col)
                        if pos < 0:
                            table = table.append_column(col, pa.array(mapped))
                        else:
                            table = table.set_column(pos, col, pa.array(mapped))
                    for code, label in [(0, 'train'), (3, 'validation')]:
                        part = table.filter(pa.array(values == code))
                        if len(part):
                            if label not in writers:
                                writers[label] = pq.ParquetWriter(output / f'{label}.parquet', part.schema, compression='zstd')
                            writers[label].write_table(part)
                    assignment = pa.table({
                        'pair_id': table['pair_id'], 'source': table['source'],
                        'source_shard': [shard['file']] * count,
                        'source_row': np.arange(local_offset, local_offset + count, dtype=np.int64),
                        'split': [STATES[int(v)] for v in values],
                    })
                    if assignment_writer is None:
                        assignment_writer = pq.ParquetWriter(output / 'assignments.parquet', assignment.schema, compression='zstd')
                    assignment_writer.write_table(assignment)
                    offset, local_offset = stop, local_offset + count
        finally:
            for writer in writers.values():
                writer.close()
            if assignment_writer is not None:
                assignment_writer.close()
        used = sorted(training_sequences | validation_sequences)
        pq.write_table(pa.table({'sequence_id': [names[i] for i in used],
                                 'cluster_id': clusters[used],
                                 'split': ['train' if i in training_sequences else 'validation' for i in used]}),
                       output / 'sequence_clusters.parquet', compression='zstd')
        counts = dict(Counter(STATES[int(v)] for v in state))
        source_stats = {}
        pair_keys = (np.minimum(endpoint_a, endpoint_b).astype(np.int64) * len(sequences)
                     + np.maximum(endpoint_a, endpoint_b))
        unique_pair_stats = {label: int(len(np.unique(pair_keys[state == code])))
                             for code, label in enumerate(STATES)}
        unique_pairs_by_source = {}
        for code, label in enumerate(source_names):
            source_stats[label] = dict(Counter(STATES[int(v)] for v in state[source_codes == code]))
            unique_pairs_by_source[label] = {
                split: int(len(np.unique(pair_keys[(source_codes == code) & (state == state_code)])))
                for state_code, split in enumerate(STATES)}
        assert counts.get('validation') == policy.validation_positives
        assert sum(counts.values()) == total
        for split in ('train', 'validation'):
            assert pq.ParquetFile(output / f'{split}.parquet').metadata.num_rows == counts[split]
        assert digest(test) == inputs['test_sha256']
        assert digest(source / 'DATASET.json') == inputs['dataset_manifest_sha256']
        outputs = {f.name: {'sha256': digest(f), 'bytes': f.stat().st_size} for f in sorted(output.glob('*.parquet'))}
        manifest = {
            'state': 'complete', 'policy': asdict(policy), 'production': production,
            'source': str(source), 'official_test': str(test), 'inputs': inputs,
            'script_sha256': digest(Path(__file__)), 'python': sys.version.split()[0],
            'numpy': np.__version__, 'pyarrow': pa.__version__, 'mmseqs_version': version,
            'counts': counts, 'sources': source_stats, 'validation_source_quotas': selected_quotas,
            'unique_sequence_pairs': unique_pair_stats, 'unique_sequence_pairs_by_source': unique_pairs_by_source,
            'sequences': {'input': len(sequences), 'train': len(training_sequences), 'validation': len(validation_sequences)},
            'sampling_clusters': {'train': cluster_offset, 'validation': len(set(val_clusters.values()))},
            'test_exclusion': test_stats, 'validation_exclusion': val_stats,
            'validation_pair_ids': [representatives[candidate[3]][0][1] for candidate in sorted(selected)],
            'mmseqs_commands': commands, 'outputs': outputs,
            'elapsed_seconds': time.time() - started, 'temporary_files_retained': False,
            'limitations': [
                'Methods interpretation: either endpoint homologous to held-out sequences removes a pair; article prose also contains a both-endpoints wording.',
                'Paper does not specify coverage mode or every MMseqs2 option; bidirectional coverage and all remaining parameters are fixed here.',
                'Sensitive two-direction MMseqs2 search plus exact matches is a reproducible heuristic, not an exhaustive proof of absence of all possible alignments.',
                'Validation is held out from this fine-tuning corpus; exclusion from original checkpoint training is not established.',
                'validation.parquet contains positive structures only, not a binary-classification benchmark; STRING-filtered fixed negative pairs remain to be prepared.',
                'Training retains homomers and multiple structures per sequence pair; validation selects distinct-sequence, contact-bearing representatives.',
            ],
        }
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        log(f'COMPLETE {output}: {counts}')
        return manifest
    except BaseException:
        # This directory was created exclusively by this invocation.
        shutil.rmtree(output)
        raise


def self_test(executable):
    """Real-MMseqs2 fixture checks leakage, repeatability, strict cutoffs and masks."""
    assert not hits_threshold(30, 100, 1, 100, 100, 1, 100, 100)
    assert not hits_threshold(60, 100, 1, 50, 100, 1, 100, 100)
    assert hits_threshold(31, 100, 1, 51, 100, 1, 51, 100)
    assert quotas({'a': 1, 'b': 2}, 2) == {'a': 1, 'b': 1}
    policy = replace(POLICY, validation_positives=2)
    rng = np.random.default_rng(12)
    alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    seqs = [''.join(rng.choice(alphabet, 120)) for _ in range(16)]
    # seq 1 is homologous but not identical to the test sequence 0.
    seqs[1] = seqs[0][:100] + seqs[1][100:]
    before = set(Path('/dev/shm').glob('flashppi_split_*'))
    with tempfile.TemporaryDirectory(prefix='flashppi_fixture_', dir='/dev/shm') as temporary:
        base = Path(temporary)
        source = base / 'input'
        (source / 'parquet/ppi').mkdir(parents=True)
        write_fasta(source / 'ppi_seqs.fasta', ((f'seq_{i:03}', s) for i, s in enumerate(seqs)))
        rows = []
        pairs = [(0, 2), (1, 3)]
        for a, b in [(4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]:
            pairs.extend([(a, b), (b, a)])
        for i, (a, b) in enumerate(pairs):
            rows.append(dict(pair_id=f'pair_{i:03}', seq1_id=f'seq_{a:03}', seq2_id=f'seq_{b:03}',
                             seq1=seqs[a], seq2=seqs[b], source='a' if i % 2 else 'b',
                             len1=120, len2=120, fully_valid=False,
                             contacts=[[0, 1]], valid1=[True] * 119 + [False], valid2=[True] * 120,
                             seq1_cluster=-1, seq2_cluster=-1))
        shard = source / 'parquet/ppi/part-00000.parquet'
        pq.write_table(pa.Table.from_pylist(rows), shard)
        (source / 'DATASET.json').write_text(json.dumps({'state': 'complete',
            'parquet_shards': [{'file': 'parquet/ppi/part-00000.parquet', 'kind': 'ppi', 'records': len(rows), 'sha256': digest(shard)}],
            'files': [{'file': 'ppi_seqs.fasta', 'sha256': digest(source / 'ppi_seqs.fasta')}]}))
        (base / 'test').mkdir()
        test = base / 'test/test.parquet'
        pq.write_table(pa.table({'seq1': [seqs[0]], 'seq2': [seqs[0]], 'label': [1]}), test)
        manifests = [prepare(source, test, base / f'result_{i}', executable, policy, production=False) for i in range(2)]
        assert manifests[0]['outputs'] == manifests[1]['outputs'], 'Repeated run is not byte-reproducible'
        assignments = pq.read_table(base / 'result_0/assignments.parquet').to_pylist()
        states = {r['pair_id']: r['split'] for r in assignments}
        assert states['pair_000'] == states['pair_001'] == 'excluded_test_homology'
        assert Counter(states.values())['excluded_validation_homology'] >= 2
        train_pairs = {tuple(sorted((r['seq1'], r['seq2']))) for r in rows if states[r['pair_id']] == 'train'}
        val_pairs = {tuple(sorted((r['seq1'], r['seq2']))) for r in rows if states[r['pair_id']] == 'validation'}
        assert not train_pairs & val_pairs
        assert not (states['pair_002'] == 'validation' and states['pair_003'] == 'train')
        assert not (states['pair_003'] == 'validation' and states['pair_002'] == 'train')
        for split in ('train', 'validation'):
            for row in pq.read_table(base / f'result_0/{split}.parquet').to_pylist():
                original = next(r for r in rows if r['pair_id'] == row['pair_id'])
                for key in original:
                    if key not in ('seq1_cluster', 'seq2_cluster'):
                        assert row[key] == original[key], (row['pair_id'], key)
        try:
            prepare(source, test, base / 'result_0', executable, policy, production=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output was overwritten')
    assert set(Path('/dev/shm').glob('flashppi_split_*')) == before
    log('SELF-TEST PASSED: strict boundaries, homolog exclusion, duplicate isolation, unchanged masks/labels, repeatability, cleanup, overwrite protection')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/ppi_finetune_split_v1')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--print-policy', action='store_true')
    args = parser.parse_args()
    if args.print_policy:
        print(json.dumps(asdict(POLICY), indent=2))
        return
    executable = shutil.which('mmseqs')
    if executable is None:
        raise RuntimeError('MMseqs2 18.8cc5c is required on PATH')
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    if args.self_test:
        self_test(executable)
    else:
        prepare(SOURCE, TEST, args.output, executable)


if __name__ == '__main__':
    main()
