"""Audit CongLab original IDs and alternative published E.coli split rules.

Read-only source data; temporary MMseqs databases are removed automatically.
The final JSON files contain reproducible source hashes, counts and matched IDs.
They are an audit, not a claim to possess the authors' private processed split.
"""
import argparse
import csv
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
import pyarrow.parquet as pq
from flashppi.ddi_common import sha256, write_json
from prepare_conglab_mixture import sequence_dictionary


def unordered_origin(value):
    return tuple(sorted(value.split(':')))


def matched_sides(table, hits):
    values = pa.array(sorted(hits), type=pa.string())
    return tuple(pc.is_in(table[f'seq{s}_id'], value_set=values).to_numpy() for s in (1, 2))


def classify_hit(numbers):
    """MMseqs query is held-out; target is the candidate protein/domain."""
    ni, length, qs, qe, ql, ts, te, tl = map(int, numbers)
    if ni <= 0 or length <= 0:
        raise ValueError('Alignment backtrace is required for exact identity')
    identity = ni * 10 > length * 3
    qc, tc = abs(qe-qs)+1, abs(te-ts)+1
    return (identity and qc*2 > ql and tc*2 > tl,
            identity and tc*5 >= tl*4, identity and tc*2 > tl)


def homology(sequences, references, config, output, name):
    """Search references against indexed candidates; audit both coverage rules."""
    started = time.monotonic()
    exact = {sid for sid, seq in sequences.items() if seq in references}
    result = {'both_coverage_50': set(exact), 'candidate_coverage_80': set(exact),
              'candidate_coverage_50': set(exact)}
    with tempfile.TemporaryDirectory(prefix=name+'_', dir=output) as temp:
        tmp = Path(temp)
        candidate, reference = tmp/'candidate.fa', tmp/'reference.fa'
        for path, records in ((candidate, sorted(sequences.items())),
                              (reference, [(f'r{i}', s) for i,s in enumerate(sorted(references))])):
            with path.open('w') as handle:
                for sid, seq in records:
                    handle.write(f'>{sid}\n{seq}\n')
        inputs = dict(candidate=sha256(candidate), reference=sha256(reference), config=config,
                      classification_version=2)
        cache = output/(name+'_hits.json')
        if cache.exists():
            old = json.loads(cache.read_text())
            if old['inputs'] == inputs:
                return {k:set(v) for k,v in old['hits'].items()}, old
        hits = tmp/'hits.tsv'
        command = [config['mmseqs'], 'easy-search', str(reference), str(candidate), str(hits), str(tmp/'mmseqs'),
                   '--min-seq-id', '0.3', '-c', '0.5', '--cov-mode', '1', '--alignment-mode', '3',
                   '--seq-id-mode', '0', '-a', '1', '-s', '7.5', '-e', '1000000',
                   '--max-seqs', str(len(sequences)), '--max-accept', '2147483647',
                   '--max-rejected', '2147483647', '--mask', '0', '--shuffle', '0',
                   '--threads', str(config['threads']), '--split-memory-limit', '16G',
                   '--format-output', 'query,target,nident,alnlen,qstart,qend,qlen,tstart,tend,tlen']
        print('SEARCH', name, len(references), 'references ->', len(sequences), 'candidates', flush=True)
        with (output/(name+'_mmseqs.log')).open('w') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        rows = 0
        with hits.open() as handle:
            for line in handle:
                _, sid, *numbers = line.rstrip().split('\t')
                if sid not in sequences:
                    raise ValueError('Unknown candidate ID')
                for key, accepted in zip(result, classify_hit(numbers)):
                    if accepted:
                        result[key].add(sid)
                rows += 1
        record = dict(inputs=inputs, hits={k:sorted(v) for k,v in result.items()},
                      command=[x.replace(temp, '<TEMP>') for x in command],
                      alignments=rows, exact_sequences=len(exact), seconds=time.monotonic()-started,
                      raw_hits_sha256=sha256(hits), search_direction='reference_to_candidate')
        write_json(cache, record)
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output, release = Path(config['output']), Path(config['release'])
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((release/'DATASET.json').read_text())
    test = pq.read_table(config['test'], columns=['seq1','seq2'])
    refs = set(test['seq1'].to_pylist()) | set(test['seq2'].to_pylist())
    report = dict(state='running', config=config, test_sha256=sha256(config['test']),
                  release_manifest_sha256=sha256(release/'DATASET.json'),
                  test_unique_sequences=len(refs), kinds={})
    write_json(output/'audit.json', report)
    columns = ['original_id','seq1_id','seq2_id','seq1','seq2','structure_source','source']
    oldval = pq.read_table('data/ppi_finetune_validation_v1/interaction.parquet').to_pylist()
    oldpairs = {tuple(sorted((r['seq1'],r['seq2']))) for r in oldval if r['label'] == 1}
    for kind in ('ppi','ddi'):
        parts = []
        for entry in metadata['parquet_shards']:
            if entry['kind'] != kind:
                continue
            part = pq.read_table(release/entry['file'], columns=columns,
                                 filters=[('structure_source','=','conglab')])
            if len(part):
                parts.append(part)
        table = pa.concat_tables(parts)
        del parts
        original = Path(config['original_metadata'])/kind/'posi_all.csv'
        with original.open() as handle:
            reader = csv.DictReader(handle)
            key = 'CHAIN1:CHAIN2' if kind == 'ppi' else 'PAIRID'
            original_ids = [unordered_origin(r[key]) for r in reader]
        actual = [unordered_origin(x) for x in table['original_id'].to_pylist()]
        if len(actual) != len(set(actual)) or set(actual) != set(original_ids):
            raise ValueError('Release does not exactly cover original positive IDs: '+kind)
        seqs = sequence_dictionary(table)
        with (original.parent/'posi_nohomo.csv').open() as handle:
            nohuman_ids = {unordered_origin(r[key]) for r in csv.DictReader(handle)}
        nohuman = np.array([x in nohuman_ids for x in actual])
        result, record = homology(seqs, refs, config, output, kind+'_ecoli')
        stats = dict(original_records=len(original_ids), release_records=len(table),
                     missing_original_ids=0, extra_original_ids=0,
                     original_metadata_sha256=sha256(original), unique_sequences=len(seqs),
                     original_nohuman_records=len(nohuman_ids),
                     rules={}, search_seconds=record['seconds'])
        for key, hits in result.items():
            a, b = matched_sides(table, hits)
            stats['rules'][key] = dict(hit_sequences=len(hits),
                retained_either_chain_exclusion=int((~(a|b)).sum()),
                retained_both_chains_exclusion=int((~(a&b)).sum()),
                nohuman_retained_either_chain_exclusion=int((nohuman & ~(a|b)).sum()),
                nohuman_retained_both_chains_exclusion=int((nohuman & ~(a&b)).sum()))
        pairs = set(zip(table['seq1'].to_pylist(), table['seq2'].to_pylist()))
        stats['old_string_mapped_val_pair_overlap'] = sum((a,b) in pairs or (b,a) in pairs for a,b in oldpairs)
        report['kinds'][kind] = stats
        write_json(output/'audit.json', report)
        print(kind, json.dumps(stats), flush=True)
        del table, seqs, pairs
    report['state'] = 'complete'
    report['script_sha256'] = sha256(__file__)
    report['limitations'] = ['Original-source membership verified; author processed IDs unavailable',
        'Paper main text says both chains; Methods says all matching sequences',
        'MMseqs is heuristic; this audit searches reference to candidate, not an exhaustive alignment']
    write_json(output/'audit.json', report)


if __name__ == '__main__':
    main()
