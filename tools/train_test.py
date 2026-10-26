"""train_test.py: build or check locked index-only data splits.

Run from the repository root. Original releases and frozen evaluation Parquets
are read-only. Only metadata, indexes, audits, and disposable build scratch are
written to the new output directory. The GPU and training process are untouched.
Use --mode build to construct and accept a split, or --mode check to independently
recheck a completed split without changing its members. Each protocol has its
own locked source scope and filtering rules.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

from flashppi.ddi_common import write_json
from flashppi.indexed_data import file_sha256, sequence_id, contact_fingerprint
from flashppi.data_protocol import (validate_config, verify_baseline, original_record,
    reference_fasta, locked_command, verify_search_log, legacy_decisions,
    restore_known_decisions, LOCK_PATH, STRICT_PROTOCOL_ID, STRICT_LOCK_PATH,
    RF2_PROTOCOL_ID, RF2_LOCK_PATH, FRESH_PROTOCOL_IDS,
    read_strict_lock, validate_strict_config, verify_strict_inputs, strict_command,
    strict_hit, strict_pair_decision, verify_strict_search_log)
from audit_conglab_ecoli import classify_hit


def quoted(value):
    return "'" + str(value).replace("'", "''") + "'"


def payload_buffers(batch):
    """Flatten one source batch without making Python lists of contact pairs."""
    def col(name):
        return batch.column(batch.schema.get_field_index(name))
    outer = col('contacts')
    contact_offsets = outer.offsets.to_numpy().astype(np.int64)
    contact_offsets -= contact_offsets[0]
    xy = outer.flatten().flatten().to_numpy(zero_copy_only=False).reshape(-1, 2)
    mask_arrays = [col('valid1'), col('valid2')]
    offsets, masks = [], []
    for a in mask_arrays:
        off = a.offsets.to_numpy().astype(np.int64)
        off -= off[0]
        offsets.append(off)
        masks.append(a.flatten().to_numpy(zero_copy_only=False))
    n = batch.num_rows
    owners = np.repeat(np.arange(n), np.diff(contact_offsets))
    lengths = [np.diff(o) for o in offsets]
    for side in (1, 2):
        if f'seq{side}' in batch.schema.names:
            expected = np.array([len(s) for s in col(f'seq{side}').to_pylist()])
            if not np.array_equal(lengths[side-1], expected):
                raise ValueError('Source sequence and mask lengths disagree')
    if len(xy) and ((xy < 0).any() or (xy[:, 0] >= lengths[0][owners]).any()
                    or (xy[:, 1] >= lengths[1][owners]).any()):
        raise ValueError('Source contacts are outside residue-mask bounds')
    valid = masks[0][offsets[0][owners]+xy[:, 0]] & masks[1][offsets[1][owners]+xy[:, 1]]
    counts = np.bincount(owners[valid], minlength=n).astype(np.int32)
    return contact_offsets, xy, offsets, masks, counts


def hash_row(batch, i, buffers, canonical=True):
    co, xy, offsets, masks, _ = buffers
    values = {name: batch.column(batch.schema.get_field_index(name))[i].as_py()
              for name in ('seq1', 'seq2', 'seq1_id', 'seq2_id', 'kind')}
    if any(sequence_id(values[f'seq{s}']) != values[f'seq{s}_id'] for s in (1, 2)):
        raise ValueError('Sequence content does not match its ID')
    return contact_fingerprint(values['seq1'], values['seq2'],
        masks[0][offsets[0][i]:offsets[0][i+1]], masks[1][offsets[1][i]:offsets[1][i+1]],
        xy[co[i]:co[i+1]], values['kind'], canonical=canonical)


class Build:
    def __init__(self, config_file):
        self.config_file = Path(config_file).resolve()
        self.config = json.loads(self.config_file.read_text())
        c = self.config
        self.protocol = verify_baseline(c)
        if not c['preserve_all_legacy'] or c['test_exclusion'] != 'both_chains' or c['val_exclusion'] != 'either_chain':
            raise ValueError('This expansion must preserve the agreed legacy protocol')
        self.output = Path(c['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.scratch = self.output / 'build_scratch'
        self.scratch.mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.release = Path(c['release'])
        self.training = self.release / 'components/training'
        self.legacy = Path(c['legacy_directory'])
        self.meta = json.loads((self.release / 'manifest.json').read_text())
        self.base_meta = json.loads((self.training / 'DATASET.json').read_text())
        self.old_meta = json.loads((self.legacy / 'manifest.json').read_text())
        if any(m['state'] != 'complete' for m in (self.meta, self.base_meta, self.old_meta)):
            raise ValueError('Incomplete source dataset')
        identity = {'config': c, 'filter_protocol': self.protocol,
                    'release_manifest_sha256': file_sha256(self.release / 'manifest.json'),
                    'legacy_manifest_sha256': file_sha256(self.legacy / 'manifest.json')}
        path = self.output / 'build_identity.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Output already belongs to a different configuration')
        write_json(path, identity)
        self.db = duckdb.connect(str(self.scratch / 'catalog.duckdb'))
        self.db.execute('SET threads = ' + str(int(c['threads'])))
        self.db.execute('SET memory_limit = ' + quoted(c['duckdb_memory_limit']))
        self.db.execute('SET temp_directory = ' + quoted(self.scratch / 'duckdb_temp'))
        self.db.execute('SET preserve_insertion_order = false')

    def status(self, stage, **details):
        record = dict(state='building', stage=stage, elapsed_seconds=round(time.monotonic()-self.started, 2), **details)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        write_json(self.output / 'status.json', record)

    def done(self, stage):
        return (self.output / (stage + '.done.json')).exists()

    def mark(self, stage, **details):
        write_json(self.output / (stage + '.done.json'),
                   dict(stage=stage, script_sha256=file_sha256(__file__), **details))
        self.db.execute('CHECKPOINT')

    def export(self, query, path):
        path = Path(path)
        pending = path.with_name(path.name + '.pending')
        pending.unlink(missing_ok=True)
        self.db.execute('COPY (' + query + ') TO ' + quoted(pending) +
                        " (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)")
        pending.replace(path)

    def scalar(self, query):
        return self.db.execute(query).fetchone()[0]

    def table(self, name, table):
        self.db.register('_arrow_input', table)
        self.db.execute('CREATE OR REPLACE TABLE ' + name + ' AS SELECT * FROM _arrow_input')
        self.db.unregister('_arrow_input')

    def catalog(self):
        if self.done('catalog'):
            self.registry = json.loads((self.output / 'sources.json').read_text())
            return
        self.status('verify_sources_and_catalog')
        registry = dict(roots={'release': str(self.release), 'legacy': str(self.legacy),
                               'test': str(Path(self.config['test']).parent)}, shards={}, frozen_splits={})
        self.db.execute('DROP TABLE IF EXISTS catalog')
        shard_number = 0
        for entry in self.meta['metadata_shards']:
            meta_path = self.release / entry['file']
            base_path = self.training / entry['base_file']
            if file_sha256(meta_path) != entry['sha256'] or file_sha256(base_path) != entry['base_sha256']:
                raise ValueError('Release shard checksum mismatch: ' + str(base_path))
            pf = pq.ParquetFile(base_path)
            starts = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])]
            if int(starts[-1]) != entry['records']:
                raise ValueError('Metadata and original source row counts differ')
            registry['shards'][str(shard_number)] = dict(root='release',
                path=str(base_path.relative_to(self.release)), sha256=entry['base_sha256'],
                rows=entry['records'], row_group_starts=starts[:-1].tolist(),
                metadata_path=entry['file'], metadata_sha256=entry['sha256'])
            query = f"""SELECT sample_id, kind, seq1_id, seq2_id, sequence_pair_id,
                structure_source, source, seq1_cluster, seq2_cluster, sequence_cluster_scope,
                structure_cluster_id, structure_cluster_scope, len1, len2,
                {shard_number}::SMALLINT AS shard_id, file_row_number::INTEGER AS row_index
                FROM read_parquet({quoted(meta_path)}, file_row_number=true)"""
            self.db.execute(('CREATE TABLE catalog AS ' if shard_number == 0 else 'INSERT INTO catalog ') + query)
            shard_number += 1
            if shard_number % 10 == 0:
                self.status('verify_sources_and_catalog', shards_done=shard_number, shards_total=len(self.meta['metadata_shards']))
        count, unique = self.db.execute('SELECT count(*),count(DISTINCT sample_id) FROM catalog').fetchone()
        if count != unique or count != sum(self.meta['counts'].values()):
            raise ValueError('Source sample IDs are not unique or count changed')
        for name, filename in [('validation_contact', 'validation.parquet'), ('validation_interaction', 'interaction.parquet'),
                               ('test', Path(self.config['test']).name)]:
            path = Path(self.config['test']) if name == 'test' else self.legacy / filename
            expected = self.config['test_sha256'] if name == 'test' else self.old_meta['outputs'][filename]['sha256']
            if file_sha256(path) != expected:
                raise ValueError('Frozen evaluation file changed: ' + str(path))
            pf = pq.ParquetFile(path)
            starts = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])]
            registry['shards'][str(shard_number)] = dict(root='test' if name == 'test' else 'legacy',
                path=filename, sha256=expected, rows=pf.metadata.num_rows, row_group_starts=starts[:-1].tolist())
            registry['frozen_splits'][name] = shard_number
            shard_number += 1
        old_path = self.legacy / 'train.parquet'
        if file_sha256(old_path) != self.old_meta['outputs']['train.parquet']['sha256']:
            raise ValueError('Legacy training source changed')
        self.db.execute(f"""CREATE OR REPLACE TABLE legacy AS SELECT pair_id AS sample_id, kind,
            file_row_number::INTEGER AS old_row_index FROM read_parquet({quoted(old_path)}, file_row_number=true)""")
        counts = dict(self.db.execute('SELECT kind,count(*) FROM legacy GROUP BY kind').fetchall())
        if counts != self.config['expected_legacy'] or self.scalar('SELECT count(*)-count(DISTINCT sample_id) FROM legacy'):
            raise ValueError('Unexpected legacy training population')
        missing = self.scalar('SELECT count(*) FROM legacy ANTI JOIN catalog USING(sample_id)')
        wrong_source = self.scalar("SELECT count(*) FROM legacy JOIN catalog USING(sample_id) WHERE structure_source!='conglab'")
        if missing or wrong_source:
            raise ValueError('Legacy rows do not map exactly to CongLab source records')
        registry['legacy_train'] = dict(path=str(old_path), sha256=self.old_meta['outputs']['train.parquet']['sha256'], counts=counts)
        self.registry = registry
        write_json(self.output / 'sources.json', registry)
        self.mark('catalog', source_rows=count, legacy_counts=counts, missing_legacy=missing)

    def homology(self):
        refs = {}
        for name in ('test', 'validation_contact', 'validation_interaction'):
            sid = self.registry['frozen_splits'][name]
            source = self.registry['shards'][str(sid)]
            path = Path(self.registry['roots'][source['root']]) / source['path']
            t = pq.read_table(path, columns=['seq1', 'seq2'])
            category = 'test' if name == 'test' else 'val'
            refs.setdefault(category, set()).update(t['seq1'].to_pylist()+t['seq2'].to_pylist())
        reference_files = {}
        for category, sequences in refs.items():
            path = self.output / (category + '_reference.fasta')
            path.write_text(reference_fasta(sequences))
            reference_files[category] = path
        audits = {}
        for kind in ('ppi', 'ddi'):
            result_file = self.output / (kind + '_homology_hits.parquet')
            audit_file = self.output / (kind + '_homology.json')
            if result_file.exists() and audit_file.exists():
                audits[kind] = json.loads(audit_file.read_text())
                if (file_sha256(result_file) != audits[kind]['hits_sha256']
                        or audits[kind].get('filter_protocol') != self.protocol):
                    raise ValueError('Cached homology hits changed')
                continue
            fasta = self.training / (kind + '_seqs.fasta')
            expected = next(f['sha256'] for f in self.base_meta['files'] if f['file'] == fasta.name)
            if file_sha256(fasta) != expected:
                raise ValueError('Source FASTA checksum mismatch')
            candidate_count = self.scalar(f"SELECT count(DISTINCT seq_id) FROM (SELECT seq1_id AS seq_id FROM catalog WHERE kind={quoted(kind)} UNION ALL SELECT seq2_id FROM catalog WHERE kind={quoted(kind)})")
            self.status('restore_original_sequence_decisions', kind=kind)
            known = legacy_decisions(self.output, kind)
            begun = time.monotonic()
            hits = {category: {sequence_id(s) for s in sequences} for category, sequences in refs.items()}
            runs = {}
            chosen_rule = 0 if kind == 'ppi' else 1
            for category in ('test', 'val'):
                previous = original_record(kind, category)
                reference_file = reference_files[category]
                if file_sha256(reference_file) != previous['inputs']['reference']:
                    raise ValueError('Reference sequences differ from original filtering inputs')
                work = self.scratch / ('mmseqs_' + kind + '_' + category)
                work.mkdir(exist_ok=True)
                alignments = work / 'alignments.tsv'
                command = locked_command(previous, self.config['mmseqs'], reference_file, fasta,
                                         alignments, work / 'tmp', candidate_count)
                self.status('homology_search_original_k6', kind=kind, category=category,
                            candidates=candidate_count, references=len(refs[category]))
                log_path = self.output / (kind + '_' + category + '_mmseqs.log')
                with log_path.open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_search_log(log_path)
                alignment_count = 0
                with alignments.open() as handle:
                    for line in handle:
                        _, target, *numbers = line.rstrip().split('\t')
                        if classify_hit(numbers)[chosen_rule]:
                            hits[category].add(target)
                        alignment_count += 1
                raw_hits = hits[category]
                hits[category] = restore_known_decisions(raw_hits, known, category)
                archive = self.output / (kind + '_' + category + '_alignments.tsv.gz')
                with alignments.open('rb') as source, archive.open('wb') as destination:
                    with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as compressed:
                        shutil.copyfileobj(source, compressed)
                runs[category] = dict(command=command, reference_sha256=file_sha256(reference_file),
                    alignments=alignment_count, raw_alignment_sha256=file_sha256(alignments),
                    alignment_archive=archive.name, archive_sha256=file_sha256(archive),
                    restored_old_hits=len(hits[category]-raw_hits),
                    preserved_old_nonhits=len(raw_hits-hits[category]))
                shutil.rmtree(work)
            ids = sorted(hits['test'] | hits['val'])
            t = pa.table({'seq_id': ids, 'test_hit': [s in hits['test'] for s in ids],
                          'val_hit': [s in hits['val'] for s in ids]})
            pq.write_table(t, result_file, compression='zstd')
            audit = dict(kind=kind, filter_protocol=self.protocol, runs=runs, mmseqs_version=subprocess.check_output(
                [self.config['mmseqs'], 'version'], text=True).strip(), candidate_fasta_sha256=expected,
                reference_counts={k:len(v) for k,v in refs.items()},
                hit_counts={k:len(v) for k,v in hits.items()}, hits_sha256=file_sha256(result_file),
                seconds=time.monotonic()-begun, coverage_rule=self.config[kind+'_coverage'],
                identity='nident/alnlen strictly >0.30', search_direction='reference_to_candidate',
                exact_sequence_matches_included=True, known_sequence_policy='original_cached_decisions',
                limitation='Original heuristic protocol; diagnostic upgrades are not applied')
            write_json(audit_file, audit)
            audits[kind] = audit
        self.mark('homology', audits=audits)

    def candidates(self):
        if self.done('candidates'):
            return
        self.status('filter_new_candidates_and_join_cluster_namespaces')
        for kind in ('ppi', 'ddi'):
            self.db.execute(f"CREATE OR REPLACE TABLE hits_{kind} AS SELECT * FROM read_parquet({quoted(self.output / (kind + '_homology_hits.parquet'))})")
        self.db.execute("CREATE OR REPLACE TABLE hits AS SELECT *, 'ppi' AS kind FROM hits_ppi UNION ALL SELECT *, 'ddi' AS kind FROM hits_ddi")
        allowed = ','.join(quoted(s) for s in self.config['new_sources'])
        self.db.execute(f"""CREATE OR REPLACE TABLE decisions AS
            SELECT c.*, l.sample_id IS NOT NULL AS legacy_train, l.old_row_index,
              coalesce(h1.test_hit,false) AS test_hit1, coalesce(h2.test_hit,false) AS test_hit2,
              coalesce(h1.val_hit,false) AS val_hit1, coalesce(h2.val_hit,false) AS val_hit2,
              CASE WHEN l.sample_id IS NOT NULL THEN 'legacy_keep'
                   WHEN structure_source NOT IN ({allowed}) THEN 'not_legacy_not_new_source'
                   WHEN coalesce(h1.val_hit,false) OR coalesce(h2.val_hit,false) THEN 'validation_homology'
                   WHEN coalesce(h1.test_hit,false) AND coalesce(h2.test_hit,false) THEN 'test_both_chain_homology'
                   ELSE 'new_candidate' END AS decision
            FROM catalog c LEFT JOIN legacy l USING(sample_id)
            LEFT JOIN hits h1 ON c.kind=h1.kind AND c.seq1_id=h1.seq_id
            LEFT JOIN hits h2 ON c.kind=h2.kind AND c.seq2_id=h2.seq_id""")
        self.export("SELECT sample_id,kind,structure_source,decision,test_hit1,test_hit2,val_hit1,val_hit2 FROM decisions WHERE decision NOT IN ('legacy_keep','new_candidate')",
                    self.output / 'excluded_homology.index.parquet')
        cluster_info = self.cluster_candidates()
        stats = self.db.execute('SELECT kind,structure_source,decision,count(*) FROM decisions GROUP BY ALL ORDER BY ALL').fetchall()
        legacy_hits = self.db.execute("SELECT kind,count(*) FILTER(WHERE val_hit1 OR val_hit2),count(*) FILTER(WHERE test_hit1 AND test_hit2),count(*) FILTER(WHERE test_hit1 OR test_hit2) FROM candidates WHERE legacy_train GROUP BY kind").fetchall()
        if any(val or test for _, val, test, _ in legacy_hits):
            raise ValueError('Original training filter decisions changed; refusing the expansion')
        self.mark('candidates', decisions=stats, legacy_recheck=legacy_hits, **cluster_info)

    def cluster_candidates(self):
        # Scope every cluster ID before joining PPI and DDI. Identical sequence
        # IDs connect the scopes; coincident numeric IDs alone never do.
        offset = self.scalar("SELECT greatest(max(seq1_cluster),max(seq2_cluster))+1 FROM catalog WHERE kind='ppi'")
        self.db.execute(f"""CREATE OR REPLACE TABLE scoped_sequences AS
            SELECT kind,seq1_id AS seq_id,seq1_cluster+CASE WHEN kind='ddi' THEN {offset} ELSE 0 END AS cluster FROM catalog
            UNION SELECT kind,seq2_id,seq2_cluster+CASE WHEN kind='ddi' THEN {offset} ELSE 0 END FROM catalog""")
        inconsistent = self.scalar('SELECT count(*) FROM (SELECT kind,seq_id FROM scoped_sequences GROUP BY kind,seq_id HAVING count(DISTINCT cluster)>1)')
        if inconsistent or self.scalar('SELECT min(cluster) FROM scoped_sequences') < 0:
            raise ValueError('Source cluster assignments are missing or inconsistent')
        bridges = self.db.execute('SELECT min(cluster),max(cluster) FROM scoped_sequences GROUP BY seq_id HAVING count(DISTINCT kind)>1').fetchall()
        parent = {}
        def find(value):
            parent.setdefault(value, value)
            root = value
            while parent[root] != root:
                root = parent[root]
            while parent[value] != value:
                next_value = parent[value]
                parent[value] = root
                value = next_value
            return root
        for a, b in bridges:
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
        keys = sorted(parent)
        self.table('cluster_bridges', pa.table({'original': pa.array(keys, type=pa.int64()),
                                              'joint': pa.array([find(k) for k in keys], type=pa.int64())}))
        self.export('SELECT * FROM cluster_bridges ORDER BY original', self.output / 'cluster_bridges.parquet')
        self.db.execute(f"""CREATE OR REPLACE TABLE candidates AS
            WITH selected AS (
                SELECT *, seq1_cluster+CASE WHEN kind='ddi' THEN {offset} ELSE 0 END AS scoped1,
                          seq2_cluster+CASE WHEN kind='ddi' THEN {offset} ELSE 0 END AS scoped2
                FROM decisions WHERE decision IN ('legacy_keep','new_candidate'))
            SELECT s.*,coalesce(b1.joint,s.scoped1)::INTEGER AS joint_cluster1,
                       coalesce(b2.joint,s.scoped2)::INTEGER AS joint_cluster2,
                       count(*) OVER (PARTITION BY sequence_pair_id)>1 OR legacy_train AS needs_hash
            FROM selected s LEFT JOIN cluster_bridges b1 ON s.scoped1=b1.original
                            LEFT JOIN cluster_bridges b2 ON s.scoped2=b2.original""")
        return dict(ppi_cluster_offset=offset, shared_exact_sequence_bridges=len(bridges),
                    cluster_multiplier=int(self.scalar('SELECT max(cluster)+1 FROM scoped_sequences')),
                    candidate_count=self.scalar('SELECT count(*) FROM candidates'))

    def legacy_payloads(self, schema, cols):
        old_path = self.scratch / 'legacy_payload_hashes.parquet'
        if not old_path.exists():
            pending = old_path.with_suffix('.pending')
            with pq.ParquetWriter(pending, schema, compression='zstd') as writer:
                seen = 0
                for batch in pq.ParquetFile(self.legacy / 'train.parquet').iter_batches(batch_size=1024, columns=cols, use_threads=False):
                    buffers = payload_buffers(batch)
                    if np.any(buffers[-1] == 0):
                        raise ValueError('Legacy train contains empty supervision')
                    rows = [dict(sample_id=batch.column(0)[i].as_py(), supervision_hash=hash_row(batch,i,buffers),
                                 ordered_hash=hash_row(batch,i,buffers,False),valid_contacts=int(buffers[-1][i]))
                            for i in range(batch.num_rows)]
                    writer.write_table(pa.Table.from_pylist(rows,schema=schema))
                    seen += batch.num_rows
                    if seen % 65536 == 0:
                        self.status('hash_legacy_payloads', rows_done=seen, rows_total=sum(self.config['expected_legacy'].values()))
            pending.replace(old_path)
        self.db.execute('CREATE OR REPLACE TABLE legacy_payloads AS SELECT * FROM read_parquet(' + quoted(old_path) + ')')

    def contents(self):
        if self.done('contents'):
            return
        self.status('verify_legacy_payloads_and_hash_duplicate_candidates')
        content_dir = self.scratch / 'contents'
        content_dir.mkdir(exist_ok=True)
        cols = ['pair_id','kind','seq1','seq2','seq1_id','seq2_id','contacts','valid1','valid2']
        schema = pa.schema([('sample_id',pa.string()),('supervision_hash',pa.binary(32)),
                            ('ordered_hash',pa.binary(32)),('valid_contacts',pa.int32())])
        self.legacy_payloads(schema, cols)
        shard_ids = [row[0] for row in self.db.execute('SELECT DISTINCT shard_id FROM candidates ORDER BY shard_id').fetchall()]
        for number, shard_id in enumerate(shard_ids, 1):
            output = content_dir / f'{shard_id:04d}.parquet'
            if output.exists():
                continue
            flags = self.db.execute(f'SELECT row_index,sample_id,legacy_train,needs_hash FROM candidates WHERE shard_id={shard_id} ORDER BY row_index').fetch_arrow_table()
            source = self.registry['shards'][str(shard_id)]
            n = source['rows']
            selected, legacy_flags, hash_flags = np.zeros(n,bool), np.zeros(n,bool), np.zeros(n,bool)
            expected_ids = np.empty(n,object)
            positions = flags['row_index'].to_numpy()
            selected[positions] = True
            legacy_flags[positions] = flags['legacy_train'].to_numpy()
            hash_flags[positions] = flags['needs_hash'].to_numpy()
            expected_ids[positions] = flags['sample_id'].to_numpy()
            path = Path(self.registry['roots'][source['root']]) / source['path']
            pending = output.with_suffix('.pending')
            with pq.ParquetWriter(pending,schema,compression='zstd') as writer:
                start = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=cols,use_threads=False):
                    chosen = np.flatnonzero(selected[start:start+batch.num_rows])
                    if len(chosen):
                        buffers = payload_buffers(batch)
                        rows = []
                        for i in chosen:
                            row = start + i
                            sample_id = batch.column(0)[int(i)].as_py()
                            if sample_id != expected_ids[row]:
                                raise ValueError('Catalog/source row identity mismatch')
                            rows.append(dict(sample_id=sample_id,
                                supervision_hash=hash_row(batch,int(i),buffers) if hash_flags[row] else None,
                                ordered_hash=hash_row(batch,int(i),buffers,False) if legacy_flags[row] else None,
                                valid_contacts=int(buffers[-1][i])))
                        writer.write_table(pa.Table.from_pylist(rows,schema=schema))
                    start += batch.num_rows
                if start != n:
                    raise ValueError('Source row count changed during reading')
            pending.replace(output)
            self.status('scan_contact_supervision', shards_done=number, shards_total=len(shard_ids), shard_id=shard_id)
        self.db.execute('CREATE OR REPLACE TABLE payloads AS SELECT * FROM read_parquet(' + quoted(content_dir / '*.parquet') + ')')
        missing = self.scalar('SELECT count(*) FROM candidates ANTI JOIN payloads USING(sample_id)')
        extra = self.scalar('SELECT count(*) FROM payloads ANTI JOIN candidates USING(sample_id)')
        mismatch = self.scalar('SELECT count(*) FROM legacy_payloads l JOIN payloads p USING(sample_id) WHERE l.ordered_hash IS DISTINCT FROM p.ordered_hash OR l.supervision_hash IS DISTINCT FROM p.supervision_hash OR l.valid_contacts!=p.valid_contacts')
        duplicate_ids = self.scalar('SELECT count(*)-count(DISTINCT sample_id) FROM payloads')
        if missing or extra or mismatch or duplicate_ids:
            raise ValueError(f'Payload audit failed: missing={missing},extra={extra},mismatch={mismatch},duplicate_ids={duplicate_ids}')
        self.export('SELECT sample_id,ordered_hash,supervision_hash FROM legacy_payloads ORDER BY sample_id',
                    self.output / 'legacy_payload_verification.parquet')
        self.mark('contents', legacy_payload_mismatches=mismatch, missing=missing, extra=extra,
                  rows=self.scalar('SELECT count(*) FROM payloads'),
                  empty_contact_rows=self.scalar('SELECT count(*) FROM payloads WHERE valid_contacts=0'))

    def finalize(self):
        from flashppi.indexed_data import IndexedSplit
        self.status('deduplicate_and_write_split_indexes')
        self.db.execute("""CREATE OR REPLACE TABLE eligible AS SELECT c.*,p.supervision_hash,p.valid_contacts,
            coalesce(p.supervision_hash,from_hex(sha256('singleton:'||c.sample_id))) AS supervision_group_id
            FROM candidates c JOIN payloads p USING(sample_id) WHERE p.valid_contacts>0""")
        self.db.execute("""CREATE OR REPLACE TABLE supervision_groups AS
            SELECT supervision_group_id, coalesce(min(sample_id) FILTER(WHERE legacy_train),min(sample_id)) AS canonical_id,
                   count(*) AS original_members FROM eligible GROUP BY supervision_group_id""")
        self.db.execute("""CREATE OR REPLACE TABLE retained AS SELECT e.*,g.canonical_id FROM eligible e
            JOIN supervision_groups g USING(supervision_group_id)
            WHERE e.legacy_train OR e.sample_id=g.canonical_id""")
        self.export("""SELECT e.sample_id AS duplicate_id,e.canonical_id,e.kind,e.structure_source,
            c.structure_source AS canonical_source,e.legacy_train,e.sequence_pair_id,e.supervision_group_id,
            CASE WHEN e.legacy_train THEN 'retained_legacy_grouped' ELSE 'excluded_exact_supervision' END AS action
            FROM (SELECT e.*,g.canonical_id FROM eligible e JOIN supervision_groups g USING(supervision_group_id)) e
            JOIN eligible c ON e.canonical_id=c.sample_id WHERE e.sample_id!=e.canonical_id ORDER BY e.sample_id""",
            self.output / 'duplicate_map.parquet')
        self.export("SELECT c.sample_id,c.kind,c.structure_source,'no_valid_contacts' AS reason FROM candidates c JOIN payloads p USING(sample_id) WHERE p.valid_contacts=0",
                    self.output / 'excluded_empty_contacts.index.parquet')
        self.db.execute("""CREATE OR REPLACE TABLE pair_sizes AS SELECT sequence_pair_id,count(*) AS pair_members,
            count(DISTINCT supervision_group_id) AS pair_supervisions FROM retained GROUP BY sequence_pair_id""")
        self.db.execute('CREATE OR REPLACE TABLE supervision_sizes AS SELECT supervision_group_id,count(*) AS supervision_members FROM retained GROUP BY supervision_group_id')
        self.export("""SELECT r.sample_id,r.shard_id,r.row_index,r.kind,r.structure_source,r.source,r.legacy_train,
            r.old_row_index,r.sequence_pair_id,r.supervision_group_id,r.canonical_id,
            r.joint_cluster1 AS seq1_cluster,r.joint_cluster2 AS seq2_cluster,
            'expanded_joint_70_70_v1' AS sequence_cluster_scope,r.structure_cluster_id,r.structure_cluster_scope,
            p.pair_members,p.pair_supervisions,s.supervision_members,
            1.0/(p.pair_supervisions*s.supervision_members) AS within_pair_weight,
            r.len1,r.len2,r.valid_contacts
            FROM retained r JOIN pair_sizes p USING(sequence_pair_id) JOIN supervision_sizes s USING(supervision_group_id)
            ORDER BY r.legacy_train DESC,r.old_row_index NULLS LAST,r.shard_id,r.row_index""", self.output / 'train.index.parquet')
        self.export("SELECT DISTINCT least(joint_cluster1,joint_cluster2) AS cluster1,greatest(joint_cluster1,joint_cluster2) AS cluster2 FROM retained ORDER BY cluster1,cluster2",
                    self.output / 'known_positive_cluster_pairs.parquet')
        splits = {}
        for name, shard_id in self.registry['frozen_splits'].items():
            source = self.registry['shards'][str(shard_id)]
            path = Path(self.registry['roots'][source['root']]) / source['path']
            pf = pq.ParquetFile(path)
            if name == 'validation_contact':
                ids = pq.read_table(path,columns=['pair_id'])['pair_id']
                kinds = pa.array(['ppi']*len(ids))
                labels = pa.nulls(len(ids),pa.int8())
            else:
                ids = pa.array([f"{name}:{source['sha256'][:16]}:{i}" for i in range(pf.metadata.num_rows)])
                kinds = pa.array(['interaction']*len(ids))
                labels = pq.read_table(path,columns=['label'])['label'].cast(pa.int8())
            t = pa.table({'sample_id':ids,'shard_id':pa.array([shard_id]*len(ids),pa.int16()),
                          'row_index':pa.array(np.arange(len(ids)),pa.int32()),'kind':kinds,'label':labels})
            pq.write_table(t,self.output/(name+'.index.parquet'),compression='zstd')
        for name in ['train',*self.registry['frozen_splits']]:
            path = self.output / (name+'.index.parquet')
            splits[name] = dict(file=path.name,rows=pq.ParquetFile(path).metadata.num_rows,sha256=file_sha256(path))
        self.status('verify_membership_weights_and_frozen_evaluation')
        counts = dict(self.db.execute('SELECT kind,count(*) FROM retained WHERE legacy_train GROUP BY kind').fetchall())
        missing = self.scalar('SELECT count(*) FROM legacy ANTI JOIN retained USING(sample_id)')
        duplicates = self.scalar('SELECT count(*)-count(DISTINCT sample_id) FROM retained')
        new_leaks = self.scalar('SELECT count(*) FROM retained WHERE NOT legacy_train AND ((val_hit1 OR val_hit2) OR (test_hit1 AND test_hit2))')
        old_counts = self.config['expected_legacy']
        if missing or duplicates or new_leaks or counts != old_counts:
            raise ValueError(f'Final membership failed: {missing=},{duplicates=},{new_leaks=},{counts=}')
        alias_path = self.output/'duplicate_map.parquet'
        alias_bad = self.scalar(f'SELECT count(*) FROM read_parquet({quoted(alias_path)}) d ANTI JOIN retained r ON d.canonical_id=r.sample_id')
        weight_bad = self.scalar(f"SELECT count(*) FROM (SELECT sequence_pair_id,sum(within_pair_weight) AS w FROM read_parquet({quoted(self.output/'train.index.parquet')}) GROUP BY sequence_pair_id HAVING abs(w-1)>1e-10)")
        if alias_bad or weight_bad:
            raise ValueError('Duplicate references or within-pair sampling weights are invalid')
        old_contact_ids = set(pq.read_table(self.legacy/'validation.parquet',columns=['pair_id'])['pair_id'].to_pylist())
        self.db.register('_val_ids',pa.table({'sample_id':sorted(old_contact_ids)}))
        direct_val_overlap = self.scalar('SELECT count(*) FROM retained JOIN _val_ids USING(sample_id)')
        if direct_val_overlap:
            raise ValueError('Structural validation sample found in train')
        by_source = self.db.execute('SELECT kind,structure_source,legacy_train,count(*),count(DISTINCT sequence_pair_id),count(DISTINCT supervision_group_id) FROM retained GROUP BY ALL ORDER BY ALL').fetchall()
        duplicate_counts = self.db.execute(f'SELECT action,kind,structure_source,count(*) FROM read_parquet({quoted(alias_path)}) GROUP BY ALL ORDER BY ALL').fetchall()
        overlap_counts = self.db.execute("""SELECT kind,count(*) FROM (
            SELECT kind,sequence_pair_id FROM candidates GROUP BY kind,sequence_pair_id
            HAVING count(DISTINCT structure_source)>1) GROUP BY kind""").fetchall()
        manifest = dict(state='verifying',version=self.config['version'],config=self.config,splits=splits,
            filter_protocol=self.protocol,
            source_registry='sources.json',source_registry_sha256=file_sha256(self.output/'sources.json'),
            release_manifest_sha256=file_sha256(self.release/'manifest.json'),
            legacy_train_sha256=self.registry['legacy_train']['sha256'],
            legacy_counts=counts,old_train_missing=missing,legacy_payload_mismatches=0,
            source_counts=[dict(kind=k,source=s,legacy=l,rows=n,sequence_pairs=p,supervision_groups=g)
                           for k,s,l,n,p,g in by_source],
            duplicate_counts=[dict(action=a,kind=k,source=s,rows=n) for a,k,s,n in duplicate_counts],
            candidate_cross_source_sequence_pair_groups=dict(overlap_counts),
            cluster_multiplier=json.loads((self.output/'candidates.done.json').read_text())['cluster_multiplier'],
            sampling=dict(kind_ratio={'ppi':0.5,'ddi':0.5},
                within_pair='choose sequence-pair group, then uniform distinct supervision, then uniform retained member',
                source_ratio='not activated; choose after inspecting overlap and source statistics',
                training_automatically_started=False),
            deduplication=dict(chain_swap_invariant=True,includes_unknown_masks=True,
                different_contact_maps='retained separately; never unioned',
                identical_legacy_members='all retained; weights shared within supervision group',
                cross_kind='PPI and DDI supervision groups remain separate; exact sequences share loss-mask cluster namespace'),
            quality_policy=self.config['quality_policy'],
            limitations=['This is an expansion of the existing E.coli protocol, not a new official PINDER split.',
                'Both-chain test filtering permits one homologous training endpoint; validation excludes either endpoint.',
                'Original known-sequence decisions are frozen; k=7/direct-alignment diagnostic hits do not change membership.',
                'MMseqs2 searches are heuristic and use the observed source sequences, not reconstructed unobserved full chains.',
                'No RF2 isolation or absence of pretrained-model exposure is claimed.',
                'No new pLDDT/PAE quality filter has been applied. Source preparation is not a confidence-quality certificate.',
                'The data contain structural positive examples. Real interaction negatives are not supplied by this expansion.',
                'PPI structure clusters are unavailable; sequence-pair/sequence clustering is not an interface-structure leakage proof.'],
            code={'builder':str(Path(__file__).resolve()),'builder_sha256':file_sha256(__file__),
                  'reader':str(ROOT/'flashppi/indexed_data.py'),'reader_sha256':file_sha256(ROOT/'flashppi/indexed_data.py')},
            versions={'python':sys.version,'numpy':np.__version__,'pyarrow':pa.__version__,'duckdb':duckdb.__version__})
        write_json(self.output/'manifest.json',manifest)
        # Exercise the public reader with shuffled accesses and bounded caches.
        rng = np.random.default_rng(self.config['seed'])
        read_checks = {}
        for name, entry in splits.items():
            reader = IndexedSplit(self.output,name,cache_row_groups=2,allow_incomplete=True)
            if name == 'train':
                indices = [0,counts['ppi']-1,counts['ppi'],sum(counts.values())-1,len(reader)-1]
                indices += rng.integers(0,len(reader),size=96).tolist()
            else:
                indices = list(range(len(reader))) if len(reader)<1000 else [0,len(reader)-1]+rng.integers(0,len(reader),size=32).tolist()
            for index in indices:
                row = reader[index]
                a = row['split_annotation']
                s = self.registry['shards'][str(a['shard_id'])]
                starts = s['row_group_starts']
                group = int(np.searchsorted(starts,a['row_index'],side='right')-1)
                direct = pq.ParquetFile(Path(self.registry['roots'][s['root']])/s['path']).read_row_group(group).slice(a['row_index']-starts[group],1).to_pylist()[0]
                for field in ('seq1','seq2','contacts','contacts_8A','valid1','valid2','label','pair_id'):
                    if field in direct and row[field] != direct[field]:
                        raise ValueError('Reader changed source payload: '+field)
            read_checks[name] = len(indices)
        source_checks = 0
        for source in self.registry['shards'].values():
            path = Path(self.registry['roots'][source['root']])/source['path']
            if file_sha256(path) != source['sha256']:
                raise ValueError('Immutable source changed during build')
            source_checks += 1
        if file_sha256(self.legacy/'train.parquet') != self.registry['legacy_train']['sha256']:
            raise ValueError('Legacy training file changed during build')
        audits = dict(state='passed',legacy_counts=counts,legacy_retention_rate=1.0,old_train_missing=missing,
            legacy_payload_mismatches=0,duplicate_train_ids=duplicates,new_homology_violations=new_leaks,
            structural_validation_id_overlap=direct_val_overlap,invalid_duplicate_targets=alias_bad,
            invalid_pair_weights=weight_bad,reader_checks=read_checks,source_hashes_verified_after_build=source_checks,
            frozen_split_rows={k:v['rows'] for k,v in splits.items() if k!='train'},
            source_counts=manifest['source_counts'],duplicate_counts=manifest['duplicate_counts'],
            candidate_cross_source_sequence_pair_groups=dict(overlap_counts),
            legacy_homology_recheck=json.loads((self.output/'candidates.done.json').read_text())['legacy_recheck'])
        write_json(self.output/'verification.json',audits)
        self.write_readme(manifest,audits)
        shutil.copyfile(LOCK_PATH, self.output / 'filter_protocol.lock.json')
        tracked = ['duplicate_map.parquet','known_positive_cluster_pairs.parquet','cluster_bridges.parquet',
                   'excluded_homology.index.parquet','excluded_empty_contacts.index.parquet',
                   'legacy_payload_verification.parquet','ppi_homology_hits.parquet','ddi_homology_hits.parquet',
                   'verification.json','README.md','filter_protocol.lock.json']
        manifest['artifacts'] = {f:dict(sha256=file_sha256(self.output/f),bytes=(self.output/f).stat().st_size) for f in tracked}
        manifest['state'] = 'complete'
        manifest['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_json(self.output/'manifest.json',manifest)
        self.mark('complete',rows=splits['train']['rows'],verification_sha256=file_sha256(self.output/'verification.json'))
        write_json(self.output/'status.json',dict(state='complete',train_rows=splits['train']['rows'],legacy_counts=counts,
                                                 verification='verification.json',training_started=False))
        print(json.dumps(audits,ensure_ascii=False),flush=True)

    def write_readme(self, manifest, audit):
        lines = ['# CongLab + PINDER + Teddymer 索引划分（固定旧过滤协议）','',
            '本目录仅保存索引、分组、过滤证据和说明。序列、接触标签及 mask 从固定版本的原始 Parquet 按需读取。',
            '原始数据没有复制、改写或删除；本次构建没有启动训练。','',
            '## 划分与保留约束','',
            f"训练索引共 {manifest['splits']['train']['rows']:,} 条。旧 CongLab PPI {manifest['legacy_counts']['ppi']:,} 条、DDI {manifest['legacy_counts']['ddi']:,} 条全部保留，保留率 100%。",
            '每条旧样本的序列轴、接触标签和 mask 已与原训练文件逐条计算指纹核对。',
            f"验证接触集 {manifest['splits']['validation_contact']['rows']:,} 条、验证互作集 {manifest['splits']['validation_interaction']['rows']:,} 条、E.coli test {manifest['splits']['test']['rows']:,} 条保持原文件、原标签、原顺序。",
            'val/test 互作索引指向原二分类文件，保留原负例；它们不需要存在于正结构数据包中。','',
            '| 类型 | 实际结构来源 | 旧训练成员 | 保留结构数 | 序列对数 | 不同监督数 |',
            '|---|---|---:|---:|---:|---:|']
        for r in manifest['source_counts']:
            lines.append(f"| {r['kind']} | {r['source']} | {r['legacy']} | {r['rows']:,} | {r['sequence_pairs']:,} | {r['supervision_groups']:,} |")
        lines += ['', '各来源的序列对数可能重叠，不能直接相加当作全局独立样本数。','',
            '## 文件','',
            '- train.index.parquet：样本 ID、源分片/行位置、旧训练标记、来源及采样分组。',
            '- validation_contact.index.parquet、validation_interaction.index.parquet、test.index.parquet：固定评估索引。',
            '- duplicate_map.parquet：重复 ID、保留代表 ID、原因和处理动作。excluded_exact_supervision 不进入训练；retained_legacy_grouped 仍在训练中。',
            '- sources.json：数据根目录、相对分片路径、行组位置及源文件 SHA256。迁移机器时可覆盖根目录。',
            '- manifest.json：机器可读的版本、参数、数量、校验值和限制。',
            '- verification.json：旧训练完整性、标签一致性、过滤、去重引用、权重和索引读取验收。',
            '- known_positive_cluster_pairs.parquet：只由最终训练正例建立的聚类对屏蔽表，不使用验证/test 标签。',
            '- excluded_*.index.parquet：被排除的新数据和未选用的其他 CongLab 样本的索引与原因。',
            '- *_homology_hits.parquet、*_homology.json、*_mmseqs.log：同源命中、实际命令及检索记录。',
            '- legacy_payload_verification.parquet：逐条旧样本指纹；不含序列或接触矩阵。','',
            '## 过滤口径','',
            '唯一允许的协议为 conglab_ecoli_v1_locked；filter_protocol.lock.json 固定旧源码、参数、评估文件和命中记录。未经用户明确同意不得改变。',
            '旧序列复用原来的命中与未命中判定。新增序列使用原命令及明确固定的 k=6、k-mer score=96；test/val 分别检索并校验实际日志。',
            'k=7 或直接比对只属于独立诊断，不自动加入排除集合；这不代表旧启发式过滤不存在漏检。',
            '完全沿用本轮 E.coli 实验的边界：identity 严格 >30%；PPI 双方覆盖度严格 >50%；DDI 训练域覆盖度 >=80%。',
            '测试集：两条训练链都命中任意 test 序列才排除；验证集：任意一条训练链命中即排除。',
            '正负评估样本两侧序列都进入参考集合。完整来源序列先过滤，之后才允许训练裁剪。',
            '新增过滤只决定新增候选；旧训练全部保留。旧样本复查命中数量见 verification.json 的 legacy_homology_recheck，',
            '每行依次为 kind、val 任一端命中数、test 双端命中数、test 任一端命中数。',
            '这不是任一 test 同源端点都排除的严格协议，也不保证 RF2 或 PINDER 官方 holdout 隔离。',
            '来源仅有观察到的坐标残基序列，不补造完整链；MMseqs2 为启发式搜索。','',
            '## 去重、标签与采样','',
            '样本 ID 与序列对 ID 保持来源定义。链交换时同时转置接触图和交换 mask，再对序列、二值标签、mask 做 SHA256 指纹。',
            '相同序列对但接触图或 mask 不同，仍是不同监督，分别保留。接触图绝不取并集。',
            '完全相同监督优先选旧 CongLab 作为代表；新副本只保留映射。旧集合内部监督相同但 ID 不同的成员全部保留，均具有正采样权重。',
            'sequence_pair_id 跨来源统一分组；supervision_group_id 区分同序列对的不同监督。',
            'within_pair_weight = 1 / (该序列对的不同监督数 × 当前监督组的保留成员数)，每个序列对组内总权重为 1。',
            '这些是组内权重，不能直接宣称来源比例已确定。新增训练配置需要按组采样、再应用来源/类型配额。',
            'PPI 与 DDI 的监督任务保持分组；同一蛋白的局部结构域不能仅因包含关系就当作重复。',
            '聚类数字按来源 scope 隔离，再通过完全相同序列合并命名空间。旧样本的聚类编号可能变动，输入和监督不变。',
            '训练接触仍为 Cβ 严格 <12 Å，Gly 用 Cα；未知位置和 padding 不能当作负标签。',
            '未追加未经验证的 pLDDT/PAE 筛选；结构正例扩充不等于补充实验确认的互作负例。','',
            '## 读取与复现','',
            '在 FlashPPI 项目根目录中：','',
            '    from flashppi.indexed_data import IndexedSplit',
            f"    train = IndexedSplit({str(self.output)!r}, 'train')",
            '    sample = train[0]',
            "    print(sample['seq1'], sample['split_annotation']['legacy_train'])",'',
            '读取器仅依赖 NumPy/PyArrow，按行组缓存；默认验证索引和实际访问源分片的 SHA256。',
            'root_overrides 可传 release、legacy、test 的新根目录；文件相对路径、版本和哈希仍须一致。',
            '构建工具另需 requirements-data-index.txt 中的 DuckDB；当前构建工具依赖安装在 artifacts/environments/data_index_deps。','',
            f'    PYTHONPATH=artifacts/environments/data_index_deps artifacts/environments/esmc_prott5_gated_py312/bin/python tools/train_test.py --config {self.config_file}','',
            '修改配置应选择新的 output 目录。build_scratch 可在完整验收后删除；全部正式索引和 reader 不依赖 scratch。',
            '生成索引不会自动修改研究训练入口或现有 checkpoint；正式续训前需接入索引读取与分组采样。','']
        (self.output/'README.md').write_text('\n'.join(lines))


class StrictBuild(Build):
    """Fresh CongLab/distilled split in the existing maintained builder.

    Source FASTA entries, source cluster mappings and test rows are immutable.
    Only indexes and evidence are written. This class never starts training.
    """
    def __init__(self, config_file):
        self.config_file = Path(config_file).resolve()
        self.config = json.loads(self.config_file.read_text())
        self.protocol = verify_strict_inputs(self.config)
        self.lock = read_strict_lock(self.protocol['id'])
        self.rf2 = self.protocol['id'] == RF2_PROTOCOL_ID
        self.release = Path(self.config['release'])
        self.legacy = Path(self.config['legacy_directory'])
        self.meta = json.loads((self.release / 'DATASET.json').read_text())
        if self.meta['state'] != 'complete':
            raise ValueError('Incomplete unified source release')
        self.output = Path(self.config['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.scratch = self.output / 'build_scratch'
        self.scratch.mkdir(exist_ok=True)
        self.started = time.monotonic()
        identity = dict(config=self.config, filter_protocol=self.protocol,
            code_sha256={name: file_sha256(ROOT / name) for name in (
                'tools/train_test.py', 'tools/audit_expanded_index_split.py',
                'flashppi/data_protocol.py', 'flashppi/indexed_data.py')},
            versions=dict(python=sys.version, numpy=np.__version__, pyarrow=pa.__version__, duckdb=duckdb.__version__))
        path = self.output / 'build_identity.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Build inputs/code/runtime changed; refusing reuse of partial results')
        write_json(path, identity)
        shutil.copyfile(RF2_LOCK_PATH if self.rf2 else STRICT_LOCK_PATH, self.output / 'filter_protocol.lock.json')
        self.db = duckdb.connect(str(self.scratch / 'catalog.duckdb'))
        self.db.execute('SET threads = ' + str(self.config['threads']))
        self.db.execute('SET memory_limit = ' + quoted(self.config['duckdb_memory_limit']))
        self.db.execute('SET temp_directory = ' + quoted(self.scratch / 'duckdb_temp'))
        self.db.execute('SET preserve_insertion_order = false')

    def catalog(self):
        if self.done('catalog'):
            self.registry = json.loads((self.output / 'sources.json').read_text())
            return
        self.status('verify_source_hashes_and_catalog')
        registry = dict(roots={'release': str(self.release), 'test': str(Path(self.config['test']).parent)},
                        shards={}, frozen_splits={})
        self.db.execute('DROP TABLE IF EXISTS catalog')
        first = True
        entries = self.meta['parquet_shards']
        for sid, entry in enumerate(entries):
            allowed = self.config['allowed_sources'][entry['kind']]
            if not set(allowed).intersection(entry['sources']):
                continue
            path = self.release / entry['file']
            if file_sha256(path) != entry['sha256']:
                raise ValueError('Source shard hash changed: ' + str(path))
            pf = pq.ParquetFile(path)
            starts = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])]
            if int(starts[-1]) != entry['records']:
                raise ValueError('Source shard row count changed')
            registry['shards'][str(sid)] = dict(root='release', path=entry['file'], sha256=entry['sha256'],
                rows=entry['records'], row_group_starts=starts[:-1].tolist())
            query = f"""SELECT sample_id,kind,seq1_id,seq2_id,sequence_pair_id,structure_source,source,
                seq1_cluster,seq2_cluster,sequence_cluster_scope,structure_cluster_id,structure_cluster_scope,
                len1,len2,seq1,seq2,{sid}::SMALLINT AS shard_id,file_row_number::INTEGER AS row_index
                FROM read_parquet({quoted(path)},file_row_number=true)
                WHERE kind={quoted(entry['kind'])} AND structure_source IN ({','.join(map(quoted, allowed))})"""
            self.db.execute(('CREATE TABLE catalog AS ' if first else 'INSERT INTO catalog ') + query)
            first = False
            if sid % 20 == 0:
                self.status('verify_source_hashes_and_catalog', source_shard=sid, total_shards=len(entries))
        if first:
            raise ValueError('No eligible source shards')
        wrong = self.scalar("""SELECT count(*) FROM catalog WHERE
            seq1_id != 'seq_'||sha256(seq1) OR seq2_id != 'seq_'||sha256(seq2)
            OR len1!=length(seq1) OR len2!=length(seq2) OR len1<1 OR len2<1
            OR len1>65535 OR len2>65535 OR seq1_cluster IS NULL OR seq2_cluster IS NULL""")
        if wrong or self.scalar('SELECT count(*)-count(DISTINCT sample_id) FROM catalog'):
            raise ValueError('Invalid source sequence IDs/lengths/clusters or duplicate sample IDs')
        if self.scalar('''SELECT count(*) FROM (SELECT kind,least(seq1_id,seq2_id),greatest(seq1_id,seq2_id)
                FROM catalog GROUP BY ALL HAVING count(DISTINCT sequence_pair_id)!=1)''') or self.scalar('''
                SELECT count(*) FROM (SELECT sequence_pair_id FROM catalog GROUP BY sequence_pair_id HAVING
                count(DISTINCT (kind,least(seq1_id,seq2_id),greatest(seq1_id,seq2_id)))!=1)'''):
            raise ValueError('Sequence-pair IDs are not consistent across sources/chain order')
        expected = Counter()
        for entry in entries:
            for source, count in entry['source_counts'].items():
                if source in self.config['allowed_sources'][entry['kind']]:
                    expected[entry['kind'], source] += count
        actual = {(k, s): n for k, s, n in self.db.execute(
            'SELECT kind,structure_source,count(*) FROM catalog GROUP BY ALL').fetchall()}
        if actual != expected:
            raise ValueError(f'Selected source counts mismatch: {actual} != {expected}')
        test = Path(self.config['test'])
        pf = pq.ParquetFile(test)
        starts = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])]
        labels = dict(self.db.execute(f'SELECT label,count(*) FROM read_parquet({quoted(test)}) GROUP BY label').fetchall())
        if labels != {0: self.lock['test']['negatives'], 1: self.lock['test']['positives']}:
            raise ValueError('Frozen test label counts changed')
        sid = len(entries)
        registry['shards'][str(sid)] = dict(root='test', path=test.name, sha256=self.config['test_sha256'],
            rows=pf.metadata.num_rows, row_group_starts=starts[:-1].tolist())
        registry['frozen_splits']['test'] = sid
        self.registry = registry
        write_json(self.output / 'sources.json', registry)
        self.export('SELECT * EXCLUDE(seq1,seq2) FROM catalog ORDER BY sample_id', self.output / 'selected_sources.index.parquet')
        self.mark('catalog', rows=self.scalar('SELECT count(*) FROM catalog'), source_counts=[
            dict(kind=k, source=s, rows=n) for (k, s), n in sorted(actual.items())])

    def sequence_inputs(self):
        """Check CSV links, FASTA content and TSV clusters, without copying payloads."""
        if self.done('sequence_inputs'):
            return
        checks = {}
        for kind in ('ppi', 'ddi'):
            self.status('verify_pair_fasta_and_cluster_mapping', kind=kind)
            self.db.execute(f"""CREATE OR REPLACE TABLE seqs_{kind} AS
                SELECT seq1_id AS seq_id,seq1 AS seq,seq1_cluster AS cluster FROM catalog WHERE kind={quoted(kind)}
                UNION SELECT seq2_id,seq2,seq2_cluster FROM catalog WHERE kind={quoted(kind)}""")
            if self.scalar(f'SELECT count(*)-count(DISTINCT seq_id) FROM seqs_{kind}'):
                raise ValueError('A source sequence has inconsistent content or cluster assignment')
            seqs = dict(self.db.execute(f'SELECT seq_id,seq FROM seqs_{kind}').fetchall())
            found = set()
            current, parts = None, []
            def consume():
                if current in seqs:
                    if current in found or ''.join(parts) != seqs[current]:
                        raise ValueError('Source FASTA/Parquet mismatch: ' + current)
                    found.add(current)
            with (self.release / (kind + '_seqs.fasta')).open() as handle:
                for line in handle:
                    if line.startswith('>'):
                        consume()
                        current, parts = line[1:].strip().split()[0], []
                    elif current in seqs:
                        parts.append(line.strip())
                consume()
            if found != seqs.keys():
                raise ValueError('Selected sequences are missing from source FASTA')
            path = self.output / (kind + '_candidate.fasta')
            with path.open('w') as handle:
                for sid, seq in sorted(seqs.items()):
                    handle.write(f'>{sid}\n{seq}\n')
            cluster_path = self.release / (kind + '_sequence_clusters.tsv')
            self.db.execute(f"""CREATE OR REPLACE TABLE verified_clusters_{kind} AS
                SELECT sequence_id AS seq_id,CAST(cluster_id AS BIGINT) AS cluster
                FROM read_csv({quoted(cluster_path)},delim='\t',header=true,all_varchar=true)
                SEMI JOIN seqs_{kind} ON sequence_id=seq_id""")
            bad = self.scalar(f"""SELECT count(*) FROM seqs_{kind} s FULL JOIN verified_clusters_{kind} t USING(seq_id)
                WHERE s.cluster IS DISTINCT FROM t.cluster""")
            if bad or self.scalar(f'SELECT count(*) FROM verified_clusters_{kind}') != len(seqs):
                raise ValueError('Parquet sequence clusters do not match the provided TSV')
            pair_path = self.release / (kind + '_pairs.csv.gz')
            self.db.execute(f"""CREATE OR REPLACE TABLE checked_pairs AS
                SELECT p.pair_id,p.seq1,p.seq2 FROM read_csv({quoted(pair_path)},header=true,all_varchar=true) p
                SEMI JOIN (SELECT sample_id FROM catalog WHERE kind={quoted(kind)}) c
                ON p.pair_id=c.sample_id""")
            bad = self.scalar(f"""SELECT count(*) FROM catalog c LEFT JOIN checked_pairs p ON c.sample_id=p.pair_id
                WHERE c.kind={quoted(kind)} AND (c.seq1_id IS DISTINCT FROM p.seq1 OR c.seq2_id IS DISTINCT FROM p.seq2)""")
            count = self.scalar(f'SELECT count(*) FROM catalog WHERE kind={quoted(kind)}')
            if bad or self.scalar('SELECT count(*) FROM checked_pairs') != count:
                raise ValueError('Pair CSV/Parquet sequence links do not match')
            checks[kind] = dict(rows=count, sequences=len(seqs), fasta_matches=len(found),
                cluster_tsv_matches=len(seqs), pair_csv_matches=count, candidate_sha256=file_sha256(path))
            self.export(f'SELECT seq_id,cluster,length(seq) AS length FROM seqs_{kind} ORDER BY seq_id',
                        self.output / (kind + '_sequence_catalog.parquet'))
        table = pq.read_table(self.config['test'], columns=['seq1', 'seq2'])
        references = set(table['seq1'].to_pylist() + table['seq2'].to_pylist())
        (self.output / 'test_reference.fasta').write_text(reference_fasta(references))
        checks['test'] = dict(sequences=len(references),
            fasta_sha256=file_sha256(self.output / 'test_reference.fasta'), all_positive_and_negative_rows=True)
        self.mark('sequence_inputs', checks=checks)

    def homology(self):
        self.sequence_inputs()
        if self.done('homology'):
            for kind in ('ppi', 'ddi'):
                record = json.loads((self.output / (kind + '_homology.json')).read_text())
                if record['hits_sha256'] != file_sha256(self.output / (kind + '_homology_hits.parquet')):
                    raise ValueError('Cached homology hits changed')
            return
        refs_table = pq.read_table(self.config['test'], columns=['seq1', 'seq2'])
        refs = sorted(set(refs_table['seq1'].to_pylist() + refs_table['seq2'].to_pylist()))
        ref_lengths = {f'r{i}': len(s) for i, s in enumerate(refs)}
        ref_ids = {sequence_id(s): f'r{i}' for i, s in enumerate(refs)}
        audit_schema = pa.schema([('seq_id', pa.string()), ('reference_id', pa.string()),
            ('direction', pa.string()), ('nident', pa.int32()), ('alnlen', pa.int32()),
            ('train_start', pa.int32()), ('train_end', pa.int32()), ('train_length', pa.int32()),
            ('test_start', pa.int32()), ('test_end', pa.int32()), ('test_length', pa.int32())])
        if self.rf2:
            audit_schema = audit_schema.append(pa.field('evalue', pa.float64())).append(pa.field('bits', pa.float64()))
        for kind in ('ppi', 'ddi'):
            lengths = dict(self.db.execute(f'SELECT seq_id,length(seq) FROM seqs_{kind}').fetchall())
            hits = set() if self.rf2 else {s for s in lengths if s in ref_ids}
            evidence_path = self.output / (kind + '_homology_evidence.parquet')
            exact_count = len(hits)
            records = []
            with pq.ParquetWriter(evidence_path, audit_schema, compression='zstd') as writer:
                buffer = [dict(seq_id=s, reference_id=ref_ids[s], direction='exact', nident=lengths[s],
                    alnlen=lengths[s], train_start=1, train_end=lengths[s], train_length=lengths[s],
                    test_start=1, test_end=lengths[s], test_length=lengths[s]) for s in sorted(hits)]
                if buffer:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=audit_schema))
                for direction in self.lock['search']['directions']:
                    candidate = self.output / (kind + '_candidate.fasta')
                    reference = self.output / 'test_reference.fasta'
                    query, target = (reference, candidate) if direction.startswith('test_to_train') else (candidate, reference)
                    target_count = len(lengths) if direction.startswith('test_to_train') else len(refs)
                    name = f'{kind}_{direction}'
                    raw, log = self.scratch / (name + '.tsv'), self.output / (name + '_mmseqs.log')
                    archive, receipt = self.output / (name + '.tsv.gz'), self.output / (name + '.search.json')
                    command = strict_command(self.config['mmseqs'], query, target, raw,
                        self.scratch / (name + '_tmp'), target_count, kind, direction, self.protocol['id'])
                    identity = dict(command=command, query_sha256=file_sha256(query), target_sha256=file_sha256(target),
                                    filter_protocol=self.protocol)
                    if receipt.exists():
                        saved = json.loads(receipt.read_text())
                        if (saved['inputs'] != identity or file_sha256(archive) != saved['archive_sha256']
                                or file_sha256(log) != saved['log_sha256']):
                            raise ValueError('Search cache identity changed')
                    else:
                        self.status('mmseqs_search', kind=kind, direction=direction,
                                    candidate_sequences=len(lengths), reference_sequences=len(refs))
                        with log.open('w') as handle:
                            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                                           env={**os.environ, 'LC_ALL': 'C', 'OMP_NUM_THREADS': '8'})
                        verify_strict_search_log(log, direction, self.protocol['id'])
                        with raw.open('rb') as source, archive.with_suffix('.pending').open('wb') as dest:
                            with gzip.GzipFile(filename='', mode='wb', fileobj=dest, mtime=0) as zipped:
                                shutil.copyfileobj(source, zipped)
                        archive.with_suffix('.pending').replace(archive)
                        saved = dict(inputs=identity, raw_sha256=file_sha256(raw), archive_sha256=file_sha256(archive),
                                     log_sha256=file_sha256(log))
                        write_json(receipt, saved)
                        raw.unlink()
                    verify_strict_search_log(log, direction, self.protocol['id'])
                    matched, total, empty_alignments, buffer = 0, 0, 0, []
                    digest = hashlib.sha256()
                    with gzip.open(archive, 'rt') as handle:
                        for line in handle:
                            digest.update(line.encode())
                            fields = line.rstrip('\n').split('\t')
                            total += 1
                            passed = strict_hit(fields, kind, direction, self.protocol['id'])
                            if self.rf2 and int(fields[3]) == 0:
                                empty_alignments += 1
                            query_id, target_id = fields[:2]
                            qs, qe, qlen, ts, te, tlen = map(int, fields[4:10])
                            sid, rid = (target_id, query_id) if direction.startswith('test_to_train') else (query_id, target_id)
                            a, b = ((ts, te, tlen), (qs, qe, qlen)) if direction.startswith('test_to_train') else ((qs, qe, qlen), (ts, te, tlen))
                            if lengths.get(sid) != a[2] or ref_lengths.get(rid) != b[2]:
                                raise ValueError('Alignment sequence identity/length does not match fixed FASTA')
                            if passed:
                                matched += 1
                                hits.add(sid)
                                evidence_row = dict(seq_id=sid, reference_id=rid, direction=direction,
                                    nident=int(fields[2]), alnlen=int(fields[3]), train_start=a[0], train_end=a[1],
                                    train_length=a[2], test_start=b[0], test_end=b[1], test_length=b[2])
                                if self.rf2:
                                    evidence_row.update(evalue=float(fields[10]), bits=float(fields[11]))
                                buffer.append(evidence_row)
                            if len(buffer) >= 65536:
                                writer.write_table(pa.Table.from_pylist(buffer, schema=audit_schema))
                                buffer = []
                        if buffer:
                            writer.write_table(pa.Table.from_pylist(buffer, schema=audit_schema))
                    if digest.hexdigest() != saved['raw_sha256']:
                        raise ValueError('Archived raw alignment content changed')
                    records.append(dict(direction=direction, alignments=total, qualified_alignments=matched,
                        empty_alignment_rows=empty_alignments,
                        archive=archive.name, receipt=receipt.name, **saved))
            hit_path = self.output / (kind + '_homology_hits.parquet')
            pq.write_table(pa.table({'seq_id': pa.array(sorted(hits), type=pa.string()),
                                    'test_hit': pa.array([True] * len(hits), type=pa.bool_())}), hit_path, compression='zstd')
            write_json(self.output / (kind + '_homology.json'), dict(filter_protocol=self.protocol,
                sequences=len(lengths), exact_sequence_hits=exact_count, hit_sequences=len(hits), searches=records,
                hits_sha256=file_sha256(hit_path), evidence_sha256=file_sha256(evidence_path)))
            self.status('homology_kind_complete', kind=kind, sequences=len(lengths), hit_sequences=len(hits))
        self.mark('homology')

    def candidates(self):
        if self.done('candidates'):
            return
        self.status('apply_both_endpoint_exclusion' if self.rf2 else 'apply_any_endpoint_exclusion')
        for kind in ('ppi', 'ddi'):
            self.db.execute(f'CREATE OR REPLACE TABLE hits_{kind} AS SELECT * FROM read_parquet(' +
                            quoted(self.output / (kind + '_homology_hits.parquet')) + ')')
        self.db.execute("CREATE OR REPLACE TABLE hits AS SELECT *, 'ppi' AS kind FROM hits_ppi UNION ALL SELECT *, 'ddi' AS kind FROM hits_ddi")
        # Old membership is an annotation for the before/after report, never an exemption.
        old = self.legacy / 'train.parquet'
        self.db.execute(f"""CREATE OR REPLACE TABLE legacy AS SELECT pair_id AS sample_id,
            file_row_number::INTEGER AS old_row_index FROM read_parquet({quoted(old)},file_row_number=true)""")
        operator = 'AND' if self.rf2 else 'OR'
        reason = 'test_both_endpoints_homology' if self.rf2 else 'test_either_endpoint_homology'
        self.db.execute(f"""CREATE OR REPLACE TABLE decisions AS SELECT c.*,
            false AS legacy_train,l.old_row_index,l.sample_id IS NOT NULL AS was_legacy_train,
            coalesce(h1.test_hit,false) AS test_hit1,coalesce(h2.test_hit,false) AS test_hit2,
            false AS val_hit1,false AS val_hit2,
            CASE WHEN coalesce(h1.test_hit,false) {operator} coalesce(h2.test_hit,false)
                THEN '{reason}' ELSE 'new_candidate' END AS decision
            FROM catalog c LEFT JOIN legacy l USING(sample_id)
            LEFT JOIN hits h1 ON c.kind=h1.kind AND c.seq1_id=h1.seq_id
            LEFT JOIN hits h2 ON c.kind=h2.kind AND c.seq2_id=h2.seq_id""")
        self.export("SELECT * EXCLUDE(seq1,seq2) FROM decisions WHERE decision!='new_candidate' ORDER BY sample_id",
                    self.output / 'excluded_homology.index.parquet')
        info = self.cluster_candidates()
        self.mark('candidates', **info)

    def legacy_payloads(self, schema, cols):
        # The shared payload scanner checks sequence axes, masks and fingerprints.
        # Fresh filtering does not inherit a requirement to retain legacy rows.
        self.table('legacy_payloads', pa.Table.from_pylist([], schema=schema))

    def finalize(self):
        from flashppi.indexed_data import IndexedSplit
        self.status('deduplicate_and_write_strict_indexes')
        self.db.execute("""CREATE OR REPLACE TABLE eligible AS SELECT c.*,p.supervision_hash,p.valid_contacts,
            coalesce(p.supervision_hash,from_hex(sha256('singleton:'||c.sample_id))) AS supervision_group_id
            FROM candidates c JOIN payloads p USING(sample_id) WHERE p.valid_contacts>0""")
        self.db.execute("""CREATE OR REPLACE TABLE supervision_groups AS SELECT supervision_group_id,
            coalesce(min(sample_id) FILTER(WHERE structure_source='conglab'),min(sample_id)) AS canonical_id
            FROM eligible GROUP BY supervision_group_id""")
        self.db.execute("""CREATE OR REPLACE TABLE retained AS SELECT e.*,g.canonical_id FROM eligible e
            JOIN supervision_groups g USING(supervision_group_id) WHERE e.sample_id=g.canonical_id""")
        self.export("""SELECT e.sample_id AS duplicate_id,g.canonical_id,e.kind,e.structure_source,
            r.structure_source AS canonical_source,e.sequence_pair_id,e.supervision_group_id,
            'excluded_exact_supervision' AS action FROM eligible e
            JOIN supervision_groups g USING(supervision_group_id) JOIN retained r ON r.sample_id=g.canonical_id
            WHERE e.sample_id!=g.canonical_id ORDER BY e.sample_id""", self.output / 'duplicate_map.parquet')
        self.export("""SELECT c.sample_id,c.kind,c.structure_source,'no_valid_contacts' AS reason
            FROM candidates c JOIN payloads p USING(sample_id) WHERE p.valid_contacts=0 ORDER BY sample_id""",
            self.output / 'excluded_empty_contacts.index.parquet')
        self.db.execute('CREATE OR REPLACE TABLE pair_sizes AS SELECT sequence_pair_id,count(*) AS pair_members FROM retained GROUP BY sequence_pair_id')
        scope = 'rf2ppi_v3_joint_sequence_70_70' if self.rf2 else 'strict_v2_joint_sequence_70_70'
        self.export(f"""SELECT r.sample_id,r.shard_id,r.row_index,r.kind,r.structure_source,r.source,
            r.was_legacy_train AS legacy_train,r.sequence_pair_id,r.supervision_group_id,r.canonical_id,
            r.seq1_id,r.seq2_id,r.seq1_cluster AS source_seq1_cluster,r.seq2_cluster AS source_seq2_cluster,
            r.sequence_cluster_scope AS source_sequence_cluster_scope,
            r.joint_cluster1 AS seq1_cluster,r.joint_cluster2 AS seq2_cluster,
            '{scope}' AS sequence_cluster_scope,r.structure_cluster_id,r.structure_cluster_scope,
            p.pair_members,p.pair_members AS pair_supervisions,1 AS supervision_members,
            1.0/p.pair_members AS within_pair_weight,r.len1,r.len2,r.valid_contacts
            FROM retained r JOIN pair_sizes p USING(sequence_pair_id) ORDER BY r.kind,r.sample_id""",
            self.output / 'train.index.parquet')
        self.export("""SELECT DISTINCT least(joint_cluster1,joint_cluster2) AS cluster1,
            greatest(joint_cluster1,joint_cluster2) AS cluster2 FROM retained ORDER BY cluster1,cluster2""",
            self.output / 'known_positive_cluster_pairs.parquet')
        sid = self.registry['frozen_splits']['test']
        src = self.registry['shards'][str(sid)]
        labels = pq.read_table(self.config['test'], columns=['label'])['label'].cast(pa.int8())
        n = len(labels)
        pq.write_table(pa.table({'sample_id': [f"test:{src['sha256'][:16]}:{i}" for i in range(n)],
            'shard_id': pa.array([sid]*n, pa.int16()), 'row_index': pa.array(np.arange(n), pa.int32()),
            'kind': ['interaction']*n, 'label': labels}), self.output / 'test.index.parquet', compression='zstd')
        splits = {}
        for name in ('train', 'test'):
            path = self.output / (name + '.index.parquet')
            splits[name] = dict(file=path.name, rows=pq.ParquetFile(path).metadata.num_rows, sha256=file_sha256(path))
        if not splits['train']['rows']:
            raise ValueError('Empty training population after filtering')
        before_after = self.db.execute("""SELECT l.sample_id,coalesce(d.kind,'missing') AS kind,
            CASE WHEN d.sample_id IS NULL THEN 'not_in_release'
                 WHEN d.decision!='new_candidate' THEN d.decision
                 WHEN p.valid_contacts=0 THEN 'no_valid_contacts'
                 WHEN r.sample_id IS NOT NULL THEN 'retained'
                 ELSE 'excluded_exact_supervision' END AS decision
            FROM legacy l LEFT JOIN decisions d USING(sample_id) LEFT JOIN payloads p USING(sample_id)
            LEFT JOIN retained r USING(sample_id) ORDER BY l.sample_id""").fetch_arrow_table()
        pq.write_table(before_after, self.output / 'legacy_before_after.index.parquet', compression='zstd')
        by_source = self.db.execute("""SELECT c.kind,c.structure_source,count(*) AS input_rows,
            count(*) FILTER(WHERE d.decision!='new_candidate') AS homology_excluded,
            count(*) FILTER(WHERE p.valid_contacts=0) AS empty_contacts,
            count(*) FILTER(WHERE p.valid_contacts>0 AND r.sample_id IS NULL) AS duplicates,
            count(r.sample_id) AS retained
            FROM catalog c JOIN decisions d USING(sample_id) LEFT JOIN payloads p USING(sample_id)
            LEFT JOIN retained r USING(sample_id) GROUP BY c.kind,c.structure_source ORDER BY c.kind,c.structure_source""").fetchall()
        source_counts = [dict(zip(('kind','source','input_rows','homology_excluded','empty_contacts','duplicates','rows'), row)) for row in by_source]
        if any(r['input_rows'] != sum(r[k] for k in ('homology_excluded','empty_contacts','duplicates','rows')) for r in source_counts):
            raise ValueError('Selected rows are not completely accounted for')
        manifest = dict(state='verifying', version=self.config['version'], config=self.config, filter_protocol=self.protocol,
            splits=splits, source_registry='sources.json', source_registry_sha256=file_sha256(self.output / 'sources.json'),
            source_counts=source_counts, cluster_multiplier=json.loads((self.output/'candidates.done.json').read_text())['cluster_multiplier'],
            sampling=dict(kind_ratio={'ppi': 0.5, 'ddi': 0.5}, ppi='balanced sequence clusters, weighted within cluster',
                ddi='balanced sequence clusters, weighted within cluster', within_pair='1 / retained distinct supervision count',
                source_quota='not activated in this data-only build', training_automatically_started=False),
            sequence_scope=self.lock['sequence_scope'], limitations=self.lock['limitations'],
            versions=json.loads((self.output/'build_identity.json').read_text())['versions'])
        write_json(self.output / 'manifest.json', manifest)
        # Exercise the actual public reader and compare its payload to source rows.
        read_checks = {}
        for name in ('train','test'):
            reader = IndexedSplit(self.output, name, cache_row_groups=2, allow_incomplete=True)
            indices = sorted(set([0,len(reader)-1] + np.random.default_rng(self.config['seed']).integers(0,len(reader),size=30).tolist()))
            for index in indices:
                row = reader[index]
                a = row['split_annotation']; source = self.registry['shards'][str(a['shard_id'])]
                starts = source['row_group_starts']; group = int(np.searchsorted(starts,a['row_index'],side='right')-1)
                direct = pq.ParquetFile(Path(self.registry['roots'][source['root']])/source['path']).read_row_group(group).slice(a['row_index']-starts[group],1).to_pylist()[0]
                for field in ('seq1','seq2','contacts','contacts_8A','valid1','valid2','label','pair_id'):
                    if field in direct and row[field] != direct[field]:
                        raise ValueError('Reader changed source payload: ' + field)
            read_checks[name] = len(indices)
        verification = dict(state='passed', source_counts=source_counts, reader_checks=read_checks,
            sequence_inputs=json.loads((self.output/'sequence_inputs.done.json').read_text())['checks'],
            test_rows=n, no_validation_split=True, training_started=False)
        write_json(self.output / 'verification.json', verification)
        self.write_readme(manifest, verification)
        tracked = ['duplicate_map.parquet','known_positive_cluster_pairs.parquet','cluster_bridges.parquet',
            'excluded_homology.index.parquet','excluded_empty_contacts.index.parquet','selected_sources.index.parquet',
            'legacy_before_after.index.parquet','verification.json','README.md','filter_protocol.lock.json','build_identity.json',
            'sequence_inputs.done.json','test_reference.fasta']
        if (self.output/'input_revision_review.json').exists():
            tracked.append('input_revision_review.json')
        for kind in ('ppi','ddi'):
            tracked.extend(kind + suffix for suffix in ('_homology_hits.parquet','_homology_evidence.parquet',
                '_homology.json','_sequence_catalog.parquet','_candidate.fasta'))
            for direction in self.lock['search']['directions']:
                tracked.extend(f'{kind}_{direction}' + suffix for suffix in ('.tsv.gz','.search.json','_mmseqs.log'))
        manifest['artifacts'] = {name: dict(sha256=file_sha256(self.output/name),bytes=(self.output/name).stat().st_size) for name in tracked}
        write_json(self.output/'manifest.json', manifest)
        # The independent auditor completes acceptance, including a second parse
        # of every raw alignment and checks on final membership and sampling weights.
        from audit_expanded_index_split import audit_strict
        self.status('independent_audit')
        report = audit_strict(self.output)
        self.mark('complete', rows=splits['train']['rows'])
        write_json(self.output/'status.json',dict(state='complete',train_rows=splits['train']['rows'],test_rows=n,
            audit_status=report['state'],training_started=False))

    def write_readme(self, manifest, audit):
        title = 'RF2-PPI 隔离规则 v3' if self.rf2 else '固定 train–test 索引 v2'
        lines = ['# CongLab + 蒸馏数据：'+title,'',
            '2026-10-24 用户授权的新协议；保留历史版本。只生成索引与证据，不复制训练 payload，不启动训练。','',
            f"Train {manifest['splits']['train']['rows']:,}；E. coli test {manifest['splits']['test']['rows']:,}（650 正例、65,000 负例）；无 val。",'',
            '| 类型 | 来源 | 原始候选 | 同源排除 | 无有效接触 | 完全重复 | 最终 train |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in manifest['source_counts']:
            lines.append('| ' + ' | '.join(str(r[k]) for k in ('kind','source','input_rows','homology_excluded','empty_contacts','duplicates','rows')) + ' |')
        search_rule = ('双方方向各做 k=6 检索和关闭预筛选的全量比对（prefilter-mode=2），合格命中取并集。完全相同序列也必须通过 E-value 门槛，无额外豁免。'
            if self.rf2 else '双方方向各做 k=6 检索和关闭预筛选的全量比对（prefilter-mode=2），四次命中取并集，并加入完全相同序列。')
        hit_rule = ('PPI/DDI 统一：identity = nident/alnlen 严格 >30%，MMseqs 报告的 E-value 严格 <1e-5，query coverage >50% 或 hit coverage >50%。三个条件同时满足。'
            if self.rf2 else 'identity = nident/alnlen 严格 >30%；PPI 两条比对序列 coverage 都 ≥50%；DDI 训练域 coverage ≥50%，不限制 test 蛋白 coverage。')
        boundary_rule = ('coverage = (end-start+1)/完整 FASTA 条目长度；1-based inclusive。identity/coverage 用整数比较；恰好 30%、恰好 50%、恰好 E-value=1e-5 均不满足相应条件。'
            if self.rf2 else 'coverage = (end-start+1)/完整 FASTA 条目长度；坐标从 1 开始且包含端点。整数比较，恰好 30% 不排除、恰好 50% 满足覆盖度。')
        pair_rule = ('参考是冻结 test 正负样本两侧全部序列。训练 pair 的两端分别有合格命中才排除整对；只有一端命中保留；不要求命中同一个 test pair。'
            if self.rf2 else '参考是冻结 test 正负样本两侧全部序列。任一训练端点命中参考集合就排除整对；不要求命中同一个 test pair。')
        lines += ['', '## 过滤条件与复现','',
            '`filter_protocol.lock.json` 固定协议、来源文件 SHA256、MMseqs 18.8cc5c 二进制 SHA256 和所有显式搜索参数。',
            search_rule,
            'k-mer 检索固定 k=6、k-mer score=96、sensitivity=7.5；所有步骤固定 8 CPU 线程、shuffle=0、mask=0。关闭预筛选的步骤不依赖 k-mer 筛选。',
            'max-seqs 始终等于目标 FASTA 的全部序列数；split=1、split-mode=0、split-memory-limit=16G。日志验证实际 k/score/version。',
            '所有 CongLab 和蒸馏样本统一重新检索，不沿用旧命中或旧未命中缓存，也不豁免旧训练成员。',
            hit_rule, boundary_rule,
            'pair CSV 中 seq1/seq2 ID 对应的 FASTA 与 Parquet 实际输入序列逐条一致，相关检验在 verification.json。',
            'FASTA 条目可为来源结构片段或结构域；不是重建的 UniProt 全长蛋白。先完成隔离，之后训练才允许 crop。',
            pair_rule,
            '测试集内容、顺序、标签和原始 SHA256 不变。旧 val 不再作为 holdout；其中来源样本与其他候选接受相同 test 过滤。',
            '全量比对复核避免 k-mer 预筛选漏检；仍以固定 MMseqs 比对打分和定义为准，不等于生物学上不存在远缘同源。',
            '并行输出的原始 TSV 行序可能变化；最终命中集合和 train 索引排序后固定。历史 test 多次评估，不是新的盲测集。','',
            '## 去重与采样','',
            '结构来源按 structure_source 筛选，仅 CongLab + 8 个预测 PPI 来源；PINDER/Teddymer 不进入 train。',
            '源数据的质量准入保留，不擅自新增质量阈值。Cβ<12Å、Gly Cα、未知 mask 与序列轴保持源数据语义。',
            '序列、接触标签、mask 完全相同才合并；交换链时同步转置接触图。不同接触图分别保留，绝不取并集。',
            '完全重复优先保留 CongLab，再按 sample_id 排序选代表。duplicate_map.parquet 逐条记录副本→代表。',
            '同序列对每个保留监督的 within_pair_weight=1/该对保留监督数，组内总和=1。',
            'PPI 与 DDI 均保留并核验已提供的 70% identity/70% coverage 序列聚类 TSV；训练时按序列簇均衡、簇内按去重权重。',
            '两种 cluster ID 先分命名空间，再仅通过完全相同序列合并。原始簇在 source_seq1_cluster/source_seq2_cluster 中保留。',
            '70% 聚类用于采样，不代替 30% test 隔离。DDI 结构簇仅保留为注释，不是本次拟用的采样依据。',
            'known_positive_cluster_pairs.parquet 只由最终 train 正结构建立，不读 test 标签来构建训练负例屏蔽。','',
            '## 文件与读取','',
            '- train.index.parquet / test.index.parquet：原始分片与行位置，训练分组与权重；没有 val 文件。',
            '- sources.json：源数据位置、哈希、行组索引。迁移可通过 IndexedSplit 的 root_overrides 重定位。',
            '- manifest.json：版本、计数、协议与产物 SHA256；audit_status=passed 才表示已验收。',
            '- excluded_homology.index.parquet / excluded_empty_contacts.index.parquet：排除成员与原因。',
            '- *_homology_evidence.parquet：所有合格比对的整数 identity/coverage 分子分母与两端 ID。',
            '- *.tsv.gz、*.search.json、*_mmseqs.log：原始比对、完整命令与日志。',
            '- legacy_before_after.index.parquet：每条旧训练成员在新规则下的去向，不再保证 100% 保留。',
            '- verification.json / independent_audit.json：数据对应关系、读取一致性与独立验收。','',
            '```python','from flashppi.indexed_data import IndexedSplit',
            f"train = IndexedSplit({str(self.output)!r}, 'train')", "sample = train[0]",'```','',
            '```bash','PYTHONPATH=artifacts/environments/data_index_deps \\',
            'artifacts/environments/esmc_prott5_gated_py312/bin/python \\',
            f'  tools/train_test.py --mode build --config {self.config_file}',
            '```','', '只检查已有划分：','',
            '```bash','PYTHONPATH=artifacts/environments/data_index_deps \\',
            'artifacts/environments/esmc_prott5_gated_py312/bin/python \\',
            f'  tools/train_test.py --mode check --config {self.config_file}',
            '```','', 'build 完成隔离并自动验收；check 独立重算归档比对与成员约束，不重新运行 MMseqs，不修改成员。',
            '检查失败返回非零退出码并写入 check_status.json；不会自动修复成员、更新参数或重建数据。',
            '构建重跑拒绝代码、配置、锁文件或运行依赖漂移。构建入口更新后，可用 check 检查已有划分。',
            'build_scratch 仅用于恢复，可在验收后清理；正式索引读取不依赖 scratch。训练入口与超参数需在后续阶段接入。','']
        if self.rf2:
            lines += ['## 论文规定与本地实现的边界','',
                '[RF2-PPI 补充方法 M3.1 第20–21页、M3.3 第27页](https://pmc.ncbi.nlm.nih.gov/articles/instance/13281779/bin/NIHMS2182373-supplement-supplemental_text_and_figures.pdf)明确了上述 identity/E-value/coverage/双端逻辑。',
                '参考集合按用户要求换为既有 E. coli test；不是重新使用作者的人类参考库。MMseqs 完整搜索命令和版本未公开，本锁记录的是本地实现，不能宣称逐参数复现原作者。',
                '为实现 coverage OR，MMseqs 搜索不施加覆盖度预筛（-c 0），输出后严格判定。-e 0.00001，输出包含 evalue,bits。每个方向使用完整目标库，E-value 随库规模变化，因此库内容、方向、参数均固定。',
                '只有双端命中才排除，因此允许单端同源，不能宣称 C3。验收报告分别统计允许的单端命中与应为零的双端命中。',
                '本次只替换隔离规则；采样仍使用用户指定的 70/70 TSV，不改成论文的 30/80 聚类。','']
        (self.output/'README.md').write_text('\n'.join(lines))


def check_existing(config):
    """Recheck formal evidence and members, never instantiate a data builder."""
    output = Path(config['output'])
    manifest_path = output/'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('No completed split to check: ' + str(output))
    previous = json.loads(manifest_path.read_text())
    if previous.get('state') != 'complete':
        raise ValueError('Check mode requires a complete split; wait for the build to finish')
    if previous.get('config') != config:
        raise ValueError('Completed output belongs to another configuration')
    with ExitStack() as stack:
        if config.get('filter_protocol') in FRESH_PROTOCOL_IDS:
            import fcntl
            guard = stack.enter_context((output/'.build.lock').open('a+'))
            try:
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A build or check is already running for this output') from None
        from audit_expanded_index_split import audit
        status = dict(mode='check', checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            entrypoint_sha256=file_sha256(__file__), membership_modified=False,
            filter_protocol=previous.get('filter_protocol'))
        try:
            report = audit(output)
        except BaseException as exc:
            # Keep the last successful detailed report for provenance, while
            # making the latest failed check explicit and blocking acceptance.
            write_json(output/'check_status.json', dict(status, state='failed', error=repr(exc)))
            current = json.loads(manifest_path.read_text())
            current.update(audit_status='failed', latest_check='check_status.json')
            write_json(manifest_path, current)
            raise
        write_json(output/'check_status.json', dict(status, state=report['state'],
            report='independent_audit.json'))
        current = json.loads(manifest_path.read_text())
        current.update(audit_status=report['state'], latest_check='check_status.json')
        write_json(manifest_path, current)
        print('Check passed: ' + str(output/'check_status.json'), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['build', 'check'], default='build',
                        help='build: construct and verify; check: verify existing members only')
    parser.add_argument('--through', choices=['homology', 'all'], default='all',
                        help='build mode only: stop after homology or complete the split')
    args = parser.parse_args(argv)
    if args.mode == 'check' and args.through != 'all':
        parser.error('--through is only meaningful in build mode')
    config = json.loads(args.config.read_text())
    strict = config.get('filter_protocol') in FRESH_PROTOCOL_IDS
    (validate_strict_config if strict else validate_config)(config)
    if args.mode == 'check':
        check_existing(config)
        return
    if strict:
        import fcntl
        output = Path(config['output'])
        output.mkdir(parents=True, exist_ok=True)
        build_guard = (output / '.build.lock').open('a+')
        fcntl.flock(build_guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = Path(config['output'])/'manifest.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get('state') == 'complete':
            if previous.get('filter_protocol') != (verify_strict_inputs if strict else verify_baseline)(config):
                raise ValueError('Completed index used a different filtering protocol; rebuild under the original protocol')
            if previous['config'] != config:
                raise ValueError('Completed output belongs to another configuration; use a new output directory')
            if strict:
                identity = json.loads((existing.parent/'build_identity.json').read_text())
                if any(file_sha256(ROOT/name) != digest for name,digest in identity['code_sha256'].items()):
                    raise ValueError('Completed build code changed; use --mode check instead of rebuilding the same version')
                from audit_expanded_index_split import audit_strict
                audit_strict(existing.parent)
            print('Already complete: '+str(existing),flush=True)
            return
    build = (StrictBuild if strict else Build)(args.config)
    try:
        build.catalog()
        build.homology()
        if args.through == 'homology':
            build.status('homology_complete')
            return
        build.candidates()
        build.contents()
        build.finalize()
    except BaseException as exc:
        write_json(build.output / 'status.json', dict(state='failed', error=repr(exc)))
        raise
    finally:
        build.db.close()


if __name__ == '__main__':
    main()
