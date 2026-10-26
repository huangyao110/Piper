"""CPU regression checks for index identity, overlap handling, and old-data retention."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'tools')]
from flashppi.indexed_data import contact_fingerprint, file_sha256, sequence_id, IndexedSplit
from train_test import Build, payload_buffers
from flashppi.data_protocol import PROTOCOL_ID, validate_config


def make_strict_fixture(tmp_path, monkeypatch, rf2=False):
    """Use synthetic immutable files; production has no verification bypass flag."""
    import csv
    import gzip
    import os
    import shutil
    from flashppi.data_protocol import (read_strict_lock, STRICT_PROTOCOL_ID, STRICT_LOCK_SHA256,
                                       RF2_PROTOCOL_ID, RF2_LOCK_SHA256)
    executable = os.environ.get('PIPER_MMSEQS') or shutil.which('mmseqs')
    if executable is None and Path('/home/zlab/miniconda3/bin/mmseqs').is_file():
        executable = '/home/zlab/miniconda3/bin/mmseqs'
    if not executable or not Path(executable).is_file():
        pytest.skip('Real MMseqs integration requires an executable; set PIPER_MMSEQS to its path')
    path, sequences = make_fixture(tmp_path)
    original = json.loads(path.read_text())
    release = Path(original['release'])
    old_meta = json.loads((release/'manifest.json').read_text())
    entries = []
    for old_entry in old_meta['metadata_shards']:
        source = release/'components/training'/old_entry['base_file']
        rows = pq.read_table(source).to_pylist()
        kind = old_entry['kind']
        for row in rows:
            row['sample_id'] = row['pair_id']
            if row['structure_source'] == 'pinder' and row['pair_id'] != 'new_unique':
                row['source'] = row['structure_source'] = 'conglab_pred'
            if row['structure_source'] == 'teddymer' and row['pair_id'] != 'new_ddi_unique':
                row['source'] = row['structure_source'] = 'conglab'
        pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=2)
        counts = {s: sum(r['structure_source']==s for r in rows) for s in {r['structure_source'] for r in rows}}
        entries.append(dict(file=str(source.relative_to(release)),kind=kind,sha256=file_sha256(source),
                            records=len(rows),sources=sorted(counts),source_counts=counts))
        seqs = {r[f'seq{s}_id']: (r[f'seq{s}'],r[f'seq{s}_cluster']) for r in rows for s in (1,2)}
        (release/(kind+'_seqs.fasta')).write_text(''.join(f'>{sid}\n{seq}\n' for sid,(seq,cluster) in sorted(seqs.items())))
        (release/(kind+'_sequence_clusters.tsv')).write_text('sequence_id\tcluster_id\trepresentative\n'+
            ''.join(f'{sid}\t{cluster}\t{sid}\n' for sid,(seq,cluster) in sorted(seqs.items())))
        with gzip.open(release/(kind+'_pairs.csv.gz'),'wt') as f:
            writer=csv.writer(f);writer.writerow(['pair_id','seq1','seq2','source'])
            writer.writerows([r['pair_id'],r['seq1_id'],r['seq2_id'],r['source']] for r in rows)
    (release/'DATASET.json').write_text(json.dumps(dict(state='complete',parquet_shards=entries)))
    # A sequence present only in a negative test row must also exclude a train endpoint.
    pq.write_table(pa.Table.from_pylist([
        dict(seq1=sequences[10],seq2=sequences[11],label=1),
        dict(seq1=sequences[17],seq2=sequences[11],label=0)]),original['test'])
    protocol_id = RF2_PROTOCOL_ID if rf2 else STRICT_PROTOCOL_ID
    lock = read_strict_lock(protocol_id)
    lock['test'].update(rows=2,positives=1,negatives=1)
    config = {**lock['required_config'], **{k:original[k] for k in ('release','legacy_directory','test','output','seed')},
        'test_sha256':file_sha256(original['test']),'version':'fixture_strict','threads':8,
        'mmseqs':executable,'duckdb_memory_limit':'1GB'}
    path.write_text(json.dumps(config))
    protocol=dict(id=protocol_id,lock_sha256=RF2_LOCK_SHA256 if rf2 else STRICT_LOCK_SHA256)
    monkeypatch.setattr('train_test.verify_strict_inputs',lambda c:protocol)
    monkeypatch.setattr('train_test.read_strict_lock',lambda *args:lock)
    monkeypatch.setattr('audit_expanded_index_split.verify_strict_inputs',lambda c:protocol)
    return path,sequences


@pytest.mark.integration
@pytest.mark.parametrize('rf2', [False, True])
def test_strict_end_to_end_fasta_clusters_test_and_deduplication(tmp_path, monkeypatch, rf2):
    from train_test import StrictBuild
    path,seqs = make_strict_fixture(tmp_path,monkeypatch,rf2)
    b=StrictBuild(path)
    try:
        b.catalog(); b.homology(); b.candidates(); b.contents(); b.finalize()
    finally:
        b.db.close()
    out=Path(json.loads(path.read_text())['output'])
    manifest=json.loads((out/'manifest.json').read_text())
    assert set(manifest['splits'])=={'train','test'} and manifest['audit_status']=='passed'
    train=pq.read_table(out/'train.index.parquet');ids=set(train['sample_id'].to_pylist())
    assert ('new_test_one_hit' in ids) == rf2
    assert 'new_test_both_hit' not in ids
    assert 'new_unique' not in ids and 'new_ddi_unique' not in ids  # PINDER/Teddymer
    assert 'new_empty' not in ids
    assert {'new_conformation','new_mask','fixed_val'} <= ids
    assert ('unselected_conglab' in ids) == rf2  # Only one endpoint hits a negative test row.
    report=json.loads((out/'independent_audit.json').read_text())
    assert report['retained_rule_violations']==0
    assert report['retained_single_endpoint_hits']==(2 if rf2 else 0)
    if rf2:
        for kind in ('ppi','ddi'):
            evidence=pq.read_table(out/(kind+'_homology_evidence.parquet'))
            assert all(e<1e-5 for e in evidence['evalue'].to_pylist())
            assert 'exact' not in evidence['direction'].to_pylist()
    assert len({'old_ppi_a','old_ppi_b','new_exact_swapped'} & ids)==1
    assert len({'old_ddi','new_ddi_duplicate'} & ids)==1
    assert pq.read_table(out/'test.index.parquet')['label'].to_pylist()==[1,0]
    reader=IndexedSplit(out,'train')
    for i in range(len(reader)):
        row=reader[i]
        assert row['seq1_id']==sequence_id(row['seq1'])
    # A changed cluster TSV is a hard data error, not an invitation to recluster.
    # The first build already checked every selected mapping against that TSV.
    checks=json.loads((out/'verification.json').read_text())['sequence_inputs']
    assert all(checks[k]['fasta_matches']==checks[k]['cluster_tsv_matches'] for k in ('ppi','ddi'))

    # The public check mode must independently reassess an existing split,
    # without constructing a builder or touching any membership index.
    import train_test as entrypoint
    def no_build(*args, **kwargs):
        raise AssertionError('Check mode attempted to build data')
    monkeypatch.setattr(entrypoint, 'StrictBuild', no_build)
    monkeypatch.setattr(entrypoint, 'validate_strict_config', lambda c: manifest['filter_protocol'])
    index_files = sorted(out.glob('*.index.parquet')) + [out/'duplicate_map.parquet']
    before = {p.name: file_sha256(p) for p in index_files}
    entrypoint.main(['--mode', 'check', '--config', str(path)])
    assert {p.name: file_sha256(p) for p in index_files} == before
    assert json.loads((out/'check_status.json').read_text())['state'] == 'passed'

    # Updating the checksum cannot hide a real endpoint leak from the auditor.
    # This is a deliberately corrupted toy fixture, never a production bypass.
    train_path, manifest_path = out/'train.index.parquet', out/'manifest.json'
    original_train, original_manifest = train_path.read_bytes(), manifest_path.read_bytes()
    leak_id = 'new_test_both_hit' if rf2 else 'new_test_one_hit'
    leaked = next(r for r in pq.read_table(out/'selected_sources.index.parquet').to_pylist()
                  if r['sample_id'] == leak_id)
    added = train.slice(0, 1).to_pylist()[0]
    added.update({k: leaked[k] for k in ('sample_id', 'kind', 'shard_id', 'row_index', 'seq1_id', 'seq2_id')})
    changed = pa.concat_tables([train, pa.Table.from_pylist([added], schema=train.schema)])
    pq.write_table(changed, train_path)
    changed_manifest = json.loads(original_manifest)
    changed_manifest['splits']['train'].update(rows=len(changed), sha256=file_sha256(train_path))
    manifest_path.write_text(json.dumps(changed_manifest))
    try:
        with pytest.raises(ValueError, match='Retained training pair violates'):
            entrypoint.main(['--mode', 'check', '--config', str(path)])
        assert json.loads((out/'check_status.json').read_text())['state'] == 'failed'
        assert json.loads(manifest_path.read_text())['audit_status'] == 'failed'
    finally:
        train_path.write_bytes(original_train)
        manifest_path.write_bytes(original_manifest)

    # A direct pass must account for every length-compatible sequence pair,
    # even if a modified log has a freshly matching artifact checksum.
    import re
    log_path = out/'ddi_test_to_train_nofilter_mmseqs.log'
    original_log = log_path.read_bytes()
    changed_log = re.sub(r'(\d+) alignments calculated',
        lambda match: str(int(match.group(1))-1)+' alignments calculated',
        original_log.decode())
    assert changed_log.encode() != original_log
    log_path.write_text(changed_log)
    changed_manifest = json.loads(original_manifest)
    changed_manifest['artifacts'][log_path.name]['sha256'] = file_sha256(log_path)
    manifest_path.write_text(json.dumps(changed_manifest))
    try:
        with pytest.raises(ValueError, match='Full alignment count differs'):
            entrypoint.main(['--mode', 'check', '--config', str(path)])
    finally:
        log_path.write_bytes(original_log)
        manifest_path.write_bytes(original_manifest)


def test_check_mode_requires_existing_output_and_never_creates_it(tmp_path):
    import train_test as entrypoint
    config = json.loads((ROOT/'configs_data/conglab_distilled_ecoli_strict_v2.json').read_text())
    output = tmp_path/'absent_split'
    config['output'] = str(output)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match='No completed split to check'):
        entrypoint.main(['--mode', 'check', '--config', str(path)])
    assert not output.exists()
    with pytest.raises(SystemExit) as error:
        entrypoint.main(['--mode', 'check', '--through', 'homology', '--config', str(path)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize('state,change_config', [('building', False), ('complete', True)])
def test_check_mode_rejects_incomplete_or_other_configuration(tmp_path, state, change_config):
    import train_test as entrypoint
    config = json.loads((ROOT/'configs_data/conglab_distilled_ecoli_strict_v2.json').read_text())
    output = tmp_path/'split'
    output.mkdir()
    config['output'] = str(output)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    saved = dict(config)
    if change_config:
        saved['seed'] += 1
    (output/'manifest.json').write_text(json.dumps(dict(state=state, config=saved)))
    with pytest.raises(ValueError, match='complete|configuration'):
        entrypoint.main(['--mode', 'check', '--config', str(path)])


def test_chain_swap_mask_and_conformation_semantics():
    a,b = 'ACDE','GHIKL'
    va,vb = [True,False,True,True],[True,True,False,True,True]
    xy = np.array([[0,1],[2,3]])
    h = contact_fingerprint(a,b,va,vb,xy)
    assert h == contact_fingerprint(b,a,vb,va,xy[:,::-1])
    assert h != contact_fingerprint(a,b,[True]*4,vb,xy)
    assert h != contact_fingerprint(a,b,va,vb,[[0,1],[3,3]])
    assert h == contact_fingerprint(a,b,va,vb,[[2,3],[0,1],[0,1]])
    # A homomer has no sequence-order tie breaker; its two orientations agree.
    h = contact_fingerprint(a,a,[True]*4,[False,True,True,True],[[0,1]])
    assert h == contact_fingerprint(a,a,[False,True,True,True],[True]*4,[[1,0]])


def test_arrow_buffers_respect_sliced_offsets_and_unknown_positions():
    t = pa.table({'contacts':pa.array([[[0,0]],[[0,0],[1,1]],[]],
                                     type=pa.list_(pa.list_(pa.int32(),2))),
                  'valid1':[[True],[False,True],[True]],
                  'valid2':[[True],[True,True],[False]]})
    batch = t.to_batches()[0].slice(1,2)
    co,xy,off,masks,counts = payload_buffers(batch)
    assert co.tolist() == [0,2,2]
    assert xy.tolist() == [[0,0],[1,1]]
    assert counts.tolist() == [1,0]
    assert masks[0][off[0][0]:off[0][1]].tolist() == [False,True]


def make_fixture(tmp_path):
    release,legacy,out = tmp_path/'release',tmp_path/'legacy',tmp_path/'split'
    training = release/'components/training'
    legacy.mkdir(parents=True)
    rng=np.random.default_rng(7)
    sequences=[''.join(rng.choice(list('ACDEFGHIKLMNPQRSTVWY'),size=32)) for _ in range(24)]
    seq_clusters={sequence_id(s):i for i,s in enumerate(sequences)}
    def row(name,a,b,kind='ppi',source='pinder',contacts=None,mask=None):
        sa,sb=sequence_id(sequences[a]),sequence_id(sequences[b])
        return dict(pair_id=name,kind=kind,seq1=sequences[a],seq2=sequences[b],
            seq1_id=sa,seq2_id=sb,sequence_pair_id=kind+'_'+hashlib.sha256('\0'.join(sorted([sa,sb])).encode()).hexdigest(),
            source=source,structure_source=source,contacts=[[0,1],[2,3]] if contacts is None else contacts,
            valid1=[True]*32 if mask is None else mask,valid2=[True]*32,
            seq1_cluster=seq_clusters[sa],seq2_cluster=seq_clusters[sb],len1=32,len2=32,
            sequence_cluster_scope=kind+'_sequence_70_70',structure_cluster_id=None,structure_cluster_scope=None)
    old=[row('old_ppi_a',0,1,source='conglab'),row('old_ppi_b',0,1,source='conglab'),
         row('old_ddi',2,3,kind='ddi',source='conglab')]
    new=[row('new_exact_swapped',1,0,contacts=[[1,0],[3,2]]),
         row('new_conformation',0,1,contacts=[[0,1],[3,4]]),
         row('new_mask',0,1,mask=[True]*31+[False]),
         row('new_unique',4,5),row('new_val_hit',6,8),
         row('new_test_one_hit',10,12),row('new_test_both_hit',10,11),
         row('new_empty',13,14,contacts=[]),
         row('new_ddi_duplicate',2,3,kind='ddi',source='teddymer'),
         row('new_ddi_unique',0,16,kind='ddi',source='teddymer'),
         row('unselected_conglab',17,18,source='conglab')]
    structural_val=[row('fixed_val',6,7,source='conglab')]
    raw=old+new+structural_val
    entries=[]
    for kind in ('ppi','ddi'):
        records=[r for r in raw if r['kind']==kind]
        schema=pa.schema([('pair_id',pa.string()),('kind',pa.string()),('seq1',pa.string()),('seq2',pa.string()),
            ('seq1_id',pa.string()),('seq2_id',pa.string()),('sequence_pair_id',pa.string()),
            ('source',pa.string()),('structure_source',pa.string()),('contacts',pa.list_(pa.list_(pa.int32(),2))),
            ('valid1',pa.list_(pa.bool_())),('valid2',pa.list_(pa.bool_())),
            ('seq1_cluster',pa.int64()),('seq2_cluster',pa.int64()),('len1',pa.int64()),('len2',pa.int64()),
            ('sequence_cluster_scope',pa.string()),('structure_cluster_id',pa.int64()),('structure_cluster_scope',pa.string())])
        t=pa.Table.from_pylist(records,schema=schema)
        base=training/f'parquet/{kind}/part.parquet';base.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(t,base,row_group_size=2)
        meta=t.drop(['seq1','seq2','contacts','valid1','valid2']).append_column('sample_id',t['pair_id'])
        mp=release/f'tables/{kind}.parquet';mp.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(meta,mp,row_group_size=3)
        entries.append(dict(file=str(mp.relative_to(release)),base_file=str(base.relative_to(training)),
            kind=kind,records=len(t),sha256=file_sha256(mp),base_sha256=file_sha256(base)))
    outputs={}
    val_interactions=[dict(seq1=sequences[6],seq2=sequences[7],label=1),
                      dict(seq1=sequences[7],seq2=sequences[6],label=0)]
    for name,records in [('train',old),('validation',structural_val),('interaction',val_interactions)]:
        p=legacy/(name+'.parquet');pq.write_table(pa.Table.from_pylist(records),p,row_group_size=2)
        outputs[p.name]=dict(sha256=file_sha256(p),rows=len(records))
    test=tmp_path/'test.parquet'
    pq.write_table(pa.Table.from_pylist([dict(seq1=sequences[10],seq2=sequences[11],label=1),
                                       dict(seq1=sequences[11],seq2=sequences[10],label=0)]),test,row_group_size=1)
    (legacy/'manifest.json').write_text(json.dumps(dict(state='complete',outputs=outputs)))
    (training/'DATASET.json').write_text(json.dumps(dict(state='complete')))
    (release/'manifest.json').write_text(json.dumps(dict(state='complete',metadata_shards=entries,
        counts={k:sum(r['kind']==k for r in raw) for k in ('ppi','ddi')})))
    config=dict(version='fixture',release=str(release),legacy_directory=str(legacy),test=str(test),output=str(out),
        mmseqs='unused',threads=1,seed=13,expected_legacy={'ppi':2,'ddi':1},test_sha256=file_sha256(test),
        new_sources=['pinder','teddymer'],test_exclusion='both_chains',val_exclusion='either_chain',
        ppi_coverage='both_coverage_50',ddi_coverage='candidate_coverage_80',preserve_all_legacy=True,
        duckdb_memory_limit='1GB',quality_policy='reuse',filter_protocol=PROTOCOL_ID)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    return path,sequences


def test_complete_index_build_preserves_old_and_distinct_supervision(tmp_path, monkeypatch):
    path,seqs=make_fixture(tmp_path)
    # The synthetic dataset tests builder behavior, not production file identity.
    # Production has no option to bypass the frozen baseline verification.
    monkeypatch.setattr('train_test.verify_baseline', validate_config)
    b=Build(path)
    try:
        b.catalog()
        # Deterministic fixture hits isolate split/dedup logic; production runs
        # actual MMseqs and the existing homology tests cover threshold boundaries.
        ids=[sequence_id(seqs[i]) for i in [6,7,10,11]]
        hits=pa.table({'seq_id':ids,'test_hit':[False,False,True,True],'val_hit':[True,True,False,False]})
        for kind in ('ppi','ddi'):
            pq.write_table(hits,b.output/(kind+'_homology_hits.parquet'))
        b.candidates();b.contents();b.finalize()
    finally:
        b.db.close()
    index=pq.read_table(b.output/'train.index.parquet')
    ids=set(index['sample_id'].to_pylist())
    assert ids=={'old_ppi_a','old_ppi_b','old_ddi','new_conformation','new_mask','new_unique',
                 'new_test_one_hit','new_ddi_unique'}
    aliases={r['duplicate_id']:r for r in pq.read_table(b.output/'duplicate_map.parquet').to_pylist()}
    assert aliases['new_exact_swapped']['canonical_id']=='old_ppi_a'
    assert aliases['old_ppi_b']['action']=='retained_legacy_grouped'
    assert aliases['new_ddi_duplicate']['canonical_id']=='old_ddi'
    assert all(x>0 for x in index['within_pair_weight'].to_pylist())
    assert json.loads((b.output/'verification.json').read_text())['legacy_retention_rate']==1
    reader=IndexedSplit(b.output,'train',cache_row_groups=1)
    assert len(reader)==8
    with pytest.raises(IndexError):reader[8]
    assert reader[-1]['sample_id']==index['sample_id'][-1].as_py()
    for split in ('validation_contact','validation_interaction','test'):
        actual=IndexedSplit(b.output,split)
        assert len(actual)==(1 if split=='validation_contact' else 2)
    # Registry corruption must fail before any source can be substituted.
    registry=b.output/'sources.json';original=registry.read_bytes();registry.write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='registry checksum'):
        IndexedSplit(b.output,'train')
    registry.write_bytes(original)
    by_id={r['sample_id']:r for r in index.to_pylist()}
    assert by_id['old_ppi_a']['seq1_cluster']==by_id['new_ddi_unique']['seq1_cluster']
    from audit_expanded_index_split import audit
    monkeypatch.setattr('audit_expanded_index_split.verify_baseline', validate_config)
    independent=audit(b.output)
    assert independent['state']=='passed'
    assert independent['old_train_count']==3
    assert independent['new_train_rows']==5
    assert sum(r['action_count'] for r in independent['duplicate_origin_counts'])==3
    manifest_path=b.output/'manifest.json'
    original_manifest=manifest_path.read_text()
    unsupported=json.loads(original_manifest)
    unsupported.pop('filter_protocol')
    manifest_path.write_text(json.dumps(unsupported))
    with pytest.raises(ValueError,match='approved locked filtering protocol'):
        IndexedSplit(b.output,'train')
    manifest_path.write_text(original_manifest)
    (b.output/'FILTER_PROTOCOL_RETIRED.json').write_text('{}')
    with pytest.raises(ValueError,match='retired filtering protocol'):
        IndexedSplit(b.output,'train')


def test_legacy_filter_decision_drift_stops_the_build(tmp_path, monkeypatch):
    path,seqs=make_fixture(tmp_path)
    monkeypatch.setattr('train_test.verify_baseline', validate_config)
    b=Build(path)
    try:
        b.catalog()
        # A newly detected val hit on an original train sequence must not silently
        # enter the approved legacy protocol, even though preserve_all_legacy=True.
        hits=pa.table({'seq_id':[sequence_id(seqs[0])],'test_hit':[False],'val_hit':[True]})
        for kind in ('ppi','ddi'):
            pq.write_table(hits,b.output/(kind+'_homology_hits.parquet'))
        with pytest.raises(ValueError,match='Original training filter decisions changed'):
            b.candidates()
    finally:
        b.db.close()
