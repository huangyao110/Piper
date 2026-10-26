"""Independent membership/holdout audit of a completed indexed expansion."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from piper.ddi_common import write_json
from piper.indexed_data import file_sha256,sequence_id
from piper.data_protocol import (verify_baseline, verify_strict_inputs, STRICT_PROTOCOL_ID,
    RF2_PROTOCOL_ID, FRESH_PROTOCOL_IDS, read_strict_lock)


def audit(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('filter_protocol', {}).get('id') in FRESH_PROTOCOL_IDS:
        return audit_strict(directory)
    if manifest['state']!='complete':
        raise ValueError('Wait for the index build to complete')
    protocol=verify_baseline(manifest['config'])
    if manifest.get('filter_protocol')!=protocol:
        raise ValueError('Index does not use the user-required original filtering protocol')
    legacy_dir=Path(manifest['config']['legacy_directory'])
    if file_sha256(legacy_dir/'train.parquet')!=manifest['legacy_train_sha256']:
        raise ValueError('Legacy training file changed')
    if file_sha256(directory/'train.index.parquet')!=manifest['splits']['train']['sha256']:
        raise ValueError('Training index changed')
    old=pq.read_table(legacy_dir/'train.parquet',columns=['pair_id','kind','seq1_id','seq2_id'])
    old_ids=set(old['pair_id'].to_pylist())
    observed_old=set()
    new_rows=0
    for batch in pq.ParquetFile(directory/'train.index.parquet').iter_batches(
            batch_size=65536,columns=['sample_id','legacy_train']):
        t=pa.Table.from_batches([batch]);flags=t['legacy_train']
        selected=t.filter(flags)['sample_id'].to_pylist()
        if observed_old.intersection(selected):
            raise ValueError('Old training ID repeated in index')
        observed_old.update(selected)
        new_rows+=batch.num_rows-len(selected)
    if observed_old!=old_ids:
        raise ValueError('Independent legacy membership audit failed')
    refs={}
    for category,paths in [('val',[legacy_dir/'validation.parquet',legacy_dir/'interaction.parquet']),
                           ('test',[Path(manifest['config']['test'])])]:
        refs[category]=set()
        for p in paths:
            t=pq.read_table(p,columns=['seq1','seq2'])
            refs[category].update(sequence_id(s) for s in set(t['seq1'].to_pylist()+t['seq2'].to_pylist()))
    summaries={};overlaps=[];hit_hashes={}
    for kind in ('ppi','ddi'):
        t=old.filter(pc.equal(old['kind'],kind))
        path=directory/(kind+'_homology_hits.parquet')
        hits=pq.read_table(path);hit_hashes[kind]=file_sha256(path)
        if hit_hashes[kind]!=manifest['artifacts'][path.name]['sha256']:
            raise ValueError('Homology audit input changed')
        selected={name:hits.filter(hits[name+'_hit'])['seq_id'] for name in ('test','val')}
        flags={}
        for name in ('test','val'):
            for side in (1,2):
                flags[f'{name}_hit{side}']=pc.is_in(t[f'seq{side}_id'],value_set=selected[name]).to_numpy()
        val=flags['val_hit1']|flags['val_hit2']
        test=flags['test_hit1']&flags['test_hit2']
        any_test=flags['test_hit1']|flags['test_hit2']
        exact_val=[pc.is_in(t[f'seq{s}_id'],value_set=pa.array(sorted(refs['val']))).to_numpy() for s in (1,2)]
        exact_test=[pc.is_in(t[f'seq{s}_id'],value_set=pa.array(sorted(refs['test']))).to_numpy() for s in (1,2)]
        summaries[kind]=dict(legacy_rows=len(t),val_either_endpoint=int(val.sum()),test_both_endpoints=int(test.sum()),
            test_either_endpoint=int(any_test.sum()),val_or_test_rule=int((val|test).sum()),
            val_exact_endpoint=int((exact_val[0]|exact_val[1]).sum()),
            test_both_exact_endpoints=int((exact_test[0]&exact_test[1]).sum()))
        for name,values in flags.items():
            t=t.append_column(name,pa.array(values))
        overlaps.append(t.filter(pa.array(val|test)))
    report_table=pa.concat_tables(overlaps)
    path=directory/'inherited_holdout_overlap.index.parquet'
    pq.write_table(report_table,path,compression='zstd')
    warnings=sum(r['val_or_test_rule'] for r in summaries.values())
    if warnings:
        raise ValueError('Original legacy filtering decisions changed')
    duplicates=pq.read_table(directory/'duplicate_map.parquet',
                             columns=['action','structure_source','canonical_source'])
    origins=duplicates.group_by(['action','structure_source','canonical_source']).aggregate([('action','count')])
    origins=origins.sort_by([('action','ascending'),('structure_source','ascending'),('canonical_source','ascending')]).to_pylist()
    report=dict(state='passed_with_inherited_overlap' if warnings else 'passed',
        old_train_count=len(old_ids),old_train_retention_rate=1.0,old_train_missing=0,new_train_rows=new_rows,
        inherited_homology=summaries,homology_hit_sha256=hit_hashes,duplicate_origin_counts=origins,
        inherited_rows_file=path.name,inherited_rows_sha256=file_sha256(path),
        filter_protocol=protocol,
        explanation='Legacy examples and original cached filtering decisions are preserved. Diagnostic k7/direct-alignment results do not change this user-required protocol.',
        evaluation_claim='This verifies the original heuristic protocol, not exhaustive homology isolation. Previously observed diagnostic overlaps remain documented.',
        builder_verification_sha256=file_sha256(directory/'verification.json'),
        audit_script_sha256=file_sha256(__file__))
    write_json(directory/'independent_audit.json',report)
    marker='\n## 独立验收：继承的同源重叠\n'
    readme=directory/'README.md'
    text=readme.read_text().split(marker)[0].replace(
        '旧集合内部重复 ID 的成员全部保留', '旧集合内部监督相同但 ID 不同的成员全部保留')
    lines=[marker,'',
        '独立扫描确认：旧训练样本全部保留，以下按锁定的旧过滤条件及原缓存统计，不使用 k=7/直接比对诊断重新判定旧样本。','',
        '| 旧训练类型 | 命中 val 任意端 | 命中 test 双端 | 两类命中的并集 |',
        '|---|---:|---:|---:|']
    for kind,r in summaries.items():
        lines.append(f"| {kind} | {r['val_either_endpoint']:,} | {r['test_both_endpoints']:,} | {r['val_or_test_rule']:,} |")
    lines+=['',
        '原条件下违规数应为 0；这不否认先前独立诊断发现的相似性，也不证明穷尽比对下没有同源重叠。',
        'inherited_holdout_overlap.index.parquet 保存相关旧样本 ID 与两端命中标记；independent_audit.json 保存独立统计。',
        'test 双端命中指两端分别命中 test 序列集合，不要求它们对应同一个测试 pair。','',
        '完成构建后执行独立验收：','',
        f'    artifacts/environments/esmc_prott5_gated_py312/bin/python tools/audit_expanded_index_split.py --split {directory}',
        '']
    lines+=['### 重复监督的来源','',
        '| 动作 | 记录来源 | 代表来源 | 记录数 |','|---|---|---|---:|']
    for row in origins:
        lines.append(f"| {row['action']} | {row['structure_source']} | {row['canonical_source']} | {row['action_count']:,} |")
    lines+=['','excluded_exact_supervision 仅排除新增副本；retained_legacy_grouped 全部保留在 train 并共享组内权重。','']
    readme.write_text(text+'\n'.join(lines))
    manifest['independent_audit']='independent_audit.json'
    manifest['inherited_homology']=summaries
    manifest['duplicate_origin_counts']=origins
    manifest['audit_status']=report['state']
    manifest['code']['independent_auditor']=str(Path(__file__).resolve())
    manifest['code']['independent_auditor_sha256']=file_sha256(__file__)
    for name in ['README.md','inherited_holdout_overlap.index.parquet','independent_audit.json']:
        p=directory/name
        manifest['artifacts'][name]=dict(sha256=file_sha256(p),bytes=p.stat().st_size)
    write_json(directory/'manifest.json',manifest)
    status_path=directory/'status.json'
    status=json.loads(status_path.read_text())
    status.update(audit_status=report['state'],inherited_homology=summaries)
    write_json(status_path,status)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def audit_strict(directory):
    """Audit from formal artifacts and immutable sources, never build scratch.

    Recompute the integer hit decisions here instead of importing the builder's
    classifier. This intentionally provides an independent boundary/axis check.
    """
    import duckdb
    from piper.data_protocol import strict_command, verify_strict_search_log
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    protocol = verify_strict_inputs(manifest['config'])
    lock = read_strict_lock(protocol['id'])
    rf2 = protocol['id'] == RF2_PROTOCOL_ID
    if manifest['state'] not in ('verifying','complete') or manifest['filter_protocol'] != protocol:
        raise ValueError('Wrong strict split protocol/state')
    if set(manifest['splits']) != {'train','test'}:
        raise ValueError('Strict split must contain exactly train and test, without validation')
    for name, record in manifest['artifacts'].items():
        if file_sha256(directory/name) != record['sha256']:
            raise ValueError('Strict artifact changed: '+name)
    for name, record in manifest['splits'].items():
        path = directory / record['file']
        if file_sha256(path) != record['sha256'] or pq.ParquetFile(path).metadata.num_rows != record['rows']:
            raise ValueError('Strict split index changed: '+name)
    registry_path = directory/'sources.json'
    if file_sha256(registry_path) != manifest['source_registry_sha256']:
        raise ValueError('Strict source registry changed')
    registry = json.loads(registry_path.read_text())
    if set(registry['frozen_splits']) != {'test'}:
        raise ValueError('Unexpected validation reference')
    for source in registry['shards'].values():
        path = Path(registry['roots'][source['root']])/source['path']
        if file_sha256(path) != source['sha256']:
            raise ValueError('Source payload changed: '+str(path))
    reference = pq.read_table(manifest['config']['test'], columns=['seq1','seq2','label'])
    ref_seqs = sorted(set(reference['seq1'].to_pylist()+reference['seq2'].to_pylist()))
    exact = {sequence_id(s) for s in ref_seqs}
    reference_lengths = {f'r{i}':len(s) for i,s in enumerate(ref_seqs)}
    if (directory/'test_reference.fasta').read_text() != ''.join(f'>r{i}\n{s}\n' for i,s in enumerate(ref_seqs)):
        raise ValueError('Reference FASTA omits/changes positive or negative test sequences')
    test = pq.read_table(directory/'test.index.parquet')
    if (len(test) != len(reference) or test['row_index'].to_pylist() != list(range(len(reference)))
            or test['label'].to_pylist() != reference['label'].to_pylist()
            or set(test['shard_id'].to_pylist()) != {registry['frozen_splits']['test']}
            or set(test['kind'].to_pylist()) != {'interaction'}):
        raise ValueError('Frozen test order, labels or row references changed')
    hits_by_kind, summaries = {}, {}
    for kind in ('ppi','ddi'):
        seqs = pq.read_table(directory/(kind+'_sequence_catalog.parquet'))
        lengths = dict(zip(seqs['seq_id'].to_pylist(),seqs['length'].to_pylist()))
        if len(lengths) != len(seqs) or any(n < 1 for n in lengths.values()):
            raise ValueError('Invalid candidate sequence catalog')
        observed_ids, last_id = set(), ''
        with (directory/(kind+'_candidate.fasta')).open() as handle:
            for header in handle:
                sid = header.rstrip('\n')[1:]
                seq = next(handle, '').rstrip('\n')
                if (not header.startswith('>') or sid <= last_id
                        or sid != sequence_id(seq) or lengths.get(sid) != len(seq)):
                    raise ValueError('Candidate FASTA sequence/content/order differs from its catalog')
                observed_ids.add(sid)
                last_id = sid
        if observed_ids != lengths.keys():
            raise ValueError('Candidate FASTA omits catalog sequences')
        # MMseqs skips pairs whose lengths make the locked 50% coverage
        # impossible, even with prefilter-mode=2. All other pairs must align.
        ref_lengths = np.sort(np.fromiter(reference_lengths.values(), dtype=np.int64))
        train_lengths = np.fromiter(lengths.values(), dtype=np.int64)
        lower = np.searchsorted(ref_lengths, (train_lengths+1)//2, side='left')
        upper = (np.searchsorted(ref_lengths, 2*train_lengths, side='right')
                 if kind == 'ppi' else len(ref_lengths))
        expected_full_alignments = len(lengths)*len(ref_lengths) if rf2 else int(np.sum(upper-lower))
        full_alignment_counts = {}
        hits = set() if rf2 else exact.intersection(lengths)
        evidence_count = len(hits)
        expected_evidence_digest = hashlib.sha256()
        qualified_by_direction, empty_by_direction = {}, {}
        for direction in lock['search']['directions']:
            name = f'{kind}_{direction}'
            receipt = json.loads((directory/(name+'.search.json')).read_text())
            inputs = receipt['inputs']
            candidate = Path(manifest['config']['output'])/(kind+'_candidate.fasta')
            ref = Path(manifest['config']['output'])/'test_reference.fasta'
            query,target = (ref,candidate) if direction.startswith('test_to_train') else (candidate,ref)
            count = len(lengths) if direction.startswith('test_to_train') else len(ref_seqs)
            expected_command = strict_command(manifest['config']['mmseqs'],query,target,
                Path(manifest['config']['output'])/'build_scratch'/(name+'.tsv'),
                Path(manifest['config']['output'])/'build_scratch'/(name+'_tmp'), count,kind,direction,protocol['id'])
            if (inputs['command'] != expected_command or inputs['filter_protocol'] != protocol
                    or inputs['query_sha256'] != file_sha256(query) or inputs['target_sha256'] != file_sha256(target)):
                raise ValueError('Recorded search command/input differs from fixed protocol')
            verify_strict_search_log(directory/(name+'_mmseqs.log'), direction,protocol['id'])
            if direction.endswith('_nofilter'):
                counts = re.findall(r'(?m)^(\d+) alignments calculated\s*$',
                                    (directory/(name+'_mmseqs.log')).read_text())
                if len(counts) != 1 or int(counts[0]) != expected_full_alignments:
                    raise ValueError('Full alignment count differs from the complete length-compatible search scope')
                full_alignment_counts[direction] = int(counts[0])
            digest, qualified, empty_alignments = hashlib.sha256(), 0, 0
            with gzip.open(directory/(name+'.tsv.gz'),'rt') as handle:
                for line in handle:
                    digest.update(line.encode())
                    f = line.rstrip('\n').split('\t')
                    if len(f)!=(12 if rf2 else 10):
                        raise ValueError('Malformed archived alignment')
                    ni,aln,qs,qe,ql,ts,te,tl = map(int,f[2:10])
                    sid,rid = (f[1],f[0]) if direction.startswith('test_to_train') else (f[0],f[1])
                    train_span,train_length,test_span,test_length = ((te-ts+1,tl,qe-qs+1,ql)
                        if direction.startswith('test_to_train') else (qe-qs+1,ql,te-ts+1,tl))
                    if lengths.get(sid)!=train_length or reference_lengths.get(rid)!=test_length:
                        raise ValueError('Archived alignment has unknown sequences or wrong lengths')
                    # Independently reject the observed MMseqs no-residue
                    # sentinel; its uninitialized E-value must never be used.
                    if (rf2 and (ni,aln,qs,ts,te)==(0,0,0,0,0)
                            and qe in (0,1) and ql>0 and tl>0):
                        empty_alignments+=1
                        continue
                    if not (aln>0 and 0<=ni<=aln and 1<=qs<=qe<=ql and 1<=ts<=te<=tl):
                        raise ValueError('Invalid archived alignment geometry')
                    # Independently written integer inequalities (no rounded pident/qcov).
                    if rf2:
                        evalue, bits = float(f[10]), float(f[11])
                        if not (math.isfinite(evalue) and evalue>=0 and math.isfinite(bits)):
                            raise ValueError('Invalid archived significance')
                        passed = (ni*100>aln*30 and evalue<0.00001 and
                            (train_span*100>train_length*50 or test_span*100>test_length*50))
                    else:
                        passed = ni*100>aln*30 and train_span*100>=train_length*50
                        if kind=='ppi':
                            passed = passed and test_span*100>=test_length*50
                    if passed:
                        hits.add(sid); qualified+=1
                        if rf2:
                            train_start,train_end,test_start,test_end = ((ts,te,qs,qe)
                                if direction.startswith('test_to_train') else (qs,qe,ts,te))
                            record = [sid,rid,direction,ni,aln,train_start,train_end,train_length,
                                      test_start,test_end,test_length,evalue,bits]
                            expected_evidence_digest.update((json.dumps(record)+'\n').encode())
            if digest.hexdigest()!=receipt['raw_sha256']:
                raise ValueError('Raw alignment checksum mismatch')
            qualified_by_direction[direction]=qualified
            empty_by_direction[direction]=empty_alignments
            evidence_count+=qualified
        hit_table = pq.read_table(directory/(kind+'_homology_hits.parquet'))
        recorded = set(hit_table['seq_id'].to_pylist())
        if hits!=recorded or len(hit_table)!=len(recorded) or not all(hit_table['test_hit'].to_pylist()):
            raise ValueError('Independent alignment reclassification changed hit membership')
        evidence = pq.read_table(directory/(kind+'_homology_evidence.parquet'))
        if len(evidence)!=evidence_count or set(evidence['seq_id'].to_pylist())!=hits:
            raise ValueError('Homology evidence rows are incomplete')
        if rf2:
            observed_digest = hashlib.sha256()
            fields = ['seq_id','reference_id','direction','nident','alnlen','train_start','train_end',
                      'train_length','test_start','test_end','test_length','evalue','bits']
            for row in evidence.select(fields).to_pylist():
                observed_digest.update((json.dumps([row[k] for k in fields])+'\n').encode())
            if observed_digest.hexdigest()!=expected_evidence_digest.hexdigest():
                raise ValueError('RF2-PPI evidence values/order differ from independently classified alignments')
        hits_by_kind[kind]=hits
        summaries[kind]=dict(candidate_sequences=len(lengths),hit_sequences=len(hits),
            exact_matches=len(exact.intersection(lengths)),unconditional_exact_hits=0 if rf2 else len(exact.intersection(lengths)),
            exact_matches_with_qualified_hit=len(exact.intersection(hits)),qualified_alignments=qualified_by_direction,
            empty_alignment_rows=empty_by_direction,
            expected_full_alignments=expected_full_alignments,full_alignment_counts=full_alignment_counts)
    # Relational invariants are checked against formal indexes, not builder DB tables.
    db=duckdb.connect()
    db.execute("SET threads=8")
    db.execute("SET memory_limit='16GB'")
    def q(path):
        return "'"+str(path).replace("'","''")+"'"
    def scalar(query):
        return db.execute(query).fetchone()[0]
    try:
        for table,filename in [('train','train.index.parquet'),('selected','selected_sources.index.parquet'),
                ('excluded','excluded_homology.index.parquet'),('empty_rows','excluded_empty_contacts.index.parquet'),
                ('aliases','duplicate_map.parquet'),('positives','known_positive_cluster_pairs.parquet')]:
            db.execute(f'CREATE TABLE {table} AS SELECT * FROM read_parquet({q(directory/filename)})')
        # Reconstruct selected row locations directly from the immutable source
        # release, so a builder omission cannot disappear from all outcome lists.
        release = Path(manifest['config']['release'])
        release_meta = json.loads((release/'DATASET.json').read_text())
        expected_sources = {str(release/e['file']):e['sha256'] for e in release_meta['parquet_shards']
            if set(e['sources']).intersection(manifest['config']['allowed_sources'][e['kind']])}
        observed_sources = {str(Path(registry['roots'][e['root']])/e['path']):e['sha256']
            for e in registry['shards'].values() if e['root']=='release'}
        if expected_sources != observed_sources:
            raise ValueError('Registry does not cover exactly the selected source release shards')
        paths = [dict(filename=str(Path(registry['roots'][e['root']])/e['path']),source_shard=int(sid))
                 for sid,e in registry['shards'].items() if e['root']=='release']
        db.register('source_paths',pa.Table.from_pylist(paths))
        file_list = '['+','.join(q(p['filename']) for p in paths)+']'
        scope = ' OR '.join('(kind='+q(k)+' AND structure_source IN ('+','.join(q(s) for s in values)+'))'
                            for k,values in manifest['config']['allowed_sources'].items())
        db.execute(f'''CREATE TABLE source_rows AS SELECT sample_id,kind,structure_source,seq1_id,seq2_id,
            source_shard AS shard_id,file_row_number AS row_index,seq1_cluster,seq2_cluster,
            length(seq1) AS len1,length(seq2) AS len2,
            seq1_id='seq_'||sha256(seq1) AS seq1_verified,seq2_id='seq_'||sha256(seq2) AS seq2_verified
            FROM read_parquet({file_list},filename=true,file_row_number=true) JOIN source_paths USING(filename)
            WHERE {scope}''')
        if scalar('''SELECT count(*) FROM source_rows WHERE seq1_verified IS DISTINCT FROM true
                OR seq2_verified IS DISTINCT FROM true OR len1<1 OR len2<1'''):
            raise ValueError('Source sequence IDs do not match the actual payload sequences')
        cols='sample_id,kind,structure_source,seq1_id,seq2_id,shard_id,row_index,seq1_cluster,seq2_cluster,len1,len2'
        if scalar(f'SELECT count(*) FROM (SELECT {cols} FROM source_rows EXCEPT ALL SELECT {cols} FROM selected)') or scalar(
                f'SELECT count(*) FROM (SELECT {cols} FROM selected EXCEPT ALL SELECT {cols} FROM source_rows)'):
            raise ValueError('Candidate index does not exactly match selected source payload rows')
        for kind in ('ppi','ddi'):
            expected = f'''SELECT seq1_id AS seq_id,seq1_cluster AS cluster,len1 AS length
                FROM source_rows WHERE kind={q(kind)} UNION SELECT seq2_id,seq2_cluster,len2
                FROM source_rows WHERE kind={q(kind)}'''
            recorded = f"SELECT seq_id,cluster,length FROM read_parquet({q(directory/(kind+'_sequence_catalog.parquet'))})"
            if scalar(f'SELECT count(*) FROM (({expected}) EXCEPT ALL ({recorded}))') or scalar(
                    f'SELECT count(*) FROM (({recorded}) EXCEPT ALL ({expected}))'):
                raise ValueError('Search candidate catalog does not cover exactly the selected source endpoints')
        hit_rows=[dict(kind=k,seq_id=s) for k,values in hits_by_kind.items() for s in sorted(values)]
        db.register('hits',pa.Table.from_pylist(hit_rows,schema=pa.schema([('kind',pa.string()),('seq_id',pa.string())])))
        # Two equality joins avoid an OR-correlated nested loop over millions
        # of sequence pairs. The boolean endpoint rule is applied afterwards.
        operator = 'AND' if rf2 else 'OR'
        db.execute(f'''CREATE TABLE assessed AS SELECT s.sample_id,
            h1.seq_id IS NOT NULL AS hit1,h2.seq_id IS NOT NULL AS hit2,
            h1.seq_id IS NOT NULL {operator} h2.seq_id IS NOT NULL AS excluded
            FROM selected s LEFT JOIN hits h1 ON s.kind=h1.kind AND s.seq1_id=h1.seq_id
                            LEFT JOIN hits h2 ON s.kind=h2.kind AND s.seq2_id=h2.seq_id''')
        leaks=scalar('SELECT count(*) FROM train JOIN assessed USING(sample_id) WHERE excluded')
        if leaks:
            raise ValueError('Retained training pair violates its locked holdout rule')
        excluded_expected=scalar('SELECT count(*) FROM assessed WHERE excluded')
        invalid_excluded=scalar('SELECT count(*) FROM excluded JOIN assessed USING(sample_id) WHERE NOT assessed.excluded')
        if invalid_excluded or scalar('SELECT count(*) FROM excluded')!=excluded_expected:
            raise ValueError('Holdout exclusion membership is incorrect')
        endpoint_counts = db.execute('''SELECT count(*) FILTER(WHERE hit1 OR hit2),
            count(*) FILTER(WHERE hit1 AND hit2),count(*) FILTER(WHERE hit1!=hit2)
            FROM train JOIN assessed USING(sample_id)''').fetchone()
        db.execute('''CREATE TABLE outcomes AS SELECT sample_id FROM train UNION ALL SELECT sample_id FROM excluded
            UNION ALL SELECT sample_id FROM empty_rows UNION ALL SELECT duplicate_id AS sample_id FROM aliases''')
        if (scalar('SELECT count(*)-count(DISTINCT sample_id) FROM outcomes')
                or scalar('SELECT count(*) FROM selected ANTI JOIN outcomes USING(sample_id)')
                or scalar('SELECT count(*) FROM outcomes ANTI JOIN selected USING(sample_id)')):
            raise ValueError('Source rows lost or assigned to multiple outcomes')
        if scalar('''SELECT count(*) FROM train t JOIN selected s USING(sample_id) WHERE
                t.shard_id!=s.shard_id OR t.row_index!=s.row_index OR t.seq1_id!=s.seq1_id OR t.seq2_id!=s.seq2_id
                OR t.source_seq1_cluster!=s.seq1_cluster OR t.source_seq2_cluster!=s.seq2_cluster
                OR t.source_sequence_cluster_scope!=s.sequence_cluster_scope'''):
            raise ValueError('Index source rows/axes/cluster annotations changed')
        if scalar('''SELECT count(*) FROM (SELECT sequence_pair_id,sum(within_pair_weight) AS weight
                FROM train GROUP BY sequence_pair_id HAVING abs(weight-1)>1e-10)'''):
            raise ValueError('Within-pair weights do not sum to one')
        if scalar('''SELECT count(*) FROM train WHERE within_pair_weight<=0 OR NOT isfinite(within_pair_weight)
                OR seq1_cluster IS NULL OR seq2_cluster IS NULL OR valid_contacts<1'''):
            raise ValueError('Invalid training weights, clusters or labels')
        if scalar('SELECT count(*) FROM aliases a ANTI JOIN train t ON a.canonical_id=t.sample_id'):
            raise ValueError('Dangling duplicate representative')
        if scalar('SELECT count(*)-count(DISTINCT supervision_group_id) FROM train'):
            raise ValueError('Exact supervision duplicates remain in train')
        for kind in ('ppi','ddi'):
            allowed=','.join(q(s) for s in manifest['config']['allowed_sources'][kind])
            if scalar(f'SELECT count(*) FROM train WHERE kind={q(kind)} AND structure_source NOT IN ({allowed})'):
                raise ValueError('Unselected source entered train')
        db.execute('''CREATE TABLE expected_positives AS SELECT DISTINCT least(seq1_cluster,seq2_cluster) AS cluster1,
            greatest(seq1_cluster,seq2_cluster) AS cluster2 FROM train''')
        if scalar('SELECT count(*) FROM positives ANTI JOIN expected_positives USING(cluster1,cluster2)') or scalar(
                'SELECT count(*) FROM expected_positives ANTI JOIN positives USING(cluster1,cluster2)'):
            raise ValueError('Known-positive screen does not exactly match retained training rows')
        legacy=db.execute(f'''SELECT kind,decision,count(*) AS rows FROM read_parquet(
            {q(directory/'legacy_before_after.index.parquet')}) GROUP BY ALL ORDER BY ALL''').fetchall()
        counts=db.execute('SELECT kind,structure_source,count(*) FROM train GROUP BY ALL ORDER BY ALL').fetchall()
        expected_counts={(r['kind'],r['source']):r['rows'] for r in manifest['source_counts'] if r['rows']}
        if {(k,s):n for k,s,n in counts}!=expected_counts:
            raise ValueError('Manifest source counts differ from final train')
        report=dict(state='passed',filter_protocol=protocol,train_rows=scalar('SELECT count(*) FROM train'),
            test_rows=len(test),test_labels={str(label):reference['label'].to_pylist().count(label) for label in (0,1)},
            validation_rows=0,selected_rows=scalar('SELECT count(*) FROM selected'),
            homology_excluded=scalar('SELECT count(*) FROM excluded'),empty_contacts=scalar('SELECT count(*) FROM empty_rows'),
            exact_supervision_duplicates=scalar('SELECT count(*) FROM aliases'),
            retained_rule_violations=leaks,retained_endpoint_hits=endpoint_counts[0],
            retained_both_endpoint_hits=endpoint_counts[1],retained_single_endpoint_hits=endpoint_counts[2],
            candidate_scope_verified=True,raw_alignments_reclassified=True,
            homology=summaries,legacy_before_after=[dict(kind=k,decision=d,rows=n) for k,d,n in legacy],
            source_payload_hashes_verified=len(registry['shards']),
            source_counts=manifest['source_counts'],audit_script_sha256=file_sha256(__file__),training_started=False)
    finally:
        db.close()
    write_json(directory/'independent_audit.json',report)
    manifest.update(state='complete',audit_status='passed',independent_audit='independent_audit.json')
    path=directory/'independent_audit.json'
    manifest['artifacts'][path.name]=dict(sha256=file_sha256(path),bytes=path.stat().st_size)
    write_json(directory/'manifest.json',manifest)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',type=Path,required=True)
    audit(parser.parse_args().split)
