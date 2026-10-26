"""Build the CongLab-only PPI/DDI experiment split without RF2 exclusions.

Original databases are read-only. All scratch FASTA/MMseqs databases are
temporary; final train/validation, audit JSON and fixed benchmark are retained.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
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
from flashppi.ddi_common import sha256, write_json
from audit_conglab_ecoli import homology, matched_sides
from prepare_conglab_mixture import sequence_dictionary, joint_clusters, lookup
from prepare_ppi_finetune_split import MMseqs, Policy


def sequence_pair(row):
    return tuple(sorted((row['seq1'],row['seq2'])))


def cluster_table(table, mmseqs, identity, coverage, name):
    seqs = sequence_dictionary(table)
    names = sorted(seqs)
    mapping = mmseqs.cluster(range(len(names)), [seqs[k] for k in names], identity, coverage, name)
    return {sid:mapping[i] for i,sid in enumerate(names)}


def choose_validation(ppi, ecoli_hits, config, mmseqs):
    """New structural cohort; reuse reliable annotation only for its AP subset."""
    benchmark = pq.read_table(config['mapping_benchmark']).to_pylist()
    ap_pairs = {sequence_pair(r) for r in benchmark if r['label'] == 1}
    a, b = matched_sides(ppi, ecoli_hits)
    valid = ppi.filter(~(a|b))  # Do not put test-homologous chains into validation.
    counts = Counter(valid['seq1_cluster'].to_pylist()+valid['seq2_cluster'].to_pylist())
    pool, anchors = {}, {}
    columns = ['pair_id','seq1','seq2','seq1_id','seq2_id','seq1_cluster','seq2_cluster','fully_valid','len1','len2']
    for row in valid.select(columns).to_pylist():
        if not row['fully_valid'] or min(row['len1'],row['len2']) < 32 or max(row['len1'],row['len2']) > 1024:
            continue
        key = sequence_pair(row)
        priority = (max(counts[row['seq1_cluster']],counts[row['seq2_cluster']]),
                    hashlib.sha256(f"{config['seed']}:{row['pair_id']}".encode()).hexdigest())
        if key in ap_pairs:
            if key not in anchors or priority < anchors[key][0]:
                anchors[key] = (priority,row)
        elif key not in pool or priority < pool[key][0]:
            pool[key] = (priority,row)
    candidates = [x[1] for x in sorted(anchors.values(),key=lambda x:x[0])]
    candidates += [x[1] for x in sorted(pool.values(),key=lambda x:x[0])[:2000]]
    clusters = cluster_table(pa.Table.from_pylist(candidates), mmseqs, .3, .8, 'val_dereplicate')
    selected, seen = [], set()
    for row in candidates:
        key = tuple(sorted((clusters[row['seq1_id']],clusters[row['seq2_id']])))
        if key not in seen:
            selected.append(row['pair_id']); seen.add(key)
        if len(selected) == config['validation_positives']:
            break
    if len(selected) != config['validation_positives']:
        raise ValueError('Insufficient nonredundant validation candidates')
    val = ppi.filter(pc.is_in(ppi['pair_id'],value_set=pa.array(selected)))
    val_pairs = {tuple(sorted(x)) for x in zip(val['seq1'].to_pylist(),val['seq2'].to_pylist())}
    positive = [r for r in benchmark if r['label'] == 1 and sequence_pair(r) in val_pairs]
    pool_sequences = {r[k] for r in positive for k in ('seq1','seq2')}
    negative = [r for r in benchmark if r['label'] == 0 and
                r['seq1'] in pool_sequences and r['seq2'] in pool_sequences]
    if not positive or not negative:
        raise ValueError('No reliable species-matched interaction validation subset')
    return val, pa.Table.from_pylist(positive+negative), dict(
        structural_pairs=len(val), clip_positive=len(positive), clip_negative=len(negative),
        candidate_anchors=len(anchors), ap_coverage=len(positive)/len(val),
        selection='100 CongLab structures; unique 30/80 cluster pairs; annotation anchors then rare clusters',
        ap_annotation='Reuse fixed STRING12 physical<700 negatives among selected positive endpoints only',
        caveat='Small AP subset; unobserved negatives are not experimentally proven non-interactions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['test_exclusion'] != 'both_chains' or config['val_exclusion'] != 'either_chain':
        raise ValueError('This protocol requires both-chain test and either-chain val exclusion')
    output, release = Path(config['output']), Path(config['release'])
    output.mkdir(parents=True,exist_ok=True)
    if (output/'train.parquet').exists():
        raise FileExistsError(output/'train.parquet')
    started = time.monotonic(); hashes = {}; stats = {}; tables = {}
    metadata = json.loads((release/'DATASET.json').read_text())
    audit = json.loads((Path(config['audit'])/'audit.json').read_text())
    if audit['state'] != 'complete' or sha256(config['test']) != audit['test_sha256']:
        raise ValueError('Source audit or frozen test is invalid')
    def stage(name):
        print(name,flush=True)
        write_json(output/'status.json',dict(state='preparing',stage=name,seconds=time.monotonic()-started))
    stage('read_and_verify_conglab_sources')
    for kind in ('ppi','ddi'):
        parts=[]
        for entry in metadata['parquet_shards']:
            if entry['kind'] != kind:
                continue
            path = release/entry['file']; actual = sha256(path)
            if actual != entry['sha256']:
                raise ValueError('Source shard checksum mismatch')
            hashes[str(path)] = actual
            part = pq.read_table(path,filters=[('structure_source','=','conglab')])
            if len(part): parts.append(part)
        raw = pa.concat_tables(parts); del parts
        hit_record = json.loads((Path(config['audit'])/(kind+'_ecoli_hits.json')).read_text())
        hits = set(hit_record['hits'][config[kind+'_test_coverage_rule']])
        a,b = matched_sides(raw,hits)
        keep = ~(a&b)
        stats[kind] = dict(raw=len(raw),test_both_chain_excluded=int((~keep).sum()))
        raw = raw.filter(keep)
        contact_keep = valid_contact_counts(raw)>0
        stats[kind]['no_valid_contacts'] = int((~contact_keep).sum())
        tables[kind] = raw.filter(contact_keep)
        stats[kind]['before_val'] = len(tables[kind])
        print(kind,stats[kind],flush=True)
    with tempfile.TemporaryDirectory(prefix='scratch_',dir=output) as temporary:
        mmseqs = MMseqs(config['mmseqs'],temporary,Policy(threads=config['threads']))
        stage('select_small_validation')
        ep = json.loads((Path(config['audit'])/'ppi_ecoli_hits.json').read_text())
        val,binary,val_stats = choose_validation(tables['ppi'],set(ep['hits'][config['ppi_test_coverage_rule']]),config,mmseqs)
        refs = set(val['seq1'].to_pylist())|set(val['seq2'].to_pylist())
        stage('exclude_validation_homology')
        searches={}
        for kind in ('ppi','ddi'):
            table=tables[kind]
            results,record=homology(sequence_dictionary(table),refs,config,output,kind+'_val')
            hits=results[config[kind+'_val_coverage_rule']]
            a,b=matched_sides(table,hits)
            keep=~(a|b)
            stats[kind]['val_homology_excluded_including_val'] = int((~keep).sum())
            tables[kind]=table.filter(keep)
            stats[kind]['train'] = len(tables[kind])
            searches[kind]={k:v for k,v in record.items() if k != 'hits'}
            print(kind,stats[kind],flush=True)
        stage('cluster_training_ppi_70_70')
        clusters=cluster_table(tables['ppi'],mmseqs,.7,.7,'train_ppi_sampling')
        ppi=tables['ppi']
        for side in (1,2):
            name=f'seq{side}_cluster'
            values=[clusters[sid] for sid in ppi[f'seq{side}_id'].to_pylist()]
            ppi=ppi.set_column(ppi.schema.get_field_index(name),name,pa.array(values,type=pa.int64()))
        tables['ppi']=ppi
        cluster_commands=mmseqs.commands
    stage('join_ddi_clusters_and_write_final_files')
    mapping=pacsv.read_csv(config['ddi_clusters'],parse_options=pacsv.ParseOptions(delimiter='\t'),
                          convert_options=pacsv.ConvertOptions(include_columns=['sequence_id','cluster_id']))
    ids=np.asarray(mapping['sequence_id'].to_numpy(),dtype='S68'); clusters=mapping['cluster_id'].to_numpy()
    ddi=tables['ddi']
    dc1,dc2=(lookup(ddi[f'seq{s}_id'].to_numpy(),ids,clusters) for s in (1,2))
    del mapping,ids,clusters
    values,cluster_stats=joint_clusters(tables['ppi'],ddi,dc1,dc2)
    for kind,left,right in (('ppi',values[0],values[1]),('ddi',values[2],values[3])):
        t=tables[kind]
        for side,x in ((1,left),(2,right)):
            name=f'seq{side}_cluster'
            t=t.set_column(t.schema.get_field_index(name),name,pa.array(x,type=pa.int64()))
        tables[kind]=t
    train=pa.concat_tables([tables['ppi'],tables['ddi']])
    train_sequences=set(train['seq1'].to_pylist())|set(train['seq2'].to_pylist())
    if refs & train_sequences:
        raise ValueError('Exact validation sequence leakage')
    outputs={}
    for filename,table in [('train.parquet',train),('validation.parquet',val),('interaction.parquet',binary)]:
        pq.write_table(table,output/filename,compression='zstd',row_group_size=1024)
        outputs[filename]=dict(sha256=sha256(output/filename),rows=len(table))
    for p in (args.config,Path(config['mapping_benchmark']),Path(config['mapping_benchmark']).parent/'manifest.json',
              Path(config['mapping_benchmark']).parent/'provenance.json',Path(config['ddi_clusters']),
              release/'DATASET.json',Path(config['audit'])/'audit.json',Path(config['test'])):
        hashes[str(p)]=sha256(p)
    manifest=dict(state='complete',config=config,stats=stats,validation=val_stats,outputs=outputs,
        benchmark_sha256=outputs['interaction.parquet']['sha256'],input_hashes=hashes,
        clusters=cluster_stats,cluster_multiplier=max(1000000,cluster_stats['joint_clusters']),
        exact_train_val_sequence_overlap=0,cluster_commands=cluster_commands,validation_searches=searches,
        script_sha256=sha256(__file__),seconds=time.monotonic()-started,
        limitations=['Source original IDs match; processed corpus differs from paper',
          'Both-chain test filtering permits one homologous training endpoint',
          'DDI uses domain coverage to avoid missing domains inside full proteins',
          'No RF2 filtering; this model must not be reported as RF2-test-isolated',
          'Small annotated AP subset; frozen examples and coverage are reported'])
    write_json(output/'manifest.json',manifest)
    write_json(output/'status.json',dict(state='complete',stats=stats,validation=val_stats,seconds=time.monotonic()-started))
    print(json.dumps(dict(stats=stats,validation=val_stats,clusters=cluster_stats)),flush=True)


if __name__ == '__main__':
    main()
