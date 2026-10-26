"""Sampling probabilities, immutable payloads and explicit phase transitions."""
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from piper.data_protocol import PROTOCOL_ID, LOCK_SHA256
from piper.indexed_data import IndexedSplit, file_sha256
from piper.indexed_training import (IndexedKindSampler, IndexedTrainingPopulation,
                                       POLICY, validate_continuation, prune_checkpoints)


def sampler(batch_size=20, seed=17):
    # PPI: duplicate supervision rows share weight; same-cluster pairs count once.
    # DDI: cluster 7 has three records, cluster 8 one; both get half the DDI mass.
    return IndexedKindSampler(
        ['ppi'] * 4 + ['ddi'] * 4, [.25, .25, .5, 1, .25, .25, .5, 1], batch_size, seed,
        [np.nan] * 4 + [7, 7, 7, 8], [None] * 4 + ['ddi_teddymer_extension'] * 4,
        [0, 0, 0, 1, 10, 11, 12, 13], [1, 1, 1, 1, 20, 21, 22, 23])


def test_exact_marginal_cluster_probabilities_and_positive_old_member_weights():
    s = sampler()
    # PPI chooses cluster 0/1 equally. Cluster 0 sees rows 0..2,
    # cluster 1 also sees row 3, so PPI probabilities are 3/16,3/16,3/8,1/4.
    np.testing.assert_allclose(s.effective_weights[:4] / s.effective_weights[:4].sum(),
                               [3/16, 3/16, 3/8, 1/4], atol=1e-15)
    np.testing.assert_allclose(s.effective_weights[4:] / s.effective_weights[4:].sum(),
                               [1/8, 1/8, 1/4, 1/2], atol=1e-15)
    assert (s.effective_weights > 0).all()
    assert s.ppi_sequence_clusters == s.ddi_structure_clusters == 2


def test_sampling_balance_frequency_and_resume_prefix():
    s = sampler()
    batches = np.array([s.batch(i) for i in range(2000)])
    assert ((batches < 4).sum(1) == 10).all()
    counts = np.bincount(batches.reshape(-1), minlength=8)
    np.testing.assert_allclose(counts[:4] / counts[:4].sum(), [3/16,3/16,3/8,1/4], atol=.012)
    np.testing.assert_allclose(counts[4:] / counts[4:].sum(), [1/8,1/8,1/4,1/2], atol=.012)
    s.num_samples = 100 * s.batch_size
    prefix = list(s)
    s.num_samples = 200 * s.batch_size
    assert list(s)[:len(prefix)] == prefix
    np.testing.assert_array_equal(sampler().batch(87), batches[87])


def test_missing_ddi_cluster_fails_instead_of_falling_back_to_uniform_rows():
    with pytest.raises(ValueError, match='structure cluster'):
        IndexedKindSampler(['ppi','ddi'], [1,1], 2, 7, [np.nan,np.nan],
                           [None,'ddi_teddymer_extension'], [1,2], [3,4])
    with pytest.raises(ValueError, match='namespace'):
        IndexedKindSampler(['ppi','ddi'], [1,1], 2, 7, [np.nan,7], [None,None], [1,2], [3,4])


def population_fixture(tmp_path):
    rows = [dict(pair_id=f'r{i}', seq1='ACDE', seq2='GHIKL', contacts=[[0,1],[2,3]],
                 valid1=[True,False,True,True], valid2=[True,True,False,True,True]) for i in range(4)]
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=2)
    annotations = [dict(sample_id=f'r{i}', shard_id=0, row_index=i, kind='ppi' if i<2 else 'ddi',
                        structure_source='conglab' if i%2 else 'teddymer', legacy_train=bool(i%2),
                        seq1_cluster=10+i, seq2_cluster=20+i, within_pair_weight=1.,
                        structure_cluster_id=None if i<2 else 7,
                        structure_cluster_scope=None if i<2 else 'ddi_teddymer_extension') for i in range(4)]
    index = tmp_path / 'train.index.parquet'
    pq.write_table(pa.Table.from_pylist(annotations), index, row_group_size=2)
    registry = tmp_path / 'sources.json'
    registry.write_text(json.dumps(dict(roots={'release':str(tmp_path)}, shards={'0':dict(
        root='release',path=source.name,sha256=file_sha256(source),row_group_starts=[0,2])})))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(state='complete',
        filter_protocol=dict(id=PROTOCOL_ID,lock_sha256=LOCK_SHA256),
        source_registry_sha256=file_sha256(registry),splits={'train':dict(
            file=index.name,rows=4,sha256=file_sha256(index))})))
    return IndexedTrainingPopulation(tmp_path), source


def test_grouped_reads_preserve_axes_masks_duplicates_and_sample_order(tmp_path):
    population, source = population_fixture(tmp_path)
    reference = IndexedSplit(tmp_path)
    selection = [3,0,2,3,1]
    actual = population.take(selection).to_pylist()
    for i, row in zip(selection, actual):
        expected = reference[i]
        for field in ('pair_id','seq1','seq2','contacts','valid1','valid2','seq1_cluster','seq2_cluster'):
            assert row[field] == expected[field]
    assert actual[0] == actual[3] and actual[0] is not actual[3]
    assert population.take([]).to_pylist() == []
    with pytest.raises(IndexError): population.take([4])
    # Source replacement must fail even after the first successful hash check.
    source.touch()
    with pytest.raises(ValueError, match='changed during training'): population.take([0])


def test_source_hash_checked_before_first_payload_read(tmp_path):
    population, source = population_fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'): population.take([0])


def test_expanded_mask_keys_use_int64_without_cluster_hash_overflow(monkeypatch):
    from piper import train as research_train
    monkeypatch.setattr(research_train, 'IndexedTrainingPopulation', lambda _: [0,1])
    pairs = pa.table({'cluster1':pa.array([13_000_000, 1], type=pa.int32()),
                      'cluster2':pa.array([13_000_001, 2], type=pa.int32())})
    monkeypatch.setattr(research_train.pq, 'read_table', lambda _: pairs)
    multiplier = 13_593_056
    _, keys, _ = research_train.load_population(dict(indexed_split='unused',eligible_train_rows=2,
                                                    cluster_multiplier=multiplier))
    assert set(keys.tolist()) == {13_000_000*multiplier+13_000_001,
                                  13_000_001*multiplier+13_000_000, multiplier+2, 2*multiplier+1}


def test_continuation_changes_data_only_and_preserves_scheduler_age():
    old = dict(train='old',steps=1616,epochs=2,learning_rate_schedule='constant',
               learning_rate=1e-4,backbone_learning_rate=3e-5,mode='full_last_layers')
    new = dict(old,train='new',indexed_split='new',indexed_sampling=POLICY,
               phase_start_step=1616,steps=2616,epochs=None)
    assert 'train' in validate_continuation(old,new,1616)
    for field,value in [('backbone_learning_rate',1e-5),('mode','lora'),('phase_start_step',0)]:
        changed = dict(new,**{field:value})
        with pytest.raises(ValueError): validate_continuation(old,changed,1616)


def test_retention_preserves_initial_best_two_recovery_and_foreign_checkpoint(tmp_path):
    output = tmp_path/'run'
    for step in (1616,1666,1716,1766,1816,1866):
        p=output/'checkpoints'/f'step_{step:06d}';p.mkdir(parents=True)
        (p/'manifest.json').write_text(json.dumps(dict(step=step)))
    foreign=tmp_path/'previous/checkpoints/step_001400';foreign.mkdir(parents=True)
    best={'contact':dict(checkpoint='checkpoints/step_001666'),
          'clip':dict(checkpoint='../previous/checkpoints/step_001400')}
    assert prune_checkpoints(output,best,keep_last=2,keep_steps=[1616]) == ['step_001716','step_001766']
    assert foreign.exists()
    assert sorted(p.name for p in (output/'checkpoints').iterdir()) == [
        'step_001616','step_001666','step_001816','step_001866']
