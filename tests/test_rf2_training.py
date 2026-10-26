"""RF2 v3 source quotas, fixed-budget scheduling and test-only monitoring."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

from piper.data_protocol import RF2_PROTOCOL_ID, RF2_LOCK_SHA256
from piper.indexed_training import (IndexedSequenceQuotaSampler, RF2_SAMPLING_POLICY,
                                       RF2_SOURCE_QUOTA, verify_indexed_inputs)
from piper.train import (evaluation_due, set_step_learning_rate,
                                    test_monitoring_context as monitoring_context)


def quota_sampler(**overrides):
    settings = dict(kinds=['ddi'] * 4 + ['ppi'] * 8,
                    sources=['conglab'] * 8 + ['afdb_complexs_pred', 'conglab_pred'] * 2,
                    weights=[.25, .25, .5, 1.] * 3, batch_size=40, seed=29,
                    seq1_clusters=[0, 0, 0, 1] * 3,
                    seq2_clusters=[1, 1, 1, 1] * 3,
                    source_quota=RF2_SOURCE_QUOTA)
    settings.update(overrides)
    return IndexedSequenceQuotaSampler(**settings)


def test_source_quota_has_exact_conditional_cluster_probabilities():
    sampler = quota_sampler()
    # Enumerated two-stage probability: cluster 0 sees the first three rows;
    # cluster 1 sees all four. Duplicate supervision weights sum to one.
    for group, fraction in enumerate((.5, .25, .25)):
        np.testing.assert_allclose(sampler.effective_weights[group * 4:(group + 1) * 4],
                                   fraction * np.array([3/16, 3/16, 3/8, 1/4]), atol=1e-15)
    assert sum(sampler.effective_weights) == pytest.approx(1.)
    assert sampler.ppi_sequence_clusters == sampler.ddi_sequence_clusters == 2
    assert sampler.group_cluster_counts == dict.fromkeys(RF2_SOURCE_QUOTA, 2)


def test_source_counts_frequency_and_resume_prefix_are_stable():
    sampler = quota_sampler()
    batches = np.array([sampler.batch(i) for i in range(1500)])
    assert ((batches < 4).sum(1) == 20).all()
    assert (((batches >= 4) & (batches < 8)).sum(1) == 10).all()
    assert ((batches >= 8).sum(1) == 10).all()
    counts = np.bincount(batches.ravel(), minlength=12) / batches.size
    np.testing.assert_allclose(counts, sampler.effective_weights, atol=.006)
    sampler.num_samples = 40 * 10
    first = list(sampler)
    sampler.num_samples *= 2
    assert list(sampler)[:len(first)] == first
    reordered = quota_sampler(source_quota=dict(reversed(list(RF2_SOURCE_QUOTA.items()))))
    np.testing.assert_array_equal(reordered.batch(91), batches[91])


@pytest.mark.parametrize('change', [
    dict(batch_size=18), dict(weights=[0.] * 12), dict(seq1_clusters=[np.nan] * 12),
    dict(sources=['pinder'] * 12), dict(source_quota={'ddi/conglab': 1.}),
    dict(sources=['conglab'] * 12),
])
def test_invalid_quota_or_annotations_fail_closed(change):
    with pytest.raises(ValueError):
        quota_sampler(**change)


def make_v3(tmp_path):
    for name in ('verification.json', 'independent_audit.json'):
        (tmp_path / name).write_text(json.dumps({'state': 'passed'}))
    artifacts = {name: {'sha256': 'fixture'} for name in (
        'known_positive_cluster_pairs.parquet', 'verification.json', 'independent_audit.json')}
    artifacts['filter_protocol.lock.json'] = {'sha256': RF2_LOCK_SHA256}
    manifest = dict(state='complete', audit_status='passed',
                    filter_protocol=dict(id=RF2_PROTOCOL_ID, lock_sha256=RF2_LOCK_SHA256),
                    splits={'train': dict(file='train.index.parquet', rows=100, sha256='train'),
                            'test': dict(file='test.index.parquet', rows=20, sha256='test')},
                    cluster_multiplier=1000, source_registry_sha256='registry', artifacts=artifacts)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return dict(indexed_split=str(tmp_path), train=str(tmp_path / 'train.index.parquet'),
                test='fixed_test.parquet', indexed_sampling=RF2_SAMPLING_POLICY,
                indexed_source_quota=RF2_SOURCE_QUOTA, balanced_ppi_ddi=True,
                eligible_train_rows=100, cluster_multiplier=1000, batch_size=40,
                epochs=4, steps=12, phase_start_step=0, evaluation_split='test')


def test_v3_verification_needs_no_val_and_checks_test_index(tmp_path):
    config = make_v3(tmp_path)
    hashes = verify_indexed_inputs(config)
    assert hashes[str(tmp_path / 'test.index.parquet')] == 'test'
    assert hashes[str(tmp_path / 'filter_protocol.lock.json')] == RF2_LOCK_SHA256
    assert hashes['fixed_test.parquet'].startswith('07dace691cac')
    assert not any('validation' in path for path in hashes)


@pytest.mark.parametrize('change', [dict(validation_structures='old_val'),
    dict(evaluation_split='validation'), dict(steps=13), dict(phase_start_step=1),
    dict(eligible_train_rows=99), dict(cluster_multiplier=1001), dict(indexed_sampling='old'),
    dict(indexed_source_quota={}), dict(epochs=None)])
def test_v3_rejects_wrong_budget_val_policy_and_namespace(tmp_path, change):
    config = make_v3(tmp_path)
    config.update(change)
    with pytest.raises(ValueError):
        verify_indexed_inputs(config)


def test_cosine_warmup_floor_and_resume_use_global_step():
    p, q = torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([dict(params=[p], lr=1e-5, base_lr=1e-5),
                                   dict(params=[q], lr=1e-4, base_lr=1e-4)])
    config = dict(steps=4216, warmup_steps=200, learning_rate=1e-4,
                   learning_rate_schedule='cosine', minimum_lr_ratio=.1)
    for step, factor in ((0, 1/200), (199, 1), (4215, .1)):
        set_step_learning_rate(optimizer, config, step)
        assert [g['lr'] for g in optimizer.param_groups] == pytest.approx([1e-5*factor, 1e-4*factor])
    set_step_learning_rate(optimizer, config, 2000)
    resumed = [g['lr'] for g in optimizer.param_groups]
    for step in range(2001):
        set_step_learning_rate(optimizer, config, step)
    assert resumed == [g['lr'] for g in optimizer.param_groups]
    config.update(learning_rate_schedule='constant')
    set_step_learning_rate(optimizer, config, 2000)
    assert [g['lr'] for g in optimizer.param_groups] == [1e-5, 1e-4]


def test_test_cadence_is_200_and_final_not_extra_epoch_boundaries():
    config = dict(evaluation_split='test', eval_interval=200, steps=4216)
    actual = [s for s in range(4217) if evaluation_due(config, s, s % 1054 == 0)]
    assert actual == list(range(200, 4201, 200)) + [4216]
    assert not evaluation_due(dict(config, smoke=True), 200)


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_monitor_restores_rng_adam_and_training_mode(device, fail):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA offload/restore check requires GPU')
    model = torch.nn.Linear(4, 2).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(2, 4, device=device)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = copy.deepcopy(optimizer.state_dict())
    weights = copy.deepcopy(model.state_dict())
    py_rng, np_rng, th_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state() if device == 'cuda' else None
    try:
        with monitoring_context(model, optimizer):
            assert all(v.device.type == 'cpu' for s in optimizer.state.values()
                       for v in s.values() if isinstance(v, torch.Tensor))
            model.eval()
            random.random(); np.random.random(); torch.rand(4, device=device)
            if fail:
                raise RuntimeError('test interrupted')
    except RuntimeError as error:
        assert fail and str(error) == 'test interrupted'
    assert model.training and random.getstate() == py_rng
    assert np.random.get_state()[2:] == np_rng[2:]
    np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
    assert torch.equal(torch.get_rng_state(), th_rng)
    if gpu_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
    for key, value in weights.items():
        assert torch.equal(model.state_dict()[key], value)
    current = optimizer.state_dict()
    assert current['param_groups'] == state['param_groups']
    for index, values in state['state'].items():
        for key, value in values.items():
            assert current['state'][index][key].device == value.device
            assert torch.equal(current['state'][index][key], value)


def test_test_metrics_keep_full_rows_and_historical_scoring(tmp_path, monkeypatch):
    from piper import evaluation as research_eval
    rows = [dict(seq1='A' * 1200, seq2='C', label=label) for label in (1, 1, 0, 0)]
    results = [dict(label=label, contact_score=score, clip_score=score,
                    contacts_precision_at_k=pk, contacts_8A_precision_at_k=p8,
                    patch_average_precision=ap, patch_precision_at_k=pk)
               for label, score, pk, p8, ap in (
                   (1, .9, .1, .05, .2), (1, .4, .3, None, .4),
                   (0, .8, None, None, None), (0, .2, None, None, None))]
    def read(path):
        assert path == 'fixed_test'
        return pa.Table.from_pylist(rows)
    def score(model, tokenizer, supplied, output, **kwargs):
        assert supplied == rows
        assert kwargs['clip_row_count'] == 4
        assert kwargs['patch_on_labeled_positives'] is True
        assert kwargs['pair_batch'] == 16
        assert kwargs['patch_spec'].window == 3
        Path(output).mkdir()
        return results
    monkeypatch.setattr(research_eval.pq, 'read_table', read)
    monkeypatch.setattr(research_eval, 'score_rows', score)
    metrics = research_eval.evaluate_test(None, None, dict(test='fixed_test',
        patch_evaluation=dict(window=3, stride=2)), tmp_path / 'test')
    assert metrics['rows'] == 4 and metrics['positive'] == 2
    assert metrics['contact_score_ap'] == pytest.approx(5/6)
    assert metrics['contact_score_auroc'] == .75
    assert metrics['contact']['tp'] == metrics['contact']['fp'] == 1
    assert metrics['contacts']['mean_precision_at_k'] == pytest.approx(.2)
    assert metrics['contacts_8A']['mean_precision_at_k'] == .05
    assert metrics['patch']['average_precision'] == pytest.approx(.3)


def test_fresh_run_config_has_user_budget_and_unchanged_gate_design():
    path = Path(__file__).resolve().parents[1] / 'config/train/conglab_distilled_rf2ppi_v3_4epochs.json'
    config = json.loads(path.read_text())
    assert config['epochs'] == 4 and config['steps'] == 4216
    assert config['initialization']['source'] == 'original_glm2'
    assert not config.get('warm_start_checkpoint') and not config.get('lora')
    assert config['full_finetune'] == dict(last_layers=16)
    assert config['clip_pooling'] == dict(type='residue_gated', hidden_dim=64)
    assert config['residue_contact_gating'] == dict(type='residual_channel', hidden_dim=128)
    assert config['indexed_source_quota'] == RF2_SOURCE_QUOTA
    assert config['eval_interval'] == 200 and config['evaluation_split'] == 'test'
