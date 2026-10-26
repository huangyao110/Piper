"""CPU tensor regressions; frozen encoder factories use tiny local test doubles."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flashppi.gated_modules import (ResidueGatedPool, GatedContrastiveHead,
                                   PatchResidualGate, EnsembleResidualFusion)
from flashppi.frozen_prott5 import FrozenProtT5, prott5_token_mapping
from flashppi.gated_warm_start import restore_inherited_parameters, warm_start
from flashppi.model import ContrastiveHead
from flashppi.patch_contact import (PatchSpec, PatchContactHead, patch_coverage_targets,
                                    structural_patch_metrics)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def test_uniform_residue_gate_equals_masked_mean():
    x = torch.randn(2, 9, 16, requires_grad=True)
    mask = torch.arange(9)[None] < torch.tensor([4, 9])[:, None]
    pool = ResidueGatedPool(16)
    expected = x.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[:, None]
    torch.testing.assert_close(pool(x, mask), expected)
    pool(x, mask).sum().backward()
    assert torch.count_nonzero(x.grad[~mask]) == 0


def test_learned_gate_changes_weighted_mean():
    pool = ResidueGatedPool(2, hidden_dim=2)
    with torch.no_grad():
        pool.score[0].weight.copy_(torch.eye(2))
        pool.score[0].bias.zero_()
        pool.score[-1].weight.copy_(torch.tensor([[2., -1.]]))
        pool.score[-1].bias.zero_()
    x = torch.tensor([[[1., 0.], [0., 2.]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    weights = pool.score(pool.norm(x)).sigmoid()
    expected = (weights * x).sum(1) / weights.sum(1)
    torch.testing.assert_close(pool(x, mask), expected)
    assert not torch.allclose(pool(x, mask), x.mean(1))


def test_pool_nan_padding_empty_mask_and_gradient():
    pool = ResidueGatedPool(4)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    x[~mask] = float('nan')
    x.requires_grad_()
    result = pool(x, mask)
    torch.testing.assert_close(result[0], pool(x[:1, :2], mask[:1, :2])[0])
    assert torch.equal(result[1], torch.zeros(4))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[~mask]) == 0
    assert all(torch.isfinite(p.grad).all() for p in pool.parameters())


def test_clip_preserves_head_names_and_uniform_behavior():
    core = ContrastiveHead(8, 4)
    head = GatedContrastiveHead(core, 8)
    assert head.head is core.head
    assert set(core.state_dict()) <= set(head.state_dict())
    x = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    torch.testing.assert_close(head(x, mask), core(x, mask))


def test_patch_gate_identity_and_staged_learning():
    gate = PatchResidualGate(8, 4)
    x = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    x[~mask] = float('nan')
    expected = x.masked_fill(~mask[..., None], 0)
    torch.testing.assert_close(gate(x, mask), expected, atol=0, rtol=0)
    gate(x, mask).square().sum().backward()
    assert gate.beta.grad.abs() > 0
    optimizer = torch.optim.SGD(gate.parameters(), lr=.01)
    optimizer.step()
    optimizer.zero_grad()
    gate(x, mask).square().sum().backward()
    assert gate.projection[-1].weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in gate.parameters())
    assert torch.equal(gate(x, mask)[~mask], torch.zeros(1, 8))


def test_fusion_identity_both_branches_learn_with_nan_padding():
    fusion = EnsembleResidualFusion(output_dim=8)
    primary = torch.randn(2, 3, 8)
    esmc, t5 = torch.randn(2, 3, 960), torch.randn(2, 3, 1024)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    for tensor in (primary, esmc, t5):
        tensor[~mask] = float('nan')
    expected = primary.masked_fill(~mask[..., None], 0)
    torch.testing.assert_close(fusion(primary, esmc, t5, mask), expected, atol=0, rtol=0)
    fusion(primary, esmc, t5, mask).square().sum().backward()
    assert fusion.alpha_esmc.grad.abs() > 0
    assert fusion.alpha_prott5.grad.abs() > 0
    optimizer = torch.optim.SGD(fusion.parameters(), lr=.01)
    optimizer.step()
    optimizer.zero_grad()
    fusion(primary, esmc, t5, mask).square().sum().backward()
    assert fusion.projection_esmc.weight.grad.norm() > 0
    assert fusion.projection_prott5.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in fusion.parameters())


def test_three_residue_grid_keeps_tail_and_coverage_labels():
    spec = PatchSpec(3, 2)
    assert spec.count(512) == 256
    for length in range(1, 20):
        covered = {i for j in range(spec.count(length))
                   for i in range(j * 2, min(length, j * 2 + 3))}
        assert covered == set(range(length))
    target = torch.zeros(1, 5, 5, dtype=torch.bool)
    target[:, 2, 2] = True
    mask = torch.ones(1, 5, dtype=torch.bool)
    fractions, valid = patch_coverage_targets(target, torch.ones_like(target), mask, mask, spec)
    torch.testing.assert_close(fractions, torch.full((1, 2, 2, 2), 1/3))
    assert valid.all()
    row = dict(seq1='AAAAA', seq2='AAAAA', contacts=[[2, 2]])
    logits = torch.ones(2, 2)
    legacy = structural_patch_metrics(logits, row, spec, True, torch.zeros(2, 2, 2))
    current = structural_patch_metrics(logits, row, spec, True, torch.zeros(2, 2, 2),
                                       coverage_small_threshold=1/3 + 1e-7)
    assert legacy['coverage_small_positive_patches'] == 0
    assert current['coverage_small_positive_patches'] == 4
    assert current['coverage_small_threshold'] == 1/3 + 1e-7


def test_patch_head_gates_before_core_without_changing_grid():
    class Core(nn.Module):
        num_heads = 2
        def forward(self, a, b, ma, mb, return_pair_features=False):
            assert a.shape[1] == b.shape[1] == 3
            values = a @ b.transpose(1, 2)
            mask = ma[:, :, None] & mb[:, None, :]
            return (values, mask, values[..., None].expand(-1, -1, -1, 2)) if return_pair_features else (values, mask)
    head = PatchContactHead(Core(), PatchSpec(3, 2), coverage=True, gating={'hidden_dim': 4}, dim=8)
    x = torch.randn(1, 6, 8)
    mask = torch.ones(1, 6, dtype=torch.bool)
    logits, valid, coverage = head.forward_with_coverage(x, x, mask, mask)
    assert logits.shape == valid.shape == (1, 3, 3)
    assert coverage.shape == (1, 3, 3, 2)
    logits.sum().backward()
    assert head.patch_gate.beta.grad.abs() > 0


class PrimaryTokenizer:
    mask_token_id = 7
    def get_vocab(self):
        return {'<pad>': 1, 'A': 3, 'C': 4, 'U': 5, 'Z': 6, '<mask>': 7, 'O': 8, 'B': 9, 'X': 10}


class T5Tokenizer:
    eos_token_id, pad_token_id, unk_token_id = 41, 43, 45
    def encode(self, text, add_special_tokens=False):
        vocab = {'A': 11, 'C': 13, 'X': 17, '<extra_id_0>': 47}
        return [vocab.get(symbol, self.unk_token_id) for symbol in text.split()]


def test_prott5_shifted_tokens_mask_and_noncanonical_residues(monkeypatch, tmp_path):
    import transformers
    class Encoder(nn.Module):
        config = SimpleNamespace(d_model=1024)
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(64, 1024)
        def forward(self, input_ids, attention_mask):
            self.tokens, self.mask = input_ids.clone(), attention_mask.clone()
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    encoder = Encoder()
    def factory(*args, **kwargs):
        assert kwargs['local_files_only'] is True
        return encoder
    monkeypatch.setattr(transformers.T5EncoderModel, 'from_pretrained', factory)
    monkeypatch.setattr(transformers.T5Tokenizer, 'from_pretrained', lambda *a, **k: T5Tokenizer())
    frozen = FrozenProtT5(tmp_path, PrimaryTokenizer())
    frozen.train()
    assert not frozen.training and not frozen.model.training
    ids = torch.tensor([[3, 7, 5, 6, 8, 9], [4, 7, 1, 1, 1, 1]])
    mask = ids != 1
    result = frozen(ids, mask)
    assert result.shape == (2, 6, 1024)
    assert not result.requires_grad
    assert not any(p.requires_grad for p in frozen.parameters())
    assert encoder.tokens.tolist() == [[11, 47, 17, 17, 17, 17, 41], [13, 47, 41, 43, 43, 43, 43]]
    assert encoder.mask.tolist() == [[1]*7, [1, 1, 1, 0, 0, 0, 0]]
    assert torch.equal(result[~mask], torch.zeros(4, 1024))
    torch.testing.assert_close(frozen(ids[1:2, :2], mask[1:2, :2]), result[1:2, :2])
    with pytest.raises(ValueError, match='right-padded'):
        frozen(ids[:, :3], torch.tensor([[1, 0, 1], [1, 1, 0]]))


def test_prott5_rejects_split_residue_tokenizer():
    class Split(T5Tokenizer):
        def encode(self, text, add_special_tokens=False):
            return [11, 13]
    with pytest.raises(ValueError, match='one token'):
        prott5_token_mapping(PrimaryTokenizer(), Split())


def test_frozen_esmc_stays_eval_without_gradients(monkeypatch, tmp_path):
    import esm.models.esmc
    import esm.tokenization
    from flashppi.research_model import FrozenESMC
    class Tokenizer:
        cls_token_id, eos_token_id, pad_token_id, mask_token_id = 0, 2, 1, 6
        def get_vocab(self):
            return {'A': 3, 'C': 4, 'X': 5}
    class Encoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.embed = nn.Embedding(8, 8)
            self.transformer = SimpleNamespace(blocks=[], norm=nn.Identity())
    monkeypatch.setattr(esm.models.esmc, 'ESMC', Encoder)
    monkeypatch.setattr(esm.tokenization, 'get_esmc_model_tokenizers', lambda: Tokenizer())
    torch.save(Encoder().state_dict(), tmp_path / 'weights.pt')
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(d_model=8, n_heads=2, n_layers=0, filename='weights.pt')))
    frozen = FrozenESMC(tmp_path, PrimaryTokenizer())
    assert not frozen.training and not frozen.model.training
    frozen.train()
    assert not frozen.training and not frozen.model.training
    ids, mask = torch.tensor([[3, 7, 1]]), torch.tensor([[1, 1, 0]], dtype=torch.bool)
    result = frozen(ids, mask)
    assert result.shape == (1, 3, 8)
    assert not result.requires_grad and not any(p.requires_grad for p in frozen.parameters())
    assert torch.equal(result[:, 2], torch.zeros(1, 8))


def test_warm_start_requires_every_inherited_tensor():
    model = nn.Module()
    model.head_q = GatedContrastiveHead(ContrastiveHead(8, 4), 8)
    model.fusion = EnsembleResidualFusion(8)
    inherited = {n: torch.randn_like(p) for n, p in model.named_parameters()
                 if n.startswith('head_q.head.')}
    report = restore_inherited_parameters(model, inherited)
    assert report['inherited_parameters'] == sorted(inherited)
    for name, tensor in inherited.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], tensor, atol=0, rtol=0)
    incomplete = dict(inherited)
    incomplete.pop(next(iter(inherited)))
    with pytest.raises(ValueError, match='inherited'):
        restore_inherited_parameters(model, incomplete)
    wrong = dict(inherited)
    wrong[next(iter(inherited))] = torch.zeros(1)
    with pytest.raises(ValueError, match='shape'):
        restore_inherited_parameters(model, wrong)
    model.fusion.unrecognized = nn.Parameter(torch.ones(()))
    with pytest.raises(ValueError, match='inherited'):
        restore_inherited_parameters(model, inherited)


def test_warm_start_checks_source_base_and_lora_hashes(monkeypatch, tmp_path):
    from flashppi.finetune import sha256
    from safetensors.torch import save_file
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(4, 2)
    saved = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    directory = tmp_path / 'run' / 'checkpoints' / 'step_001616'
    directory.mkdir(parents=True)
    code = tmp_path / 'run' / 'code' / 'flashppi'
    code.mkdir(parents=True)
    (code / 'research_model.py').write_text('pinned predecessor')
    base = tmp_path / 'artifacts' / 'glm'
    base.mkdir(parents=True)
    (base / 'weights').write_text('pinned base')
    source = dict(model='artifacts/architecture', mode='lora',
                  initialization={'source': 'original_glm2', 'backbone': 'artifacts/glm'},
                  lora={'last_layers': 16, 'rank': 8, 'alpha': 16, 'dropout': .05})
    config = dict(source, model=str(tmp_path / 'artifacts' / 'architecture'),
                  initialization={'source': 'original_glm2', 'backbone': str(base)})
    (directory / 'config.json').write_text(json.dumps(source))
    save_file(saved, str(directory / 'trainable.safetensors'))
    state = dict(config=source, step=1616,
                 source_hashes={'flashppi/research_model.py': sha256(code / 'research_model.py')},
                 input_hashes={'artifacts/glm/weights': sha256(base / 'weights')})
    torch.save(state, directory / 'training_state.pt')
    manifest = dict(step=1616, trainable_sha256=sha256(directory / 'trainable.safetensors'),
                    training_state_sha256=sha256(directory / 'training_state.pt'))
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    report = warm_start(model, directory, config)
    assert report['inherited_parameters'] == ['bias', 'weight']
    assert report['optimizer'].startswith('fresh')
    with pytest.raises(ValueError, match='LoRA'):
        warm_start(model, directory, dict(config, lora=dict(config['lora'], rank=4)))
    (code / 'research_model.py').write_text('changed source')
    with pytest.raises(ValueError, match='source checksum'):
        warm_start(model, directory, config)
    (code / 'research_model.py').write_text('pinned predecessor')
    (base / 'weights').write_text('changed base')
    with pytest.raises(ValueError, match='input checksum'):
        warm_start(model, directory, config)


def test_verify_inputs_tracks_every_prott5_manifest_file(monkeypatch, tmp_path):
    import flashppi.research_train as trainer
    from flashppi.finetune import sha256
    paths = {}
    for name in ('model', 'base', 'train', 'validation_structures', 'validation_interactions', 'test', 'esmc', 'prott5'):
        paths[name] = tmp_path / name
        if name in ('model', 'base', 'esmc', 'prott5'):
            paths[name].mkdir()
        else:
            paths[name].write_text(name)
    (paths['base'] / 'weights').write_text('base')
    (paths['base'] / 'manifest.json').write_text(json.dumps({'files': {'weights': sha256(paths['base'] / 'weights')}}))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        outputs={name: {'sha256': sha256(paths[name])} for name in ('train', 'validation_structures')},
        benchmark_sha256=sha256(paths['validation_interactions']))))
    (paths['esmc'] / 'weights').write_text('esmc')
    (paths['esmc'] / 'manifest.json').write_text(json.dumps(dict(filename='weights', sha256='323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea')))
    t5 = paths['prott5']
    for name in ('config.json', 'spiece.model', 'pytorch_model.bin'):
        (t5 / name).write_text(name)
    manifest = dict(repository='Rostlab/prot_t5_xl_half_uniref50-enc',
                    revision='94a6abc029ae13029317b140b7424e012bf8dfbf',
                    files={name: sha256(t5 / name) for name in ('config.json', 'spiece.model', 'pytorch_model.bin')})
    (t5 / 'manifest.json').write_text(json.dumps(manifest))
    actual_sha = trainer.sha256
    monkeypatch.setattr(trainer, 'sha256', lambda path: '323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea'
                        if str(path) == str(paths['esmc'] / 'weights') else actual_sha(path))
    monkeypatch.setattr(trainer, 'TEST_SHA', sha256(paths['test']))
    config = {k: str(v) for k, v in paths.items() if k not in ('esmc', 'prott5', 'base')}
    config['initialization'] = {'source': 'original_glm2', 'backbone': str(paths['base'])}
    config['fusion'] = {'source': 'esmc_prott5', 'esmc_model': str(paths['esmc']), 'prott5_model': str(t5)}
    hashes = trainer.verify_inputs(config)
    assert all(str(t5 / name) in hashes for name in manifest['files'])
    (t5 / 'spiece.model').write_text('tampered tokenizer')
    with pytest.raises(ValueError, match='checksum mismatch'):
        trainer.verify_inputs(config)


def test_build_model_and_checkpoint_keep_legacy_and_gated_interfaces(monkeypatch, tmp_path):
    import flashppi.research_model as research
    from flashppi.finetune import sha256
    from flashppi.lora import save_trainable
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 1280)
        def forward(self, ids, mask):
            return self.embedding(ids).masked_fill(~mask.bool()[..., None], 0)
    class Core(nn.Module):
        num_heads = 2
        def __init__(self):
            super().__init__()
            self.transformer = nn.Linear(1280, 4)
        def forward(self, a, b, ma, mb, return_pair_features=False):
            scores = self.transformer(a) @ self.transformer(b).transpose(1, 2)
            valid = ma[:, :, None].bool() & mb[:, None, :].bool()
            return (scores, valid, scores[..., None].expand(-1, -1, -1, 2)) if return_pair_features else (scores, valid)
    def released(_):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            return SimpleNamespace(config=SimpleNamespace(plm_dim=1280, clip_embed_dim=4), plm=Backbone(),
                                   head_q=ContrastiveHead(1280, 4), head_k=ContrastiveHead(1280, 4),
                                   contact_head=Core(), logit_scale=nn.Parameter(torch.ones(()))), PrimaryTokenizer(), {}
    class Frozen(nn.Module):
        dim = 960
        def __init__(self, *args):
            super().__init__()
        def forward(self, ids, mask):
            return ids.float()[..., None].expand(-1, -1, self.dim).masked_fill(~mask.bool()[..., None], 0)
    class T5(Frozen):
        dim = 1024
    monkeypatch.setattr(research, 'load_released', released)
    monkeypatch.setattr(research, 'FrozenESMC', Frozen)
    monkeypatch.setattr(research, 'FrozenProtT5', T5)
    base = dict(mode='head_only', model='unused', seed=5, swap_self_negative=True,
                contact_random_neg_ratio=0., contact_self_neg_ratio=0., contact_hard_neg_ratio=2.,
                contact_patch={'window': 3, 'stride': 2}, contact_coverage={'loss_weight': .1})
    legacy, _, _ = research.build_model(base, torch.empty(0, dtype=torch.long), 'cpu')
    assert legacy.fusion is None and legacy.prott5 is None
    assert legacy.contact_head.patch_gate is None
    config = dict(base, fusion={'source': 'esmc_prott5', 'esmc_model': 'unused', 'prott5_model': 'unused'},
                  clip_pooling={'type': 'residue_gated', 'hidden_dim': 8},
                  patch_gating={'type': 'residual_channel', 'hidden_dim': 8})
    torch.manual_seed(5)
    model, _, _ = research.build_model(config, torch.empty(0, dtype=torch.long), 'cpu')
    ids, mask = torch.tensor([[3, 4, 3, 4, 3]]), torch.ones(1, 5, dtype=torch.bool)
    encoded = model.encode_protein(ids, mask)
    torch.testing.assert_close(encoded, model.plm(ids, mask), atol=0, rtol=0)
    q = model.head_q(encoded, mask)
    logits, valid, coverage = model.contact_head.forward_with_coverage(encoded, encoded, mask, mask)
    assert q.shape == (1, 4) and logits.shape == valid.shape == (1, 2, 2)
    assert coverage.shape == (1, 2, 2, 2)
    config['warm_start_checkpoint'] = 'predecessor must not be loaded by checkpoint loader'
    (tmp_path / 'config.json').write_text(json.dumps(config))
    save_trainable(model, tmp_path / 'trainable.safetensors')
    (tmp_path / 'manifest.json').write_text(json.dumps({'trainable_sha256': sha256(tmp_path / 'trainable.safetensors')}))
    restored, _, loaded_config = research.load_checkpoint(tmp_path, 'cpu')
    assert loaded_config == config
    torch.testing.assert_close(restored.head_q(restored.encode_protein(ids, mask), mask), q)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(dict(restored.named_parameters())[name], parameter, atol=0, rtol=0)


def test_validation_and_final_evaluation_share_configured_coverage_group(monkeypatch, tmp_path):
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import BatchEncoding
    from flashppi.research_eval import evaluate_validation
    import tools.evaluate_research_checkpoint as final_eval
    class Tokenizer:
        def __call__(self, sequences, **kwargs):
            ids = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, sequence in enumerate(sequences):
                ids[i, :len(sequence)] = torch.tensor([ord(c) - 64 for c in sequence])
                mask[i, :len(sequence)] = 1
            return BatchEncoding(dict(input_ids=ids, attention_mask=mask))
    class Head(nn.Module):
        def forward(self, x, mask):
            return x.mean(1)
    class Core(nn.Module):
        num_heads = 2
        def forward(self, a, b, ma, mb, return_pair_features=False):
            scores = a @ b.transpose(1, 2)
            valid = ma[:, :, None].bool() & mb[:, None, :].bool()
            features = scores[..., None].expand(-1, -1, -1, 2)
            return (scores, valid, features) if return_pair_features else (scores, valid)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.head_q, self.head_k = Head(), Head()
            self.contact_head = PatchContactHead(Core(), PatchSpec(3, 2), coverage=True)
        def encode_protein(self, ids, mask):
            return ids.float()[..., None] * self.weight
    binary = [dict(seq1='ACACA', seq2='ACACA', label=1, contacts=[[2, 2]]),
              dict(seq1='AAAAA', seq2='AAAAA', label=0, contacts=[])]
    structure = [binary[0]]
    for name, rows in (('interactions', binary), ('structures', structure), ('test', binary)):
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / (name + '.parquet'))
    config = dict(validation_interactions=str(tmp_path / 'interactions.parquet'),
                  validation_structures=str(tmp_path / 'structures.parquet'), test=str(tmp_path / 'test.parquet'),
                  contact_patch={'window': 3, 'stride': 2}, patch_evaluation={'window': 3, 'stride': 2},
                  contact_coverage={'loss_weight': .1}, coverage_evaluation={'small_threshold': 1/3 + 1e-7})
    model, tokenizer = Model(), Tokenizer()
    metrics = evaluate_validation(model, tokenizer, config, tmp_path / 'validation')
    assert metrics['coverage_small_positive_patches'] == 4
    assert metrics['coverage_small_threshold'] == config['coverage_evaluation']['small_threshold']
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps(config))
    (checkpoint / 'trainable.safetensors').write_text('test checkpoint metadata')
    monkeypatch.setattr(final_eval, 'verify_inputs', lambda c: {})
    monkeypatch.setattr(final_eval, 'load_checkpoint', lambda *a: (model, tokenizer, config))
    monkeypatch.setattr(sys, 'argv', ['evaluate_research_checkpoint', '--checkpoint', str(checkpoint),
                                    '--output', str(tmp_path / 'final'), '--device', 'cpu'])
    final_eval.main()
    final_metrics = json.loads((tmp_path / 'final' / 'metrics.json').read_text())
    assert final_metrics['coverage_small_positive_patches'] == 4
    assert final_metrics['coverage_small_threshold'] == metrics['coverage_small_threshold']
