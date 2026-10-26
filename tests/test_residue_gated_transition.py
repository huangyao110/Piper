"""Residue resolution, unchanged gates, strict checkpoint transfer, and masks."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from flashppi.gated_modules import ResidueGatedContactHead, GatedContrastiveHead, EnsembleResidualFusion
from flashppi.model import ContactHead, ContrastiveHead
from flashppi.patch_contact import PatchContactHead, PatchSpec
from flashppi.residue_warm_start import restore_residue_parameters, validate_transition_config, DROPPED


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(29)


def model(patch):
    m = nn.Module()
    m.contact_head = (PatchContactHead(ContactHead(16, 16, 2), PatchSpec(3, 2), coverage=True,
                                     gating={'hidden_dim': 4}, dim=16) if patch else
                      ResidueGatedContactHead(ContactHead(16, 16, 2), 16, 4))
    m.head_q = GatedContrastiveHead(ContrastiveHead(16, 8), 16, 4)
    m.head_k = GatedContrastiveHead(ContrastiveHead(16, 8), 16, 4)
    m.fusion = EnsembleResidualFusion(16)
    m.backbone = nn.Linear(16, 16).requires_grad_(False)
    m.lora = nn.Linear(16, 4)
    return m


def test_zero_gate_is_exact_original_residue_head():
    core = ContactHead(16, 16, 2).eval()
    head = ResidueGatedContactHead(copy.deepcopy(core), 16, 4).eval()
    a, b = torch.randn(2, 7, 16), torch.randn(2, 9, 16)
    ma = torch.arange(7)[None] < torch.tensor([4, 7])[:, None]
    mb = torch.arange(9)[None] < torch.tensor([6, 9])[:, None]
    original, valid = core(a, b, ma, mb)
    actual, mask = head(a, b, ma, mb)
    assert actual.shape == (2, 7, 9)
    assert torch.equal(mask, valid)
    torch.testing.assert_close(actual[mask], original[valid], rtol=0, atol=0)


def test_gate_preserved_with_residue_padding_and_gradients():
    source, target = model(True), model(False)
    with torch.no_grad():
        source.contact_head.patch_gate.beta.fill_(.1)
    report = restore_residue_parameters(target, {n: p.detach().clone() for n, p in source.named_parameters()
                                               if p.requires_grad})
    assert report['all_copied_exact'] and not report['new_parameters']
    a, b = torch.randn(1, 5, 16), torch.randn(1, 8, 16)
    ma, mb = torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)
    torch.testing.assert_close(target.contact_head.residue_gate(a, ma),
                               source.contact_head.patch_gate(a, ma), rtol=0, atol=0)
    torch.testing.assert_close(target.head_q(a, ma), source.head_q(a, ma), rtol=0, atol=0)
    clean, _ = target.contact_head(a, b, ma, mb)
    padded = torch.cat([a, torch.full((1, 4, 16), float('nan'))], dim=1).requires_grad_()
    pma = torch.arange(9)[None] < 5
    out, valid = target.contact_head(padded, b, pma, mb)
    torch.testing.assert_close(out[:, :5], clean, rtol=2e-5, atol=2e-6)
    assert not valid[:, 5:].any()
    out[valid].square().mean().backward()
    assert torch.isfinite(padded.grad).all()
    assert not padded.grad[:, 5:].count_nonzero()
    for p in target.contact_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert target.contact_head.residue_gate.beta.grad.abs() > 0


@pytest.mark.parametrize('fault', ['missing_gate', 'extra_tensor', 'wrong_shape', 'wrong_dtype', 'nan'])
def test_transition_rejects_incompatible_weights_before_any_copy(fault):
    source, target = model(True), model(False)
    saved = {n: p.detach().clone() for n, p in source.named_parameters() if p.requires_grad}
    key = 'contact_head.patch_gate.beta'
    if fault == 'missing_gate': saved.pop(key)
    if fault == 'extra_tensor': saved['unrecognized'] = torch.ones(1)
    if fault == 'wrong_shape': saved[key] = torch.zeros(3)
    if fault == 'wrong_dtype': saved[key] = saved[key].double()
    if fault == 'nan': saved[key].fill_(float('nan'))
    before = {n: p.detach().clone() for n, p in target.named_parameters()}
    with pytest.raises(ValueError):
        restore_residue_parameters(target, saved)
    for n, p in target.named_parameters():
        assert torch.equal(p, before[n])


def test_pinned_transition_configs_keep_gates_and_change_only_contact_resolution():
    root = Path(__file__).resolve().parent / 'fixtures'
    source = json.loads((root / 'gated_transition_source.json').read_text())
    target = json.loads((root / 'gated_transition_target.json').read_text())
    validate_transition_config(source, target)
    for key in ['batch_size', 'chunk_size', 'max_len', 'contact_negative_sampling', 'sequence_masking',
                'lora', 'clip_loss_weight', 'contact_loss_weight', 'crop']:
        assert source[key] == target[key]
    for key in ['fusion', 'clip_pooling', 'residue_contact_gating']:
        altered = dict(target, **{key: {}})
        with pytest.raises(ValueError): validate_transition_config(source, altered)
    with pytest.raises(ValueError):
        validate_transition_config(source, dict(target, contact_patch={'window': 1, 'stride': 1}))
    assert DROPPED == {'contact_head.coverage_mix.weight', 'contact_head.coverage_mix.bias'}


def test_collator_does_not_expand_single_residue_label():
    from flashppi.dataset import DataCollatorForPPI
    from flashppi.sequence_masking import SequenceMaskingCollator
    from test_sequence_masking import Tokenizer
    row = dict(seq1='ACDEF', seq2='ACDEFGH', contacts=[(2, 3)],
               valid1=[True, True, True, True, False], valid2=[True]*7,
               seq1_cluster=1, seq2_cluster=2)
    collator = SequenceMaskingCollator(DataCollatorForPPI(Tokenizer(), 8, False, False), .15)
    batch = collator([row])
    target = batch['contact_target']
    assert target.shape == (1, 8, 8)
    assert target.sum() == 1 and target[0, 2, 3]
    assert not batch['contact_target_valid_mask'][0, 4:].any()
    assert not any(k.startswith('coverage') for k in batch)
