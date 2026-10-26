"""Directional coverage semantics, compatibility, and masked gradient checks."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from piper.model import ContactHead
from piper.patch_contact import (PatchSpec, PatchContactHead, patch_coverage_targets,
                                    coverage_loss_sum, structural_patch_metrics)

torch.set_num_threads(1)


def test_counts_residues_not_edges_and_swaps_direction():
    target = torch.zeros(1, 16, 16, dtype=torch.bool)
    target[:, :2, :12] = True  # 24 edges, two A residues, twelve B residues.
    observed = torch.ones_like(target)
    mask = torch.ones(1, 16, dtype=torch.bool)
    labels, valid = patch_coverage_targets(target, observed, mask, mask, PatchSpec())
    torch.testing.assert_close(labels[0, 0, 0], torch.tensor([2/16, 12/16]))
    assert valid.all()
    reversed_labels, _ = patch_coverage_targets(target.transpose(1, 2), observed.transpose(1, 2), mask, mask, PatchSpec())
    torch.testing.assert_close(reversed_labels, labels.transpose(1, 2).flip(-1))


def test_tail_padding_and_unknown_never_change_denominators():
    target = torch.zeros(1, 24, 24, dtype=torch.bool)
    target[:, 16, 6] = True
    mask_a = torch.arange(24)[None] < 17
    mask_b = torch.arange(24)[None] < 7
    observed = mask_a[:, :, None] & mask_b[:, None, :]
    labels, valid = patch_coverage_targets(target, observed, mask_a, mask_b, PatchSpec())
    assert valid.sum() == 1
    torch.testing.assert_close(labels[0, 1, 0], torch.tensor([1/13, 1/7]))
    observed[:, 5, 2] = False  # Missing position inside the positive tail window.
    _, valid = patch_coverage_targets(target, observed, mask_a, mask_b, PatchSpec())
    assert not valid.any()


def test_zero_contact_is_not_conditional_coverage_supervision():
    target = torch.zeros(2, 16, 16, dtype=torch.bool)
    mask = torch.ones(2, 16, dtype=torch.bool)
    labels, valid = patch_coverage_targets(target, torch.ones_like(target), mask, mask, PatchSpec())
    assert not labels.any() and not valid.any()


def test_loss_excludes_unknown_and_mined_negatives_and_weights_structures():
    logits = torch.randn(5, 2, 2, 2, requires_grad=True)
    target = torch.rand(3, 2, 2, 2)
    mask = torch.zeros(3, 2, 2, dtype=torch.bool)
    mask[0, 0, 0] = True; mask[1] = True
    actual = coverage_loss_sum(logits, target, mask) / 2
    expected = (F.binary_cross_entropy_with_logits(logits[0, 0, 0], target[0, 0, 0])
                + F.binary_cross_entropy_with_logits(logits[1], target[1])) / 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert not logits.grad[2:].any()
    assert not logits.grad[0, 1].any()
    empty = coverage_loss_sum(logits, target, torch.zeros_like(mask))
    assert empty == 0 and empty.requires_grad


def test_loss_is_invariant_to_microbatch_partition():
    logits = torch.randn(5, 2, 2, 2, requires_grad=True)
    targets = torch.rand_like(logits)
    mask = torch.rand(5, 2, 2) > 0.5; mask[-1] = False
    full = coverage_loss_sum(logits, targets, mask)
    split = sum(coverage_loss_sum(logits[i:i+2], targets[i:i+2], mask[i:i+2]) for i in range(0, 5, 2))
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(torch.autograd.grad(full, logits)[0], torch.autograd.grad(split, logits)[0])


def test_existing_output_and_gradients_unchanged_when_auxiliary_disabled():
    torch.manual_seed(3)
    core = ContactHead(16, 8, 2)
    old = PatchContactHead(copy.deepcopy(core), PatchSpec(4, 2))
    state = torch.get_rng_state().clone()
    new = PatchContactHead(copy.deepcopy(core), PatchSpec(4, 2), coverage=True, seed=19)
    assert torch.equal(state, torch.get_rng_state())
    a, b = torch.randn(2, 9, 16), torch.randn(2, 11, 16)
    ma, mb = torch.ones(2, 9, dtype=torch.bool), torch.ones(2, 11, dtype=torch.bool)
    expected, _ = old(a, b, ma, mb)
    actual, valid, coverage = new.forward_with_coverage(a, b, ma, mb)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert coverage.shape == (*actual.shape, 2)
    actual.sum().backward(); expected.sum().backward()
    for p, q in zip(old.core.parameters(), new.core.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert new.coverage_mix.weight.grad is None


def test_perfect_coverage_metric_and_small_patch_recall():
    row = dict(seq1='A'*16, seq2='C'*16, valid1=[True]*16, valid2=[True]*16,
               contacts=[[0, 0]])
    coverage = torch.logit(torch.full((1, 1, 2), 1/16))
    metrics = structural_patch_metrics(torch.ones(1, 1), row, PatchSpec(), True, coverage)
    assert metrics['coverage_mae'] < 1e-7
    assert metrics['coverage_small_positive_patches'] == 1
    assert metrics['small_patch_existence_recall_at_half'] == 1


def test_coverage_gradient_cache_matches_direct_global_objective():
    from test_research_multigpu import Toy, make_batches
    from piper.distributed import GlobalGradientCache
    from piper.loss import GradCacheClipLoss, ContactLoss

    class CoverageToy(Toy):
        coverage_weight = 0.1

        def __init__(self):
            super().__init__()
            self.coverage_mix = torch.nn.Linear(1, 2)

        def forward(self, *args, return_coverage=False):
            result = super().forward(*args)
            return result + (self.coverage_mix(result[2][..., None]),) if return_coverage else result

    torch.manual_seed(5)
    model = CoverageToy(); direct = copy.deepcopy(model)
    batches = make_batches(7)
    for i, b in enumerate(batches):
        b['coverage_target'] = torch.rand(*b['contact_target'].shape, 2)
        b['coverage_valid_mask'] = b['contact_target'].clone()
        if i == 1: b['coverage_valid_mask'].zero_()
    actual = GlobalGradientCache(model, 'cpu').step(batches, 7)
    eligible = sum(int(b['coverage_valid_mask'].flatten(1).any(1).sum()) for b in batches)
    q, k, ca, cb, contacts, coverage = [], [], [], [], 0., 0.
    cache = GlobalGradientCache(direct, 'cpu')
    for b in batches:
        torch.manual_seed(b['replay_seed'])
        outputs = direct(*cache.arguments(b), return_coverage=True)
        q.append(outputs[0]); k.append(outputs[1])
        ca.append(torch.cat([b['seq1_cluster'], b['seq2_cluster']]))
        cb.append(torch.cat([b['seq2_cluster'], b['seq1_cluster']]))
        contacts += ContactLoss(True)(outputs[2], b['contact_target'], outputs[4], b['contact_target_valid_mask']) * len(b['contact_target']) / 7
        coverage += coverage_loss_sum(outputs[-1], b['coverage_target'], b['coverage_valid_mask']) / eligible
    clip = GradCacheClipLoss(valid_keys_tensor=direct.valid_keys)(torch.cat(q), torch.cat(k),
                     direct.logit_scale.exp(), torch.cat(ca), torch.cat(cb))
    (0.1 * clip + contacts + 0.1 * coverage).backward()
    assert actual['coverage_structures'] == eligible
    assert abs(actual['coverage_loss'] - float(coverage.detach())) < 1e-6
    for p, r in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(p.grad, r.grad, atol=2e-6, rtol=2e-5)


def test_checkpoint_migration_preserves_moments_with_shifted_parameter_order():
    from piper.coverage_migration import restore_existing_parameters, migrate_adam_state
    from piper.train import build_optimizer, cpu_state
    old = torch.nn.Module()
    old.adapter = torch.nn.Module()
    old.adapter.lora_b_q = torch.nn.Parameter(torch.ones(2, 2))
    old.contact_head = torch.nn.Module()
    old.contact_head.core = torch.nn.Linear(2, 2)
    cfg = dict(learning_rate=1e-4, lora_learning_rate=1e-5, weight_decay=.1)
    opt = build_optimizer(old, cfg)
    sum(p.square().sum() for p in old.parameters()).backward(); opt.step()
    new = copy.deepcopy(old)
    new.contact_head = torch.nn.Module()
    new.contact_head.coverage_mix = torch.nn.Linear(2, 2)  # Insert before old core.
    new.contact_head.core = torch.nn.Linear(2, 2)
    saved = {n: p.detach().clone() for n, p in old.named_parameters()}
    restore_existing_parameters(new, saved)
    new_opt = build_optimizer(new, cfg)
    report = migrate_adam_state(new, new_opt, cpu_state(opt.state_dict()), list(saved))
    assert report['preserved_parameter_states'] == 3
    for name, p in new.named_parameters():
        if 'coverage_mix' in name:
            assert not new_opt.state.get(p)
        else:
            torch.testing.assert_close(p, saved[name], atol=0, rtol=0)
