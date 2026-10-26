"""CPU regressions for patch grids, unknown labels and matched-resolution metrics."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from flashppi.loss import ContactLoss
from flashppi.patch_contact import (PatchSpec, PatchContactHead, PatchTargetCollator,
    patch_mask, patch_targets, pool_embeddings, pool_matrix_max, structural_patch_metrics)


class DotHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3)

    def forward(self, a, b, ma, mb):
        return self.proj(a) @ self.proj(b).transpose(1, 2), ma[:, :, None] & mb[:, None, :]


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_every_residue_including_tail_is_covered(self):
        for spec in (PatchSpec(16, 4), PatchSpec(20, 4), PatchSpec(3, 3), PatchSpec(1, 1)):
            for n in range(1, 70):
                covered = set()
                for u in range(spec.count(n)):
                    covered.update(range(u * spec.stride, min(n, u * spec.stride + spec.window)))
                self.assertEqual(covered, set(range(n)))
        self.assertEqual(PatchSpec().count(512), 125)
        self.assertEqual(PatchSpec().count(17), 2)

    def test_pool_matches_manual_windows_and_ignores_padding(self):
        spec = PatchSpec()
        for n in (1, 7, 16, 17, 20, 21, 35):
            x = torch.randn(1, n, 3)
            expected = torch.stack([x[0, u * 4:min(n, u * 4 + 16)].mean(0)
                                    for u in range(spec.count(n))])[None]
            pooled, _ = pool_embeddings(x, torch.ones(1, n, dtype=torch.bool), spec)
            torch.testing.assert_close(pooled, expected)
            padded = torch.cat([x, torch.full((1, 71 - n, 3), float("nan"))], dim=1)
            mask = torch.arange(71)[None] < n
            actual, valid = pool_embeddings(padded, mask, spec)
            torch.testing.assert_close(actual[:, :spec.count(n)], expected)
            self.assertEqual(int(valid.sum()), spec.count(n))
            self.assertEqual(int(actual[~valid].count_nonzero()), 0)

    def test_pool_backpropagates_only_to_real_residues(self):
        x = torch.randn(2, 40, 3, requires_grad=True)
        mask = torch.arange(40)[None] < torch.tensor([17, 35])[:, None]
        pooled, valid = pool_embeddings(x, mask, PatchSpec())
        pooled[valid].sum().backward()
        self.assertTrue((x.grad[mask] > 0).all())
        self.assertTrue((x.grad[~mask] == 0).all())

    def test_targets_match_explicit_or_and_unknown_policy(self):
        spec = PatchSpec(4, 2)
        mask1 = torch.arange(11)[None] < torch.tensor([7, 11])[:, None]
        mask2 = torch.arange(9)[None] < torch.tensor([5, 9])[:, None]
        target = torch.rand(2, 11, 9) < 0.1
        observed = torch.rand(2, 11, 9) > 0.08
        labels, valid = patch_targets(target, observed, mask1, mask2, spec)
        for b, (n, m) in enumerate(((7, 5), (11, 9))):
            for u in range(spec.count(11)):
                for v in range(spec.count(9)):
                    if u >= spec.count(n) or v >= spec.count(m):
                        self.assertFalse(labels[b, u, v]); self.assertFalse(valid[b, u, v])
                        continue
                    a, z = slice(2*u, min(n, 2*u+4)), slice(2*v, min(m, 2*v+4))
                    positive = bool((target[b, a, z] & observed[b, a, z]).any())
                    known = bool(observed[b, a, z].all())
                    self.assertEqual(bool(labels[b, u, v]), positive)
                    self.assertEqual(bool(valid[b, u, v]), positive or known)

    def test_unknown_coordinates_are_not_false_negative_patches(self):
        target = torch.zeros(1, 20, 20, dtype=torch.bool)
        observed = torch.ones_like(target)
        observed[:, 0, :] = False
        seq = torch.ones(1, 20, dtype=torch.bool)
        labels, valid = patch_targets(target, observed, seq, seq, PatchSpec())
        self.assertEqual(valid.tolist(), [[[False, False], [True, True]]])
        target[0, 1, 1] = True
        labels, valid = patch_targets(target, observed, seq, seq, PatchSpec())
        self.assertTrue(labels[0, 0, 0]); self.assertTrue(valid[0, 0, 0])
        target[0, 0, 19] = True  # An unobserved contact cannot make a patch positive.
        labels, _ = patch_targets(target, observed, seq, seq, PatchSpec())
        self.assertFalse(labels[0, 0, 1])

    def test_partial_tail_contacts_and_interior_overlap_multiplicity(self):
        for n, index, multiplicity in ((40, 15, 16), (17, 16, 1), (7, 6, 1)):
            target = torch.zeros(1, n, n, dtype=torch.bool)
            target[:, index, index] = True
            seq = torch.ones(1, n, dtype=torch.bool)
            labels, valid = patch_targets(target, torch.ones_like(target), seq, seq, PatchSpec())
            self.assertEqual(int(labels.sum()), multiplicity)
            self.assertTrue(valid.all())

    def test_window_one_reuses_exact_head_and_gradients(self):
        core = DotHead(); original = copy.deepcopy(core)
        patched = PatchContactHead(core, PatchSpec(1, 1))
        a, b = torch.randn(2, 5, 3), torch.randn(2, 7, 3)
        ma, mb = torch.ones(2, 5, dtype=torch.bool), torch.ones(2, 7, dtype=torch.bool)
        actual, av = patched(a, b, ma, mb)
        expected, ev = original(a, b, ma, mb)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(av, ev))
        actual.sum().backward(); expected.sum().backward()
        for p, q in zip(core.parameters(), original.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)

    def test_negative_expansion_and_unsupervised_gradients(self):
        logits = torch.zeros(3, 2, 2, requires_grad=True)
        target = torch.zeros(1, 2, 2, dtype=torch.bool); target[:, 0, 0] = True
        observed = torch.ones_like(target); observed[:, 0, 1] = False
        loss = ContactLoss(True)(logits, target, torch.ones_like(logits, dtype=torch.bool), observed)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0, 1]), 0)
        self.assertLess(float(logits.grad[0, 0, 0]), 0)
        self.assertTrue((logits.grad[1:] > 0).all())

    def test_baseline_and_patch_evaluation_use_identical_targets(self):
        row = dict(seq1="A" * 21, seq2="C" * 17, contacts=[[0, 0], [20, 16]],
                   valid1=[True] * 21, valid2=[True] * 17)
        logits = torch.randn(21, 17)
        patch_logits = pool_matrix_max(logits[None], PatchSpec(), -torch.inf)[0]
        a = structural_patch_metrics(logits, row, PatchSpec(), False)
        b = structural_patch_metrics(patch_logits, row, PatchSpec(), True)
        self.assertEqual(a, b)
        self.assertEqual(a["patch_positive_count"], 2)
        self.assertEqual(a["patch_total_count"], 6)
        self.assertEqual(float(patch_logits.max()), float(logits.max()))

    def test_collator_preserves_tokens_and_clusters(self):
        tokens = {"input_ids": torch.arange(20)[None], "attention_mask": torch.ones(1, 20, dtype=torch.bool)}
        def collate(rows):
            return dict(seq1_tokens=tokens, seq2_tokens=tokens,
                        contact_target=torch.ones(1, 20, 20, dtype=torch.bool),
                        contact_target_valid_mask=torch.ones(1, 20, 20, dtype=torch.bool),
                        seq1_cluster=torch.tensor([9]), sequence_masking_counts=torch.tensor([6, 40]))
        batch = PatchTargetCollator(collate, PatchSpec())([{}])
        self.assertIs(batch["seq1_tokens"], tokens)
        self.assertEqual(batch["contact_target"].shape, (1, 2, 2))
        self.assertEqual(batch["seq1_cluster"].tolist(), [9])
        self.assertEqual(batch["sequence_masking_counts"].tolist(), [6, 40])

    def test_rejects_invalid_windows_and_nonsequence_masks(self):
        for w, s in ((0, 4), (4, 0), (4, 5), (4.0, 1)):
            with self.assertRaises(ValueError):
                PatchSpec(w, s)
        for mask in (torch.tensor([[0, 0]]), torch.tensor([[1, 0, 1]])):
            with self.assertRaises(ValueError):
                patch_mask(mask, PatchSpec())


if __name__ == "__main__":
    unittest.main()
