"""Contact-negative identities, budget, exclusions, replay and legacy isolation."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from piper.dataset import DataCollatorForPPI
from piper.model import get_contact_negatives
from piper.negative_sampling import unique_heteroprotein_indices, validate_policy, POLICY
from piper.loss import ContactLoss, focal_loss
from test_finetune_training import Tokenizer


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def population(self, count=16):
        a, b = torch.arange(count), torch.arange(count) + count
        x, y = torch.cat([a, b]), torch.cat([b, a])
        known = x * 1000 + y
        # Every self candidate is deliberately the highest-scoring hard negative.
        q = torch.eye(2*count)
        k = q[(torch.arange(2*count)+count) % (2*count)]
        return x, y, known, q, k

    def test_replaces_duplicate_self_and_reverse_with_full_budget(self):
        x, y, known, q, k = self.population()
        n = len(x)
        sentinel = torch.arange(n).float().view(n, 1, 1)
        mask = torch.ones(n, 1, dtype=torch.bool)
        old = get_contact_negatives(sentinel, sentinel, mask, mask, 0, 1, 2,
                                    x, y, known, 1000, q, k)
        state = torch.get_rng_state().clone()
        i, j, stats = unique_heteroprotein_indices(x, y, x, y, known, 1000, q, k)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        pairs = [tuple(sorted((int(x[a]), int(y[b])))) for a, b in zip(i, j)]
        self.assertEqual(len(i), len(old[0]))
        self.assertEqual(len(i), 96)
        self.assertEqual(len(set(pairs)), len(pairs))
        self.assertTrue(all(a != b for a, b in pairs))
        self.assertFalse(torch.isin(x[i]*1000+y[j], known).any())
        self.assertEqual(stats['negative_shortfall'], 0)
        self.assertGreater(stats['negative_refill'], 0)
        again = unique_heteroprotein_indices(x, y, x, y, known, 1000, q, k)
        self.assertTrue(torch.equal(i, again[0]) and torch.equal(j, again[1]))

    def test_duplicate_sequences_and_known_positive_any_representation(self):
        x, y, known, q, k = self.population(4)
        sx, sy = x.clone(), y.clone()
        # Two sequence records assigned different clusters represent one protein.
        sx[sx == 1] = 0; sy[sy == 1] = 0
        known = torch.cat([known, torch.tensor([0*1000+6, 6*1000+0])])
        i, j, stats = unique_heteroprotein_indices(x, y, sx, sy, known, 1000, q, k)
        pairs = [tuple(sorted((int(sx[a]), int(sy[b])))) for a, b in zip(i, j)]
        self.assertNotIn((0, 6), pairs)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(a != b for a, b in pairs))
        self.assertEqual(stats['negative_selected'], stats['negative_unique_available'])
        self.assertGreater(stats['negative_shortfall'], 0)

    def test_insufficient_or_empty_pool_is_reported_without_duplicates(self):
        x, y, known, q, k = self.population(1)
        i, j, stats = unique_heteroprotein_indices(x, y, x, y, known, 1000, q, k)
        self.assertEqual(len(i), 0)
        self.assertEqual(stats['negative_shortfall'], stats['negative_budget'])
        x, y, known, q, k = self.population(3)
        known = (x[:, None]*1000+y[None, :]).flatten()
        i, j, stats = unique_heteroprotein_indices(x, y, x, y, known, 1000, q, k)
        self.assertEqual(len(i), 0)
        self.assertEqual(stats['negative_budget'], 0)

    def test_identity_is_pre_crop_and_follows_swap_without_rng_change(self):
        a = 'ACDEFGHIKL'*100
        rows = [dict(seq1=a, seq2='MNPQRSTVWY'*100, contacts=[[5, 5]],
                     seq1_cluster=1, seq2_cluster=2),
                dict(seq1=a, seq2='WY'*500, contacts=[[950, 900]],
                     seq1_cluster=1, seq2_cluster=3)]
        cfg = dict(mode='interface', candidates=16, near_best_fraction=.9)
        old = DataCollatorForPPI(Tokenizer(), 32, True, False, cfg)
        new = DataCollatorForPPI(Tokenizer(), 32, True, False, cfg, sequence_identity=True)
        torch.manual_seed(11); expected = old(rows); rng = torch.get_rng_state().clone()
        torch.manual_seed(11); actual = new(rows)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name in ('contact_target', 'contact_target_valid_mask', 'seq1_cluster', 'seq2_cluster'):
            self.assertTrue(torch.equal(actual[name], expected[name]))
        for name in ('seq1_tokens', 'seq2_tokens'):
            for key in actual[name]:
                self.assertTrue(torch.equal(actual[name][key], expected[name][key]))
        identities = torch.cat([actual['seq1_identity'], actual['seq2_identity']])
        clusters = torch.cat([actual['seq1_cluster'], actual['seq2_cluster']])
        self.assertEqual(len(identities[clusters == 1].unique()), 1)
        with patch('torch.rand', return_value=torch.tensor([0.])):
            swapped = new(rows)
        self.assertTrue(torch.equal(swapped['seq2_identity'], torch.tensor([0, 0])))

    def test_focal_diagnostic_parts_equal_existing_loss(self):
        torch.manual_seed(2)
        z = torch.randn(7, 3, 4, requires_grad=True)
        y = torch.rand(2, 3, 4) > .5
        valid = torch.rand_like(z) > .2
        observed = torch.rand(2, 3, 4) > .1
        mask = valid[:2] & observed
        structural = focal_loss(z[:2], y.float(), mask)
        negative = focal_loss(z[2:], torch.zeros_like(z[2:]), valid[2:]) / y[mask].sum().clamp_min(1)
        torch.testing.assert_close(structural+negative, ContactLoss(True)(z, y, valid, observed))

    def test_policy_rejects_unrelated_sampler_changes(self):
        cfg = dict(contact_negative_sampling=copy.deepcopy(POLICY), contact_self_neg_ratio=0,
                   contact_random_neg_ratio=0, contact_hard_neg_ratio=2, swap_self_negative=True)
        self.assertEqual(validate_policy(cfg), POLICY)
        cfg['contact_random_neg_ratio'] = 1
        with self.assertRaises(ValueError): validate_policy(cfg)
        self.assertIsNone(validate_policy({}))


if __name__ == '__main__':
    unittest.main()
