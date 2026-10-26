"""CPU regressions for source provenance, joint cluster IDs and 1:1 sampling."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from flashppi.mixed_sampling import BalancedKindSampler
from flashppi.research_distributed import GlobalGradientCache
from tools.prepare_conglab_mixture import joint_clusters, lookup


class Tests(unittest.TestCase):
    def test_structure_provenance_not_shared_sequence_source(self):
        table = pa.table(dict(source=["conglab", "conglab+pinder", "conglab+pinder", "pinder"],
                              structure_source=["conglab", "conglab", "pinder", "pinder"]))
        selected = table.filter(pc.equal(table["structure_source"], "conglab"))
        self.assertEqual(len(selected), 2)

    def test_joint_namespaces_merge_only_through_identical_sequences(self):
        ppi = pa.table(dict(seq1_id=["a", "b"], seq2_id=["c", "d"],
                            seq1_cluster=[0, 0], seq2_cluster=[1, 2]))
        ddi = pa.table(dict(seq1_id=["a", "x"], seq2_id=["y", "z"]))
        (p1, p2, d1, d2), report = joint_clusters(ppi, ddi, np.array([10, 0]), np.array([11, 12]))
        self.assertEqual(p1[0], d1[0])  # same actual sequence, despite different source clusters
        self.assertNotEqual(p1[0], d1[1])  # source cluster ID 0 alone does not mean identical
        self.assertEqual(p1[0], p1[1])
        self.assertEqual(len(set(np.r_[p1, p2, d1, d2])), 6)
        self.assertEqual(report["exact_sequences_shared_across_kinds"], 1)

    def test_cluster_lookup_fails_for_missing_sequences(self):
        ids = np.array([b"a", b"b", b"d"], dtype="S68")
        np.testing.assert_array_equal(lookup(["d", "a"], ids, np.array([1, 7, 9])), [9, 1])
        for query in (["c"], ["e"]):
            with self.assertRaises(ValueError):
                lookup(query, ids, np.array([1, 7, 9]))

    def test_exact_ratio_every_batch_and_reproducible_schedule(self):
        kinds = np.array(["ppi"] * 7 + ["ddi"] * 31)
        def draw(seed):
            sampler = BalancedKindSampler(np.arange(38), np.arange(38) + 50, kinds, 32, seed)
            sampler.num_samples = 320
            return np.array(list(sampler))
        a, b = draw(123), draw(123)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, draw(124)))
        for batch in a.reshape(-1, 32):
            self.assertEqual(np.count_nonzero(kinds[batch] == "ppi"), 16)
            self.assertEqual(np.count_nonzero(kinds[batch] == "ddi"), 16)

    def test_sampler_rejects_missing_cluster_sentinels_and_odd_batches(self):
        with self.assertRaises(ValueError):
            BalancedKindSampler([1, -1], [2, -1], ["ppi", "ddi"], 16, 9)
        with self.assertRaises(ValueError):
            BalancedKindSampler([1, 3], [2, 4], ["ppi", "ddi"], 15, 9)

    def test_global_clip_uses_the_same_cluster_key_multiplier_as_contact_mining(self):
        class Model:
            multiplier = 12_345_678
            valid_keys = torch.tensor([12_345_680])
        cache = GlobalGradientCache(Model(), "cpu")
        self.assertEqual(cache.clip_loss.multiplier, Model.multiplier)

    def test_either_endpoint_exclusion_with_arrow_masks(self):
        table = pa.table(dict(seq1_id=["a", "c", "x"], seq2_id=["b", "b", "y"]))
        hits = pa.array(["b"])
        keep = pc.invert(pc.or_(pc.is_in(table["seq1_id"], value_set=hits), pc.is_in(table["seq2_id"], value_set=hits)))
        self.assertEqual(table.filter(keep)["seq1_id"].to_pylist(), ["x"])


if __name__ == "__main__":
    unittest.main()
