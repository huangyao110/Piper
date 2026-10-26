"""Synthetic CPU checks of residue mapping and annotation quality, not biological data."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from clip_gate.annotations import map_positions, missing_atoms, msa_quality, parse_grades
from annotation_report import rank_partial_correlation


class AnnotationTests(unittest.TestCase):
    def test_coordinate_sequence_deletion_and_terminal_truncation(self):
        reference = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
        target = reference[3:15] + reference[19:-2]
        mapping, info = map_positions(target, reference)
        self.assertEqual(mapping, list(range(3, 15)) + list(range(19, len(reference) - 2)))
        self.assertEqual(info["identity"], 1)
        self.assertEqual(info["coverage"], 1)

    def test_reject_ambiguous_repeat_mapping(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            map_positions("ACDEFGHIK", "ACDEFGHIKACDEFGHIK")

    def test_reject_unrelated_reference(self):
        with self.assertRaisesRegex(ValueError, "rejected"):
            map_positions("ACDEFGHIKLMNPQ", "WWWWWWWWWWWWWW")

    def test_sidechain_completeness_is_stricter_than_cb_label(self):
        self.assertEqual(missing_atoms("G", {"N", "CA", "C", "O"}), [])
        self.assertEqual(missing_atoms("A", {"N", "CA", "C", "O", "CB"}), [])
        self.assertEqual(missing_atoms("E", {"N", "CA", "C", "O", "CB"}), ["CD", "CG", "OE1", "OE2"])

    def test_consurf_direction_and_uncertainty(self):
        text = "header\n 1\t A\tALA:102:A\t-1.5\t9\t-1.7,-1.3 9,9\tb\ts\t99/100\tA\n"
        text += " 2\t C\t-\t0.3\t4*\t-0.4,0.8 7,2\te\t \t4/100\tC\n"
        rows = parse_grades(text)
        self.assertEqual(rows[0]["conservation"], 1.5)
        self.assertEqual(rows[1]["conservation"], -.3)
        self.assertTrue(rows[1]["insufficient"])
        self.assertEqual(rows[1]["msa_count"], 4)

    def test_identical_msa_duplicates_do_not_inflate_neff(self):
        text = ">query\nACD-EFGHIK\n>duplicate\nACD-EFGHIK\n>duplicate2\nACD-EFGHIK\n"
        info, neff, coverage = msa_quality(text, "ACDEFGHIK")
        self.assertAlmostEqual(info["neff"], 1)
        np.testing.assert_allclose(neff, 1)
        np.testing.assert_array_equal(coverage, np.ones(9))

    def test_msa_gaps_and_short_fragment_excluded(self):
        text = ">query\nACDEFGHIKL\n>fragment\nAC--------\n>diverse\nWYDEFG----\n"
        info, neff, coverage = msa_quality(text, "ACDEFGHIKL")
        self.assertEqual(info["n_retained"], 2)
        np.testing.assert_allclose(coverage, [1] * 6 + [.5] * 4)
        np.testing.assert_allclose(neff, [2] * 6 + [1] * 4)

    def test_msa_requires_exact_reference(self):
        with self.assertRaisesRegex(ValueError, "exact reference"):
            msa_quality(">a\nACDEFG\n>b\nACDEFG\n", "CCCCCC")

    def test_adjustment_removes_pure_amino_acid_and_rsa_confounding(self):
        aa = np.array(list("A" * 60 + "G" * 60))
        gate = np.arange(120, dtype=float)
        rsa = gate / 120
        outcome = gate[::-1]
        # If the gate is entirely explained by RSA, residual association is
        # unidentifiable, not evidence for a functional relation (and not zero).
        self.assertIsNone(rank_partial_correlation(gate, outcome, aa, rsa))
        self.assertIsNone(rank_partial_correlation(np.repeat([1., 2.], 60), outcome, aa))

    def test_adjusted_relation_retained_within_amino_acid_groups(self):
        aa = np.array(list("ACDEFGHIKL") * 20)
        rng = np.random.default_rng(19)
        gate = rng.normal(size=200)
        rsa = rng.uniform(size=200)
        self.assertAlmostEqual(rank_partial_correlation(gate, gate, aa, rsa), 1.)
        self.assertAlmostEqual(rank_partial_correlation(gate, -gate, aa, rsa), -1.)


if __name__ == "__main__":
    unittest.main()
