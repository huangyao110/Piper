"""CPU regressions; synthetic inputs only, never reported as biological results."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from clip_gate.common import local_path, sequence_id, setup_environment
setup_environment()

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from clip_gate.analysis import (aggregate_associations, analyze, association_rows, interaction_report,
                                load_annotations)
from clip_gate.extraction import gpu_available
from clip_gate.gates import extract_branch, permute_gate, project, raw_gate
from clip_gate.panel import interface_labels, make_panel, write_table
from clip_gate.statistics import pair_scores, residue_metrics, top_enrichment


def native_module(alias):
    path = RESEARCH / f"tests/fixtures/{alias}_gated_modules.py"
    spec = importlib.util.spec_from_file_location(f"snapshot_gate_{alias}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GateMathTests(unittest.TestCase):
    def test_native_replay_padding_and_permutation(self):
        for alias in ("before_1616", "after_2182"):
            module = native_module(alias)
            torch.manual_seed(11)
            core = nn.Module()
            core.head = nn.Sequential(nn.Linear(8, 6), nn.SiLU(), nn.Linear(6, 4))
            head = module.GatedContrastiveHead(core, 8, 3).eval()
            nn.init.normal_(head.residue_pool.score[-1].weight)
            x = torch.randn(1, 7, 8)
            mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
            x[0, 5:] = float("nan")
            with torch.inference_mode():
                raw, alpha, relative, vectors = extract_branch(head, x, mask, "a", "q", 19, 2, True)
                weights = raw_gate(head, x, mask)
                actual = project(head, x, mask, weights)
                torch.testing.assert_close(actual, head(x, mask), rtol=0, atol=0)
                torch.testing.assert_close(actual, head(x[:, :5], mask[:, :5]), rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(alpha.sum(), 1)
                np.testing.assert_allclose(relative.mean(), 1)
                self.assertEqual(len(raw), 5)
                self.assertEqual(set(vectors), {"learned", "uniform", "shuffle_000", "shuffle_001"})
                shuffled = permute_gate(weights, mask, "a", "q", 19, 0)
                torch.testing.assert_close(shuffled, permute_gate(weights, mask, "a", "q", 19, 0), rtol=0, atol=0)
                torch.testing.assert_close(weights.sort(dim=1).values, shuffled.sort(dim=1).values, rtol=0, atol=0)
                self.assertEqual(float(shuffled[0, 5:].sum()), 0)
                torch.testing.assert_close(project(head, x, mask, weights * .5), actual, rtol=0, atol=0)

    def test_initial_gate_is_uniform_and_not_importance(self):
        module = native_module("after_2182")
        core = nn.Module()
        core.head = nn.Linear(4, 2)
        head = module.GatedContrastiveHead(core, 4)
        with torch.inference_mode():
            raw, alpha, relative, _ = extract_branch(head, torch.randn(1, 10, 4), torch.ones(1, 10),
                                                     "s", "k", 1, 0, False)
        np.testing.assert_array_equal(raw, .5 * np.ones(10))
        np.testing.assert_allclose(relative, np.ones(10))

    def test_masked_only_input_rejected(self):
        module = native_module("after_2182")
        core = nn.Module()
        core.head = nn.Linear(4, 2)
        head = module.GatedContrastiveHead(core, 4)
        with self.assertRaises(FloatingPointError):
            raw_gate(head, torch.zeros(1, 3, 4), torch.zeros(1, 3))


class BiologicalReferenceTests(unittest.TestCase):
    def row(self):
        return dict(seq1="ACDE", seq2="FGH", contacts=[[1, 0]], valid1=[True] * 4,
                    valid2=[True] * 3, seq1_cluster=1, seq2_cluster=2, pair_id="p")

    def test_existing_contact_union_and_unknown_mask(self):
        row = self.row()
        self.assertEqual(interface_labels(row), [[0, 1, 0, 0], [1, 0, 0]])
        row["valid2"][2] = False
        self.assertEqual(interface_labels(row), [[-1, 1, -1, -1], [1, 0, -1]])
        row["contacts"] = [[1, 2]]
        with self.assertRaises(ValueError):
            interface_labels(row)

    def test_invalid_indices_rejected(self):
        for contacts in ([[1.0, 0]], [[4, 0]], [[-1, 0]]):
            row = self.row()
            row["contacts"] = contacts
            with self.assertRaises(ValueError):
                interface_labels(row)

    def test_exact_sequence_dedup_preserves_membership(self):
        row = self.row()
        binary = dict(seq1=row["seq1"], seq2=row["seq2"], pair_id="i", label=0)
        seqs, interfaces, pairs = make_panel([row, row], [binary])
        self.assertEqual(len(seqs), 2)
        self.assertEqual(len(interfaces), 4)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len({x["block_id"] for x in interfaces}), 1)
        self.assertTrue(all(x["for_interface"] and x["for_interaction"] for x in seqs))
        self.assertEqual(pairs[0]["label"], 0)

    def test_ties_uniform_and_incomplete_observations(self):
        y = np.array([0, 0, 1, 0, 1, 0, 0])
        self.assertAlmostEqual(top_enrichment(y, np.ones(7), .1), 1)
        self.assertAlmostEqual(top_enrichment(y, y, .1), 3.5)
        self.assertIsNone(residue_metrics([-1, 1, 1], [0, .8, .9], .1))
        result = residue_metrics([-1, 1, 0], [1, .8, .1], .1)
        self.assertEqual(result["ap"], 1)
        self.assertEqual(result["n_observed"], 2)

    def test_end_to_end_association_macro_and_missing_annotations(self):
        row = self.row()
        seqs, interfaces, _ = make_panel([row], [])
        sequences = {x["sequence_id"]: x["sequence"] for x in seqs}
        gates = {(sid, branch): dict(relative=np.ones(len(sequence)))
                 for sid, sequence in sequences.items() for branch in ("q", "k")}
        config = dict(top_fraction=.1, seed=13, annotation_permutations=3, bootstrap_replicates=10)
        rows = association_rows(config, sequences, gates, interfaces, {})
        result = aggregate_associations(config, rows)
        for branch in ("q", "k"):
            self.assertEqual(result[branch]["metrics"]["top_enrichment"]["sequence_macro"], 1)
            self.assertEqual(result[branch]["metrics"]["aa_excess_enrichment"]["sequence_macro"], 0)
            self.assertIsNone(result[branch]["metrics"]["aa_rsa_excess_enrichment"]["sequence_macro"])
            self.assertIsNone(result[branch]["metrics"]["top_enrichment"]["block_macro"]["ci95"])

    def test_annotations_reject_misalignment_and_duplicates(self):
        with tempfile.TemporaryDirectory(dir=local_path("tmp")) as temporary:
            path = Path(temporary) / "annotations.csv"
            path.with_suffix(".json").write_text(json.dumps(dict(
                conservation_direction="higher_is_more_conserved", method="test", source="synthetic")))
            sid = sequence_id("ACD")
            header = "sequence_id,position,aa,conservation,relative_sasa\n"
            path.write_text(header + f"{sid},2,C,0.8,0.4\n")
            values, _ = load_annotations(path, {sid: "ACD"})
            self.assertTrue(np.isnan(values[(sid, 2)]["msa_neff"]))
            path.write_text(header + f"{sid},2,A,0.8,0.4\n")
            with self.assertRaises(ValueError):
                load_annotations(path, {sid: "ACD"})
            path.write_text(header + 2 * f"{sid},2,C,0.8,0.4\n")
            with self.assertRaises(ValueError):
                load_annotations(path, {sid: "ACD"})


class InterventionAndIsolationTests(unittest.TestCase):
    def test_busy_gpu_stops_before_loading(self):
        with patch("clip_gate.extraction.subprocess.check_output", side_effect=["12345\n", "2000\n"]):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                gpu_available(dict(minimum_free_gpu_mib=18000))

    def test_only_labeled_pairs_and_separate_branches(self):
        pairs = [dict(row_index=0, pair_id="a", sequence1="a", sequence2="b", label=1),
                 dict(row_index=1, pair_id="b", sequence1="a", sequence2="c", label=0)]
        vectors = {}
        for sid, learned in (("a", [1, 0]), ("b", [1, 0]), ("c", [0, 1])):
            for branch in ("q", "k"):
                vectors[(sid, branch, "learned")] = learned
                vectors[(sid, branch, "uniform")] = [0, 1]
                vectors[(sid, branch, "shuffle_000")] = learned
        scores = pair_scores(pairs, vectors, 1)
        self.assertEqual(len(scores), 14)
        result = interaction_report(scores)
        self.assertEqual(result["learned"]["ap"], 1)
        self.assertEqual(result["uniform_q"]["ap"], .5)
        self.assertEqual(result["shuffle_000_q"]["ap"], 1)
        self.assertEqual(result["learned"]["eligible_labeled_candidate_queries"], 1)
        self.assertEqual(result["learned"]["labeled_candidate_mrr"], 1)
        single = interaction_report([dict(pairs[0], variant="learned", clip_score=1),
                                     dict(pairs[1], sequence1="other", variant="learned", clip_score=0)])
        self.assertEqual(single["learned"]["eligible_labeled_candidate_queries"], 0)
        self.assertIsNone(single["learned"]["labeled_candidate_mrr"])

    def test_output_escape_rejected_including_symlink(self):
        with self.assertRaises(ValueError):
            local_path("../flashppi/probe.py")
        with tempfile.TemporaryDirectory(dir=local_path("tmp")) as temporary:
            link = Path(temporary) / "external"
            link.symlink_to(RESEARCH.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                local_path(link / "probe.py")

    def test_optional_metrics_survive_first_excluded_row(self):
        with tempfile.TemporaryDirectory(dir=local_path("tmp")) as temporary:
            path = Path(temporary) / "rows.parquet"
            write_table(path, [dict(eligible=False), dict(eligible=True, ap=.75)])
            self.assertEqual(pq.read_table(path).to_pylist()[1]["ap"], .75)

    def test_synthetic_full_report_pipeline(self):
        # Exercise report tables/JSON/PNG without loading a model or saving fake
        # findings among real results. Temporary fixtures are removed on exit.
        with tempfile.TemporaryDirectory(dir=local_path("tmp")) as temporary:
            base = Path(temporary)
            output = base / "results/synthetic"
            output.mkdir(parents=True)
            row = BiologicalReferenceTests().row()
            binary = [dict(seq1=row["seq1"], seq2=row["seq2"], pair_id="positive", label=1),
                      dict(seq1=row["seq2"], seq2=row["seq1"], pair_id="negative", label=0)]
            seqs, interfaces, pairs = make_panel([row], binary)
            sequences = {r["sequence_id"]: r["sequence"] for r in seqs}
            write_table(base / "data/interfaces.parquet", interfaces)
            write_table(base / "data/pairs.parquet", pairs)
            gates, vectors = {}, []
            for sid, sequence in sequences.items():
                for branch in ("q", "k"):
                    raw = np.linspace(.2, .8, len(sequence))
                    gates[(sid, branch)] = dict(raw=raw, alpha=raw / raw.sum(), relative=raw / raw.mean())
                    for variant in ("learned", "uniform", "shuffle_000"):
                        vectors.append(dict(sequence_id=sid, branch=branch, variant=variant, vector=[.8, .6]))
            write_table(output / "vectors.parquet", vectors)
            (output / "provenance.json").write_text("{}\n")
            config = dict(seed=3, top_fraction=.1, annotation_permutations=2, bootstrap_replicates=5, gate_shuffles=1)

            def test_path(path):
                path = Path(path)
                return path if path.is_absolute() else base / path

            with patch("clip_gate.analysis.load_result", return_value=({"summary": {"synthetic_test": True}},
                                                                       output, sequences, gates)), \
                 patch("clip_gate.analysis.local_path", side_effect=test_path):
                analyze(config, "synthetic")
            report = json.loads((output / "analysis/summary.json").read_text())
            self.assertEqual(report["clip_interventions"]["learned"]["ap"], .5)
            self.assertFalse(report["annotations"]["conservation"])
            self.assertTrue((output / "analysis/overview.png").stat().st_size > 1000)
            self.assertTrue((output / "analysis/REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
