"""Mathematical invariants for diagnostics; CPU only, no checkpoint download."""
import sys
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from flashppi.research_eval import score_rows
from flashppi.improvement_objectives import listwise_distillation, tail_ranking


class Tokens(SimpleNamespace):
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode_protein(self, ids, mask):
        return ids.float()[..., None].expand(-1, -1, 3)

    def head_q(self, encoded, mask):
        if (mask.sum(1) < 3).any():
            raise AssertionError("A structural-only peptide was sent to the retrieval head")
        return torch.nn.functional.normalize(encoded.sum(1), dim=-1)

    head_k = head_q

    def predict_contacts(self, a, b, ma, mb):
        return a @ b.transpose(1, 2), ma[:, :, None] & mb[:, None, :]


def tiny_tokenizer(seqs, **kwargs):
    ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = 1
    return Tokens(input_ids=ids, attention_mask=(ids != 0).long())


class Tests(unittest.TestCase):
    def test_structural_rows_skip_unused_clip_but_keep_contacts(self):
        rows = [dict(seq1="ABC", seq2="DEFG", label=0),
                dict(seq1="A", seq2="B", label=1, contacts=[[0, 0]])]
        with tempfile.TemporaryDirectory() as directory:
            result = score_rows(TinyModel(), tiny_tokenizer, rows, Path(directory)/"scores", clip_row_count=1)
        self.assertIsNotNone(result[0]["clip_score"])
        self.assertIsNone(result[1]["clip_score"])
        self.assertEqual(result[1]["contacts_precision_at_k"], 1.)

    def test_default_scoring_still_requires_all_clip_rows(self):
        rows = [dict(seq1="ABC", seq2="DEFG", label=0),
                dict(seq1="AAA", seq2="BBBB", label=1, contacts=[[0, 0]])]
        with tempfile.TemporaryDirectory() as directory:
            default = score_rows(TinyModel(), tiny_tokenizer, rows, Path(directory)/"default")
            explicit = score_rows(TinyModel(), tiny_tokenizer, rows, Path(directory)/"explicit", clip_row_count=2)
        self.assertEqual(default, explicit)
        self.assertTrue(all(r["clip_score"] is not None for r in default))

    def test_distillation_teacher_is_detached_and_gate_can_be_empty(self):
        teacher = torch.tensor([[3., 0., -1.]], requires_grad=True)
        student = torch.tensor([[0., 1., 2.]], requires_grad=True)
        loss, count = listwise_distillation(student, teacher)
        loss.backward()
        self.assertEqual(count, 1)
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertLess(student.grad[0,0].item(), 0)
        empty, count = listwise_distillation(student, teacher.flip(-1))
        self.assertEqual(count, 0)
        self.assertEqual(empty.item(), 0.)

    def test_tail_masks_uncertain_negatives_and_has_finite_empty_gradient(self):
        student = torch.tensor([[1., 10., 2., 0.]], requires_grad=True)
        teacher = torch.tensor([[3., 5., -5., -6.]], requires_grad=True)
        loss, count = tail_ranking(student, teacher, top_k=1)
        torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(1.)))
        loss.backward()
        self.assertEqual(count, 1)
        self.assertEqual(student.grad[0,1].item(), 0.)
        self.assertIsNone(teacher.grad)
        student.grad = None
        empty, count = tail_ranking(student, torch.ones_like(teacher)*5)
        empty.backward()
        self.assertEqual(count, 0)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
