"""CPU regressions for global gradients, uneven ranks, fusion and token alignment."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from transformers import BatchEncoding

from piper.loss import GradCacheClipLoss, ContactLoss
from piper.distributed import GlobalGradientCache, assigned_chunks, chunk_seed, broadcast_trainable
from piper.ensemble import ResidualFusion, align_secondary_tokens
from piper.evaluation import score_rows


class Toy(nn.Module):
    clip_dim = 5
    swap_self_negative = True

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 6)
        self.dropout = nn.Dropout(0.2)
        self.head_q = nn.Linear(6, self.clip_dim)
        self.head_k = nn.Linear(6, self.clip_dim)
        self.contact = nn.Linear(6, 4)
        self.logit_scale = nn.Parameter(torch.tensor(0.3))
        # Known off-diagonal positives make cross-rank cluster gathering matter.
        self.register_buffer("valid_keys", torch.tensor([a * 1_000_000 + b
                             for a, b in [(0, 101), (101, 0), (1, 100), (100, 1)]]))

    def encode(self, a, b):
        joined = self.dropout(self.backbone(torch.cat([a, b])))
        x, y = joined.chunk(2)
        q = F.normalize(self.head_q(x.mean(1)), dim=-1)
        k = F.normalize(self.head_k(y.mean(1)), dim=-1)
        q2 = F.normalize(self.head_q(y.mean(1)), dim=-1)
        k2 = F.normalize(self.head_k(x.mean(1)), dim=-1)
        return torch.cat([q, q2]), torch.cat([k, k2]), x, y

    def embed_pair(self, a, b, ma, mb, ca, cb):
        q, k, _, _ = self.encode(a, b)
        return q, k, torch.cat([ca, cb]), torch.cat([cb, ca])

    def forward(self, a, b, ma, mb, ca, cb):
        q, k, x, y = self.encode(a, b)
        logits = self.contact(x) @ self.contact(y).transpose(-1, -2)
        # Synthetic negatives also exercise the contact target expansion path.
        logits = torch.cat([logits, self.contact(x) @ self.contact(x).transpose(-1, -2)])
        return q, k, logits, self.logit_scale.exp(), torch.ones_like(logits, dtype=torch.bool)


def make_batches(count, micro=2):
    generator = torch.Generator().manual_seed(29)
    a = torch.randn(count, 3, 3, generator=generator)
    b = torch.randn(count, 3, 3, generator=generator)
    chunks = []
    for index, ids in assigned_chunks(list(range(count)), micro, 1, 0):
        target = torch.zeros(len(ids), 3, 3, dtype=torch.bool)
        target[:, 0, 0] = True
        target[-1, 1, 2] = True
        structural = torch.ones_like(target)
        structural[:, -1, -1] = False
        chunks.append(dict(seq1_tokens=dict(input_ids=a[ids], attention_mask=torch.ones(len(ids), 3)),
                           seq2_tokens=dict(input_ids=b[ids], attention_mask=torch.ones(len(ids), 3)),
                           seq1_cluster=torch.tensor(ids), seq2_cluster=torch.tensor(ids) + 100,
                           contact_target=target, contact_target_valid_mask=structural,
                           replay_seed=chunk_seed(23, 0, index)))
    return chunks


def reference(model, chunks, count):
    cache = GlobalGradientCache(model, "cpu")
    q, k, ca, cb = [], [], [], []
    contact = 0.0
    for batch in chunks:
        torch.manual_seed(batch["replay_seed"])
        outputs = model(*cache.arguments(batch))
        q.append(outputs[0]); k.append(outputs[1])
        ca.append(torch.cat([batch["seq1_cluster"], batch["seq2_cluster"]]))
        cb.append(torch.cat([batch["seq2_cluster"], batch["seq1_cluster"]]))
        contact = contact + ContactLoss(True)(outputs[2], batch["contact_target"], outputs[4],
                                             batch["contact_target_valid_mask"]) * len(batch["contact_target"]) / count
    clip = GradCacheClipLoss(valid_keys_tensor=model.valid_keys)(torch.cat(q), torch.cat(k),
                        model.logit_scale.exp(), torch.cat(ca), torch.cat(cb))
    (0.1 * clip + contact).backward()
    return dict(clip_loss=float(clip.detach()), contact_loss=float(contact.detach()))


def worker(rank, world, init_file, output, count):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=world)
    torch.manual_seed(11 + rank)
    model = Toy()
    broadcast_trainable(model)
    chunks = make_batches(count)
    result = GlobalGradientCache(model, "cpu").step(chunks[rank::world], count)
    state = {n: p.grad.clone() for n, p in model.named_parameters()}
    # Every rank must finish with identical synchronized parameter gradients.
    for gradient in state.values():
        expected = gradient.clone()
        dist.broadcast(expected, src=0)
        torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
    if rank == 0:
        torch.save(dict(gradients=state, metrics=result), output)
    dist.destroy_process_group()


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fusion_identity_and_learnable_gate(self):
        torch.manual_seed(13)
        fusion = ResidualFusion(4, 6)
        a, b = torch.randn(2, 3, 6), torch.randn(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        output = fusion(a, b, mask)
        torch.testing.assert_close(output, a, atol=0, rtol=0)
        output.square().sum().backward()
        self.assertGreater(abs(float(fusion.alpha.grad)), 0)
        optimizer = torch.optim.SGD(fusion.parameters(), lr=0.01)
        optimizer.step(); optimizer.zero_grad()
        fusion(a, b, mask).square().sum().backward()
        self.assertGreater(float(fusion.projection.weight.grad.norm()), 0)
        torch.testing.assert_close(fusion(a, b, mask)[0, 2], a[0, 2], atol=0, rtol=0)

    def test_token_alignment_keeps_internal_eos_out_of_residues(self):
        ids = torch.tensor([[3, 4, 1, 1], [5, 6, 7, 8]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        tokens = align_secondary_tokens(ids, mask, torch.arange(10) + 10, 0, 2, 1)
        self.assertEqual(tokens.tolist(), [[0, 13, 14, 2, 1, 1], [0, 15, 16, 17, 18, 2]])
        with self.assertRaises(ValueError):
            align_secondary_tokens(ids, torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]]), torch.arange(10), 0, 2, 1)

    def test_partition_never_drops_or_repeats_draws(self):
        for world in (1, 2, 4, 7):
            for count in (1, 17, 181, 1024):
                pieces = [piece for rank in range(world) for piece in assigned_chunks(list(range(count)), 8, world, rank)]
                actual = [item for _, items in sorted(pieces) for item in items]
                self.assertEqual(actual, list(range(count)))

    def test_single_process_cache_matches_direct_with_dropout(self):
        torch.manual_seed(11)
        model = Toy(); direct = copy.deepcopy(model)
        chunks = make_batches(7)
        result = GlobalGradientCache(model, "cpu").step(chunks, 7)
        expected = reference(direct, chunks, 7)
        for p, q in zip(model.parameters(), direct.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
        for key in expected:
            self.assertAlmostEqual(result[key], expected[key], places=6)

    def test_three_processes_match_global_objective_with_uneven_and_empty_ranks(self):
        for count in (7, 1):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                output = str(Path(directory) / "result.pt")
                mp.spawn(worker, args=(3, str(Path(directory) / "init"), output, count), nprocs=3, join=True)
                actual = torch.load(output, weights_only=True)
                torch.manual_seed(11)
                direct = Toy()
                expected = reference(direct, make_batches(count), count)
                for name, parameter in direct.named_parameters():
                    torch.testing.assert_close(actual["gradients"][name], parameter.grad, atol=3e-6, rtol=3e-5)
                for key in expected:
                    self.assertAlmostEqual(actual["metrics"][key], expected[key], places=6)

    def test_evaluation_preserves_full_sequence_and_masks_only_contact_labels(self):
        class Tokenizer:
            def __call__(self, sequences, **kwargs):
                self.assert_no_truncation = kwargs["truncation"] is False
                tokens = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long)
                mask = torch.zeros_like(tokens)
                for i, sequence in enumerate(sequences):
                    tokens[i, :len(sequence)] = torch.tensor([ord(c) - 64 for c in sequence])
                    mask[i, :len(sequence)] = 1
                return BatchEncoding(dict(input_ids=tokens, attention_mask=mask))

        class Head(nn.Module):
            def forward(self, x, mask):
                return (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.head_q, self.head_k = Head(), Head()

            def encode_protein(self, ids, mask):
                return ids.float()[..., None] * self.weight

            def predict_contacts(self, a, b, ma, mb):
                return a @ b.transpose(-1, -2), ma[:, :, None] & mb[:, None, :]

        rows = [dict(seq1="AC", seq2="BD", label=1, contacts=[[0, 1]], valid1=[True, False], valid2=[True, True]),
                dict(seq1="B", seq2="A", label=0)]
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Tokenizer()
            result = score_rows(Model(), tokenizer, rows, Path(directory) / "batch", pair_batch=2)
            single = score_rows(Model(), tokenizer, rows, Path(directory) / "single", pair_batch=1)
        self.assertEqual(result, single)
        self.assertTrue(tokenizer.assert_no_truncation)
        self.assertEqual(result[0]["contacts_precision_at_k"], 1.0)
        self.assertEqual(result[0]["max_contact_logit"], 12.0)
        self.assertEqual(result[0]["length1"], 2)


if __name__ == "__main__":
    unittest.main()
