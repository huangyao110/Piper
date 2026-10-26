"""CPU regressions for masking, structural alignment and gradient-cache replay."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import BatchEncoding

from piper.dataset import DataCollatorForPPI
from piper.distributed import GlobalGradientCache, assigned_chunks, chunk_seed
from piper.ensemble import align_secondary_tokens, secondary_token_mapping
from piper.sequence_masking import ResidueMasker, SequenceMaskingCollator
from test_research_multigpu import Toy, reference


class Tokenizer:
    def __init__(self, special=False):
        self.special = special
        self.vocab = {s: i for i, s in enumerate(
            ["<cls>", "<pad>", "<eos>", "<unk>", *"ACDEFGHIKLMNPQRSTVWY", "X", "<mask>"])}
        self.cls_token_id, self.pad_token_id, self.eos_token_id = 0, 1, 2
        self.unk_token_id, self.mask_token_id, self.sep_token_id = 3, 25, None

    def get_vocab(self):
        return self.vocab

    def __call__(self, seqs, **kwargs):
        width = kwargs["max_length"]
        ids = torch.full((len(seqs), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            values = [self.vocab.get(s, self.unk_token_id) for s in seq[:width - 2 if self.special else width]]
            if self.special:
                values = [self.cls_token_id, *values, self.eos_token_id]
            ids[i, :len(values)] = torch.tensor(values)
            mask[i, :len(values)] = 1
        return BatchEncoding(dict(input_ids=ids, attention_mask=mask))


class TokenToy(Toy):
    def __init__(self):
        super().__init__()
        self.tokens = torch.nn.Embedding(26, 3)
        self.seen = []

    def encode(self, a, b):
        self.seen.append((a.detach().clone(), b.detach().clone()))
        return super().encode(self.tokens(a), self.tokens(b))


def rows(count=3):
    a, b = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN", "YWVTSRQPNMLKIHGFEDCA"
    return [dict(seq1=a, seq2=b,
                 contacts=[(3, 4), (12, 13)], valid1=[True] * len(a), valid2=[True] * len(b),
                 seq1_cluster=i, seq2_cluster=i + 100) for i in range(count)]


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_special_padding_unknown_and_source_tokens_are_preserved(self):
        tok = Tokenizer(special=True)
        batch = tok(["ACDX", "A"], max_length=8)
        original = copy.deepcopy(batch)
        output, (masked, eligible) = ResidueMasker(tok, 1)(batch, torch.Generator().manual_seed(7))
        self.assertEqual((masked, eligible), (2, 4))
        unchanged = (batch.input_ids < 4) | (batch.input_ids == tok.vocab["X"])
        torch.testing.assert_close(output["input_ids"][unchanged], batch.input_ids[unchanged])
        torch.testing.assert_close(output["attention_mask"], batch.attention_mask)
        torch.testing.assert_close(batch.input_ids, original.input_ids)
        self.assertEqual(int((output["input_ids"][1] == tok.mask_token_id).sum()), 0)

    def test_crop_swap_and_targets_are_unchanged(self):
        for special in (False, True):
            with self.subTest(special=special):
                source = rows()
                original = copy.deepcopy(source)
                collator = DataCollatorForPPI(Tokenizer(special), 12, True, special,
                                            {"mode": "interface", "candidates": 16, "near_best_fraction": 0.9})
                torch.manual_seed(77)
                plain = collator(source)
                torch.manual_seed(77)
                masked = SequenceMaskingCollator(collator, 0.5)(source)
                for key in ("contact_target", "contact_target_valid_mask", "seq1_cluster", "seq2_cluster"):
                    torch.testing.assert_close(plain[key], masked[key], atol=0, rtol=0)
                self.assertEqual(plain["crop_metadata"], masked["crop_metadata"])
                self.assertEqual(source, original)
                changes = 0
                for side in ("seq1_tokens", "seq2_tokens"):
                    torch.testing.assert_close(plain[side]["attention_mask"], masked[side]["attention_mask"])
                    changed = plain[side]["input_ids"] != masked[side]["input_ids"]
                    self.assertTrue(torch.all(masked[side]["input_ids"][changed] == 25))
                    changes += int(changed.sum())
                self.assertGreater(changes, 0)
                self.assertEqual(changes, int(masked["sequence_masking_counts"][0]))

    def test_disabled_path_consumes_no_extra_rng(self):
        collator = DataCollatorForPPI(Tokenizer(), 32, False, False)
        plain = collator(rows())
        before = torch.get_rng_state().clone()
        actual = SequenceMaskingCollator(collator, 0)(rows())
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(set(plain), set(actual))
        for side in ("seq1_tokens", "seq2_tokens"):
            torch.testing.assert_close(plain[side]["input_ids"], actual[side]["input_ids"], atol=0, rtol=0)

    def test_mask_rate_and_seed_are_reproducible(self):
        tok = Tokenizer()
        batch = tok(["ACDEFGHIKLMNPQRSTVWY" * 50] * 100, max_length=1000)
        masker = ResidueMasker(tok)
        a, counts = masker(batch, torch.Generator().manual_seed(21))
        b, repeated = masker(batch, torch.Generator().manual_seed(21))
        c, _ = masker(batch, torch.Generator().manual_seed(22))
        torch.testing.assert_close(a["input_ids"], b["input_ids"], atol=0, rtol=0)
        self.assertEqual(counts, repeated)
        self.assertFalse(torch.equal(a["input_ids"], c["input_ids"]))
        self.assertAlmostEqual(counts[0] / counts[1], 0.15, delta=0.005)

    def test_mask_token_is_translated_to_secondary_native_mask(self):
        primary, secondary = Tokenizer(), Tokenizer()
        secondary.vocab["<mask>"] = secondary.mask_token_id = 30
        mapping = secondary_token_mapping(primary, secondary)
        ids = torch.tensor([[primary.vocab["A"], primary.mask_token_id, primary.pad_token_id]])
        mask = torch.tensor([[1, 1, 0]])
        actual = align_secondary_tokens(ids, mask, mapping, 0, 2, 1)
        self.assertEqual(actual.tolist(), [[0, secondary.vocab["A"], 30, 2, 1]])
        self.assertNotEqual(int(mapping[primary.mask_token_id]), secondary.vocab["X"])

    def test_invalid_probability_and_missing_native_mask_fail(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ResidueMasker(Tokenizer(), value)
        tok = Tokenizer()
        for value in (None, tok.pad_token_id, tok.vocab["A"], 1000):
            tok.mask_token_id = value
            with self.assertRaises(ValueError):
                ResidueMasker(tok, 0.15)

    def test_global_chunk_seed_preserves_masks_across_ranks(self):
        collator = SequenceMaskingCollator(DataCollatorForPPI(Tokenizer(), 32, True, False), 0.15)
        source = rows(7)

        def build(world):
            result = {}
            for rank in range(world):
                for index, selected in assigned_chunks(list(range(7)), 2, world, rank):
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(chunk_seed(23, 17, index))
                        result[index] = collator([source[i] for i in selected])
            return result

        a, b = build(1), build(3)
        for index in a:
            for side in ("seq1_tokens", "seq2_tokens"):
                torch.testing.assert_close(a[index][side]["input_ids"], b[index][side]["input_ids"], atol=0, rtol=0)

    def test_two_pass_cache_reuses_masks_and_matches_direct_gradient(self):
        source = rows(5)
        collator = SequenceMaskingCollator(DataCollatorForPPI(Tokenizer(), 8, False, False), 0.2)
        batches = []
        for index, selected in assigned_chunks(list(range(5)), 2, 1, 0):
            seed = chunk_seed(23, 0, index)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                batch = collator([source[i] for i in selected])
            batch["replay_seed"] = seed
            batches.append(batch)
        torch.manual_seed(11)
        model = TokenToy()
        direct = copy.deepcopy(model)
        actual = GlobalGradientCache(model, "cpu").step(batches, 5)
        expected = reference(direct, batches, 5)
        n = len(batches)
        self.assertEqual(len(model.seen), 2 * n)
        for first, second in zip(model.seen[:n], model.seen[n:]):
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(model.parameters(), direct.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)
        for key in expected:
            # FP32 reduction order can differ by one ULP between CPU builds.
            # Keep token/mask replay exact and the gradient checks above intact.
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
