"""Residue/EOS/position alignment through the actual ProtT5 wrapper and buckets."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flashppi.frozen_embedding_cache import FrozenEmbeddingCache
from flashppi.frozen_prott5 import FrozenProtT5
from test_gated_ensemble import PrimaryTokenizer, T5Tokenizer


def test_bucketing_preserves_native_t5_tokens_eos_context_and_positions(monkeypatch, tmp_path):
    import transformers

    class ContextEncoder(nn.Module):
        config = SimpleNamespace(d_model=1024)

        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, input_ids, attention_mask):
            for ids, mask in zip(input_ids, attention_mask):
                self.seen.append(tuple(ids[mask].tolist()))
            # Exact integer-valued arithmetic in float64. An output depends on
            # its token, position and whole valid context, including EOS.
            x = input_ids.double()
            context = (x * attention_mask).sum(1, keepdim=True)
            positions = torch.arange(x.shape[1], dtype=torch.float64)[None, :]
            value = x + 100 * positions + 10000 * context
            return SimpleNamespace(last_hidden_state=value[..., None].expand(-1, -1, 1024))

    encoder = ContextEncoder()
    monkeypatch.setattr(transformers.T5EncoderModel, 'from_pretrained', lambda *a, **k: encoder)
    monkeypatch.setattr(transformers.T5Tokenizer, 'from_pretrained', lambda *a, **k: T5Tokenizer())
    frozen = FrozenProtT5(tmp_path, PrimaryTokenizer())
    sequences = [[3, 7, 5, 6, 8, 9], [4, 7], [3], [4, 3, 7, 4], [4, 7], [7, 3, 4]]
    calls = []
    for start in (0, 2, 4):
        ids = torch.full((2, 9), -99, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, sequence in enumerate(sequences[start:start + 2]):
            ids[row, :len(sequence)] = torch.tensor(sequence)
            mask[row, :len(sequence)] = True
        calls.append((ids, mask))
    expected = [frozen(ids, mask) for ids, mask in calls]
    original_tokens = list(encoder.seen)
    assert original_tokens == [(11, 47, 17, 17, 17, 17, 41), (13, 47, 41),
                               (11, 41), (13, 11, 47, 13, 41), (13, 47, 41), (47, 11, 13, 41)]
    encoder.seen.clear()
    cache = FrozenEmbeddingCache({'prott5': frozen})
    cache.prepare_buckets(calls, 'cpu', 2)
    assert Counter(encoder.seen) == Counter(original_tokens)
    for (ids, mask), reference in zip(calls, expected):
        actual = cache.encode('prott5', frozen, ids, mask)
        assert torch.equal(actual, reference)
        assert torch.count_nonzero(actual[~mask]) == 0
    cache.start_replay()
    for (ids, mask), reference in zip(calls, expected):
        assert torch.equal(cache.encode('prott5', frozen, ids, mask), reference)
    assert cache.finish()['frozen_cache_reused_calls'] == 3
    assert len(encoder.seen) == len(sequences)  # Replay does not encode again.


def test_bucket_handles_an_empty_distributed_rank():
    module = nn.Embedding(4, 2).requires_grad_(False).eval()
    cache = FrozenEmbeddingCache({'empty': module})
    cache.prepare_buckets([], 'cpu', 32)
    cache.start_replay()
    assert cache.finish()['frozen_cache_bytes'] == 0


@pytest.mark.parametrize('mask', [torch.tensor([[1, 0, 1]]), torch.tensor([[0, 0, 0]])])
def test_bucket_rejects_holes_or_empty_sequences(mask):
    module = nn.Embedding(4, 2).requires_grad_(False).eval()
    cache = FrozenEmbeddingCache({'a': module})
    with pytest.raises(ValueError, match='right-padded'):
        cache.prepare_buckets([(torch.ones_like(mask), mask)], 'cpu', 32)
