"""CPU replay checks; the full production checkpoint has a separate GPU probe."""
import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from piper.frozen_embedding_cache import FrozenEmbeddingCache
from piper.frozen_prefix_cache import frozen_prefix_replay
from piper.full_finetune import merge_lora_and_unfreeze
from piper.distributed import GlobalGradientCache
from test_frozen_embedding_cache import Frozen, batches
from test_full_finetune import source_model
from piper.checkpoint import compare


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(73)


@pytest.mark.parametrize('prefix,bucket', [('cpu', 0), ('off', 3), ('cpu', 3)])
def test_two_steps_preserve_gradients_adam_rng_and_mining(prefix, bucket, monkeypatch):
    initial = source_model()
    merge_lora_and_unfreeze(initial, 2)
    models = [copy.deepcopy(initial), copy.deepcopy(initial)]
    optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
                  for m in models]
    from flashppi import negative_sampling
    original = negative_sampling.unique_heteroprotein_indices
    mined = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        mined.append((result[0].clone(), result[1].clone(), copy.deepcopy(result[2])))
        return result

    monkeypatch.setattr(negative_sampling, 'unique_heteroprotein_indices', record)
    for index in range(2):
        records = []
        for variant, (model, optimizer) in enumerate(zip(models, optimizers)):
            mined.clear()
            optimizer.zero_grad(set_to_none=True)
            metrics = GlobalGradientCache(model, 'cpu', frozen_embedding_cache='cpu',
                frozen_prefix_cache=prefix if variant else 'off',
                frozen_length_bucket_size=bucket if variant else 0).step(batches(index), 5)
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
            optimizer.step()
            records.append(dict(metrics={k: v for k, v in metrics.items() if not k.startswith('frozen_')},
                                gradients=gradients, optimizer=copy.deepcopy(optimizer.state_dict()),
                                rng=torch.get_rng_state().clone(), mined=copy.deepcopy(mined)))
            assert not any(p.grad is not None for p in model.plm.parameters() if not p.requires_grad)
            assert getattr(model, '_frozen_prefix_cache', None) is None
            assert all('forward' not in layer.__dict__ for layer in model.plm.encoder.layers)
        compare(records[0], records[1])
        compare(models[0].state_dict(), models[1].state_dict())
        if prefix == 'cpu':
            assert metrics['frozen_prefix_captured_calls'] == metrics['frozen_prefix_reused_calls'] == 3
            assert metrics['frozen_prefix_blocks'] == 2


def test_bucket_restores_order_mask_tokens_padding_and_dtype():
    encoder = Frozen(4).double()
    inputs = [(torch.tensor([[1, 31, 0, 0], [4, 5, 6, 7]]), torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])),
              (torch.tensor([[8, 0, 0, 0]]), torch.tensor([[1, 0, 0, 0]]))]
    cache = FrozenEmbeddingCache({'a': encoder})
    cache.prepare_buckets(inputs, 'cpu', 2)
    for ids, mask in inputs:
        assert torch.equal(cache.encode('a', encoder, ids, mask), encoder(ids, mask))
    cache.start_replay()
    for ids, mask in inputs:
        assert torch.equal(cache.encode('a', encoder, ids, mask), encoder(ids, mask))
    assert cache.finish()['frozen_cache_bucket_positions'] == 8


@pytest.mark.parametrize('packed', [False, True])
def test_prefix_with_native_activation_checkpointing_and_unpadding(packed):
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 8).requires_grad_(False)
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            for layer in self.encoder.layers[:2]:
                layer.requires_grad_(False)

        def forward(self, ids, mask):
            x = self.embed(ids)
            shape = x.shape
            if packed:
                x = x[mask.bool()]
            for layer in self.encoder.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            if packed:
                padded = x.new_zeros(shape)
                padded[mask.bool()] = x
                x = padded
            return x

    model = nn.Module()
    model.plm = Backbone()
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = ids != 0
    baseline = copy.deepcopy(model)
    expected = baseline.plm(ids, mask)
    expected.square().sum().backward()
    keys = list(model.state_dict())
    with frozen_prefix_replay(model, 'cpu') as cache:
        with torch.no_grad():
            captured = cache.encode(ids, mask)
        cache.start_replay()
        replayed = cache.encode(ids, mask)
        replayed.square().sum().backward()
        cache.finish()
    assert torch.equal(captured, replayed) and torch.equal(expected, replayed)
    assert list(model.state_dict()) == keys
    for left, right in zip(model.parameters(), baseline.parameters()):
        compare(left.grad, right.grad)


@pytest.mark.parametrize('fault', ['input', 'precision', 'mutation', 'incomplete', 'exception'])
def test_prefix_guards_and_exception_cleanup(fault):
    model = source_model()
    merge_lora_and_unfreeze(model, 2)
    ids = torch.tensor([[1, 31, 0]])
    mask = ids != 0
    with pytest.raises(RuntimeError):
        with frozen_prefix_replay(model, 'cpu') as cache:
            with torch.no_grad():
                cache.encode(ids, mask)
            if fault == 'mutation':
                with torch.no_grad():
                    next(model.plm.encoder.layers[0].parameters()).add_(1)
            if fault == 'exception':
                raise RuntimeError('abort')
            cache.start_replay()
            if fault == 'incomplete':
                cache.finish()
            if fault == 'input':
                ids[0, 0] = 2
            with torch.autocast('cpu', enabled=fault == 'precision', dtype=torch.bfloat16):
                cache.encode(ids, mask)
    assert model._frozen_prefix_cache is None
    assert all('forward' not in layer.__dict__ for layer in model.plm.encoder.layers)
    assert cache.entries == []


def test_prefix_rejects_trainable_embedding_and_stochastic_prefix():
    model = source_model()
    merge_lora_and_unfreeze(model, 2)
    model.plm.embed.requires_grad_(True)
    with pytest.raises(ValueError, match='embedding'):
        with frozen_prefix_replay(model, 'cpu'):
            pass
    model.plm.embed.requires_grad_(False)
    model.plm.encoder.layers[0].ffn.add_module('dropout', nn.Dropout(.2))
    with pytest.raises(RuntimeError, match='RNG'):
        with frozen_prefix_replay(model, 'cpu') as cache, torch.no_grad():
            cache.encode(torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool))
