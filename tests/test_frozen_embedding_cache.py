"""Replay equivalence through real fusion, gates, concat head and negative loss."""
import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from flashppi.frozen_embedding_cache import FrozenEmbeddingCache, frozen_embedding_replay
from flashppi.gated_modules import EnsembleResidualFusion, GatedContrastiveHead, ResidueGatedContactHead
from flashppi.model import ContactHead, ContrastiveHead
from flashppi.negative_sampling import POLICY
from flashppi.research_distributed import GlobalGradientCache, chunk_seed
from flashppi.research_model import ResearchModel
from tools.check_research_replay import compare


class Frozen(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Embedding(32, width)
        self.requires_grad_(False)
        self.calls = 0
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, ids, mask):
        self.calls += 1
        return self.embed(ids).masked_fill(~mask.bool()[..., None], 0)


class Primary(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16).requires_grad_(False)
        self.lora_projection = nn.Linear(16, 16)
        self.dropout = nn.Dropout(.2)

    def forward(self, ids, mask):
        x = self.embed(ids)
        return (x + self.lora_projection(self.dropout(x))).masked_fill(~mask.bool()[..., None], 0)


def tiny_model():
    model = ResearchModel.__new__(ResearchModel)
    nn.Module.__init__(model)
    model.plm = Primary()
    model.secondary, model.prott5 = Frozen(960), Frozen(1024)
    model.fusion = EnsembleResidualFusion(16)
    model.head_q = GatedContrastiveHead(ContrastiveHead(16, 8), 16, 4)
    model.head_k = GatedContrastiveHead(ContrastiveHead(16, 8), 16, 4)
    model.contact_head = ResidueGatedContactHead(ContactHead(16, 16, 2), 16, 4)
    model.contact_head.core.transformer.gradient_checkpointing = True
    model.logit_scale = nn.Parameter(torch.tensor(.3))
    model.register_buffer("valid_keys", torch.tensor([1_000_102, 102_000_001]))
    model.backbone_lora, model.swap_self_negative = True, True
    model._adds_special_tokens = False
    model.fusion_source, model.clip_dim = "esmc_prott5", 8
    model.contact_negative_sampling = copy.deepcopy(POLICY)
    model.multiplier = 1_000_000
    model.contact_hard_neg_ratio = 2
    model.coverage_weight = 0
    with torch.no_grad():
        model.fusion.alpha_esmc.fill_(.15)
        model.fusion.alpha_prott5.fill_(-.1)
        model.contact_head.residue_gate.beta.fill_(.2)
        for head in [model.head_q, model.head_k]:
            head.residue_pool.score[-1].weight.fill_(.1)
    return model.train()


def batches(step=0, count=5):
    generator = torch.Generator().manual_seed(31 + step)
    result = []
    for chunk, start in enumerate(range(0, count, 2)):
        n = min(2, count - start)
        a, b = [torch.randint(1, 30, (n, 6), generator=generator) for _ in range(2)]
        a[:, 2] = 31  # An actual masked token is part of the cache identity.
        ma, mb = torch.ones_like(a), torch.ones_like(b)
        ma[-1, -2:] = 0; a[-1, -2:] = 0
        mb[-1, -1:] = 0; b[-1, -1:] = 0
        ca = torch.arange(start, start + n)
        cb = ca + 100
        target = torch.zeros(n, 6, 6, dtype=torch.bool); target[:, 1, 1] = True
        observed = ma[:, :, None].bool() & mb[:, None, :].bool(); observed[:, 0, 0] = False
        result.append(dict(seq1_tokens=dict(input_ids=a, attention_mask=ma),
                           seq2_tokens=dict(input_ids=b, attention_mask=mb),
                           seq1_cluster=ca, seq2_cluster=cb, seq1_identity=ca, seq2_identity=cb,
                           contact_target=target, contact_target_valid_mask=observed,
                           replay_seed=chunk_seed(37, step, chunk)))
    return result


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(41)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_two_steps_match_losses_gradients_adam_rng_and_negative_sampling(device):
    initial = tiny_model().to(device)
    reference, reused = copy.deepcopy(initial), copy.deepcopy(initial)
    optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
                  for m in [reference, reused]]
    for step in range(2):
        snapshots = []
        for model, optimizer, mode in zip([reference, reused], optimizers, ["off", "cpu"]):
            optimizer.zero_grad(set_to_none=True)
            result = GlobalGradientCache(model, device, frozen_embedding_cache=mode).step(batches(step), 5)
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
            optimizer.step()
            snapshots.append(dict(metrics={k: v for k, v in result.items() if not k.startswith("frozen_cache_")},
                                  gradients=gradients, optimizer=copy.deepcopy(optimizer.state_dict()),
                                  rng=torch.get_rng_state().clone(),
                                  cuda_rng=torch.cuda.get_rng_state() if device == "cuda" else None))
            assert getattr(model, "_frozen_embedding_cache", None) is None
            assert all(p.grad is None for p in model.secondary.parameters())
            assert all(p.grad is None for p in model.prott5.parameters())
        compare(snapshots[0], snapshots[1])
        compare(reference.state_dict(), reused.state_dict())
        assert reused.secondary.calls == reused.prott5.calls == 3 * (step + 1)
        assert reference.secondary.calls == reference.prott5.calls == 6 * (step + 1)
        assert result["frozen_cache_captured_calls"] == result["frozen_cache_reused_calls"] == 6
        for name in ["fusion.alpha_esmc", "fusion.projection_prott5.weight",
                     "contact_head.residue_gate.beta", "plm.lora_projection.weight"]:
            assert snapshots[1]["gradients"][name].norm() > 0


@pytest.mark.parametrize("fault", ["token", "mask", "order", "precision"])
def test_replay_rejects_changed_inputs_or_context(fault):
    encoder = Frozen(4)
    ids = torch.tensor([[1, 31, 0]]); mask = torch.tensor([[1, 1, 0]])
    cache = FrozenEmbeddingCache({"a": encoder, "b": encoder})
    cache.encode("a", encoder, ids, mask)
    cache.start_replay()
    if fault == "token": ids[0, 1] = 3
    if fault == "mask": mask[0, 2] = 1
    with pytest.raises(RuntimeError, match="mismatch"):
        with torch.autocast("cpu", enabled=fault == "precision", dtype=torch.bfloat16):
            cache.encode("b" if fault == "order" else "a", encoder, ids, mask)


def test_cache_cleans_up_after_failure_and_cannot_leak_to_next_step():
    model = tiny_model()
    with pytest.raises(RuntimeError, match="abort"):
        with frozen_embedding_replay(model, "cpu") as cache:
            model.encode_protein(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))
            assert len(cache.entries) == 2
            raise RuntimeError("abort")
    assert model._frozen_embedding_cache is None and cache.entries == []
    with frozen_embedding_replay(model, "cpu") as cache:
        assert cache.entries == []
        cache.start_replay(); assert cache.finish()["frozen_cache_bytes"] == 0


def test_cache_rejects_unfrozen_training_stochastic_or_mutated_encoder():
    encoder = Frozen(4)
    encoder.embed.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        FrozenEmbeddingCache({"a": encoder})
    encoder.requires_grad_(False); encoder.embed.train()
    with pytest.raises(ValueError, match="eval"):
        FrozenEmbeddingCache({"a": encoder})
    encoder.eval()
    cache = FrozenEmbeddingCache({"a": encoder})
    with torch.no_grad(): encoder.embed.weight.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        cache.start_replay()
    class Stochastic(Frozen):
        def forward(self, ids, mask):
            return super().forward(ids, mask) + torch.rand(())
    random_encoder = Stochastic(4)
    cache = FrozenEmbeddingCache({"a": random_encoder})
    with pytest.raises(RuntimeError, match="RNG"):
        cache.encode("a", random_encoder, torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))


def test_cache_is_lossless_and_requires_complete_consumption():
    encoder = Frozen(4).double()
    cache = FrozenEmbeddingCache({"a": encoder})
    ids = torch.ones(1, 2, dtype=torch.long); mask = torch.ones_like(ids)
    original = cache.encode("a", encoder, ids, mask)
    cache.start_replay()
    with pytest.raises(RuntimeError, match="consume"):
        cache.finish()
    replayed = cache.encode("a", encoder, ids, mask.bool())
    assert original.dtype == replayed.dtype == torch.float64
    assert torch.equal(original, replayed)
    cache.finish()
    with pytest.raises(RuntimeError, match="more calls"):
        cache.encode("a", encoder, ids, mask)


def test_checkpoint_between_cached_steps_preserves_optimizer_and_model(tmp_path):
    first = tiny_model()
    optimizer = torch.optim.AdamW([p for p in first.parameters() if p.requires_grad], lr=1e-4)
    GlobalGradientCache(first, "cpu", frozen_embedding_cache="cpu").step(batches(0), 5)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "state.pt"
    torch.save(dict(model=first.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state()), path)
    assert first._frozen_embedding_cache is None
    restored = tiny_model()
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-4)
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state["model"]); restored_optimizer.load_state_dict(state["optimizer"])
    for model, opt in [(first, optimizer), (restored, restored_optimizer)]:
        torch.set_rng_state(state["rng"])
        GlobalGradientCache(model, "cpu", frozen_embedding_cache="cpu").step(batches(1), 5)
        opt.step()
    compare(first.state_dict(), restored.state_dict())
    compare(optimizer.state_dict(), restored_optimizer.state_dict())


def distributed_worker(rank, init_file, output, count):
    torch.set_num_threads(1); torch.manual_seed(41)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=3)
    model = tiny_model()
    metrics = GlobalGradientCache(model, "cpu", frozen_embedding_cache="cpu").step(batches(count=count)[rank::3], count)
    gradients = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    if rank == 0: torch.save(dict(gradients=gradients, metrics=metrics), output)
    dist.destroy_process_group()


@pytest.mark.parametrize("count", [5, 1])
def test_cached_distributed_gradients_match_single_rank_with_empty_ranks(tmp_path, count):
    output = str(tmp_path / "result.pt")
    mp.spawn(distributed_worker, args=(str(tmp_path / "init"), output, count), nprocs=3, join=True)
    actual = torch.load(output, weights_only=True)
    torch.manual_seed(41); model = tiny_model()
    expected = GlobalGradientCache(model, "cpu", frozen_embedding_cache="off").step(batches(count=count), count)
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, actual["gradients"][name], atol=3e-6, rtol=3e-5)
    for key in ["clip_loss", "contact_loss", "negative_selected", "negative_shortfall"]:
        assert actual["metrics"][key] == pytest.approx(expected[key], rel=2e-6, abs=2e-7)
