"""Dense-block gradients, preserved head state, and exact checkpoint replay."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from piper.full_finetune import (configure_full_finetune, full_finetune_config,
                                    merge_lora_and_unfreeze, migrate_optimizer,
                                    check_full_block_gradients)
from piper.lora import QVLoRALinear, load_trainable, save_trainable
from piper.model import ContactHead, ContrastiveHead
from piper.ensemble import build_model
from piper.train import build_optimizer, set_step_learning_rate
from piper.distributed import GlobalGradientCache
from piper.checkpoint import compare
from test_frozen_embedding_cache import tiny_model, batches


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.wqkv = nn.Linear(16, 48)
        self.attention.out = nn.Linear(16, 16)
        self.ffn = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        q, k, v = self.attention.wqkv(x).chunk(3, -1)
        return x + self.attention.out(q + k + v) + self.ffn(self.norm(x))


class Primary(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Block() for _ in range(4)])

    def forward(self, ids, mask):
        x = self.embed(ids)
        for layer in self.encoder.layers:
            x = layer(x)
        return x.masked_fill(~mask.bool()[..., None], 0)


def source_config():
    return dict(mode="lora", lora=dict(last_layers=2, rank=2, alpha=4, dropout=.05),
                lora_learning_rate=1e-5, learning_rate=1e-4, weight_decay=.1,
                warmup_steps=5, steps=50, learning_rate_schedule="cosine", seed=41,
                warm_start_checkpoint="/obsolete", warm_start_transition="old",
                batch_size=5, chunk_size=2, max_len=6, clip_loss_weight=.1,
                contact_loss_weight=1., frozen_embedding_cache="cpu")


def source_model():
    model = tiny_model()
    model.plm = Primary().requires_grad_(False)
    for layer in model.plm.encoder.layers[-2:]:
        layer.attention.wqkv = QVLoRALinear(layer.attention.wqkv, rank=2, alpha=4, dropout=.05)
        with torch.no_grad():
            layer.attention.wqkv.lora_b_q.normal_(std=.02)
            layer.attention.wqkv.lora_b_v.normal_(std=.02)
    return model


def step(model, optimizer, config, index):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    set_step_learning_rate(optimizer, config, index)
    metrics = GlobalGradientCache(model, "cpu", frozen_embedding_cache="cpu").step(batches(index), 5)
    optimizer.step()
    return metrics


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(61)


def test_merge_output_and_all_final_block_gradients_with_frozen_encoder_reuse():
    model, config = source_model(), source_config()
    sample = batches()[0]["seq1_tokens"]
    model.eval()
    with torch.no_grad():
        before = model.encode_protein(**sample)
    prefix = {n: p.clone() for n, p in model.plm.named_parameters()
              if n.startswith(("embed.", "encoder.layers.0.", "encoder.layers.1."))}
    merge_lora_and_unfreeze(model, 2)
    with torch.no_grad():
        torch.testing.assert_close(model.encode_protein(**sample), before, atol=3e-6, rtol=3e-6)
    config = full_finetune_config(config)
    optimizer = build_optimizer(model, config)
    metrics = step(model, optimizer, config, 10)
    assert set(check_full_block_gradients(model, 2)) == {"2", "3"}
    assert all(p.grad is not None for p in model.plm.encoder.layers[-1].parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    for name, tensor in prefix.items():
        compare(tensor, dict(model.plm.named_parameters())[name])
    assert not any(isinstance(m, QVLoRALinear) for m in model.modules())
    assert metrics["frozen_cache_captured_calls"] == metrics["frozen_cache_reused_calls"] == 6
    assert metrics["negative_self_pairs"] == metrics["negative_duplicate_pairs"] == 0


def test_migration_preserves_named_head_adam_and_scheduler_but_resets_backbone():
    model, config = source_model(), source_config()
    optimizer = build_optimizer(model, config)
    step(model, optimizer, config, 10)
    old_named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    saved = copy.deepcopy(optimizer.state_dict())
    head_state = {n: copy.deepcopy(optimizer.state[p]) for n, p in old_named.items() if not n.startswith("plm.")}
    merge_lora_and_unfreeze(model, 2)
    target = full_finetune_config(config)
    optimizer = build_optimizer(model, target)
    new_named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    migrate_optimizer(optimizer, saved, old_named, new_named)
    for name, parameter in new_named.items():
        if name.startswith("plm."):
            assert parameter not in optimizer.state
        else:
            compare(head_state[name], optimizer.state[parameter])
    assert optimizer.param_groups[0]["name"] == "backbone"
    for old, new in zip(saved["param_groups"], optimizer.param_groups):
        for key in ["lr", "base_lr", "betas", "eps", "weight_decay"]:
            assert old[key] == new[key]
    step(model, optimizer, target, 11)
    for name, p in new_named.items():
        assert optimizer.state[p]["step"].item() == (1 if name.startswith("plm.") else 2)


def test_full_checkpoint_reload_and_optimizer_replay_are_exact(tmp_path):
    model, config = source_model(), full_finetune_config(source_config())
    merge_lora_and_unfreeze(model, 2)
    # Keep a copy of the pinned base/frozen secondary encoders for rebuilding.
    rebuild = copy.deepcopy(model)
    optimizer = build_optimizer(model, config)
    step(model, optimizer, config, 10)
    save_trainable(model, tmp_path / "weights.safetensors")
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    rng = torch.get_rng_state()
    reference_metrics = step(model, optimizer, config, 11)
    load_trainable(rebuild, tmp_path / "weights.safetensors")
    resumed_optimizer = build_optimizer(rebuild, config)
    resumed_optimizer.load_state_dict(torch.load(tmp_path / "optimizer.pt", weights_only=True))
    torch.set_rng_state(rng)
    resumed_metrics = step(rebuild, resumed_optimizer, config, 11)
    compare(reference_metrics, resumed_metrics)
    compare(model.state_dict(), rebuild.state_dict())
    compare(optimizer.state_dict(), resumed_optimizer.state_dict())


def test_production_builder_enables_backbone_autograd_and_gates(monkeypatch):
    released = SimpleNamespace(plm=Primary(), config=SimpleNamespace(plm_dim=16, clip_embed_dim=8),
                               contact_head=ContactHead(16, 16, 2), head_q=ContrastiveHead(16, 8),
                               head_k=ContrastiveHead(16, 8), logit_scale=nn.Parameter(torch.tensor(.3)))
    monkeypatch.setattr("piper.ensemble.load_released", lambda _: (released, None, {}))
    config = full_finetune_config(source_config())
    config.update(model="unused", swap_self_negative=True, contact_random_neg_ratio=0.,
                  contact_self_neg_ratio=0., contact_hard_neg_ratio=2.,
                  clip_pooling=dict(type="residue_gated", hidden_dim=4),
                  residue_contact_gating=dict(type="residual_channel", hidden_dim=4))
    model, _, _ = build_model(config, torch.empty(0, dtype=torch.long), "cpu")
    model.train()
    assert model.backbone_lora and model.plm.training
    assert model.plm.encoder.gradient_checkpointing
    result = model.encode_protein(torch.ones(2, 5, dtype=torch.long), torch.ones(2, 5))
    result.square().sum().backward()
    check_full_block_gradients(model, 2)
    assert not model.plm.embed.weight.requires_grad
    assert model.contact_head.residue_gate.beta.requires_grad
    assert all(p.requires_grad for p in model.head_q.residue_pool.parameters())


@pytest.mark.parametrize("count", [0, 5, True, 1.5])
def test_invalid_block_selection_is_rejected_before_mutation(count):
    model = source_model()
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    with pytest.raises(ValueError):
        merge_lora_and_unfreeze(model, count)
    assert before == {n: p.requires_grad for n, p in model.named_parameters()}


def test_config_changes_only_backbone_training_and_drops_obsolete_warm_start():
    source = source_config()
    target = full_finetune_config(source)
    assert source["mode"] == "lora" and target["mode"] == "full_last_layers"
    changed = {k for k in source.keys() | target.keys() if source.get(k) != target.get(k)}
    assert changed == {"mode", "lora", "lora_learning_rate", "full_finetune", "backbone_learning_rate",
                       "warm_start_checkpoint", "warm_start_transition"}
    with pytest.raises(ValueError):
        full_finetune_config(target)


def test_optimizer_rejects_corrupt_moments_or_changed_head_names():
    model, config = source_model(), source_config()
    old_optimizer = build_optimizer(model, config)
    step(model, old_optimizer, config, 1)
    old_named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    saved = copy.deepcopy(old_optimizer.state_dict())
    merge_lora_and_unfreeze(model, 2)
    optimizer = build_optimizer(model, full_finetune_config(config))
    new_named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    with pytest.raises(ValueError, match="names changed"):
        migrate_optimizer(optimizer, saved, old_named, {n: p for n, p in new_named.items() if n != "logit_scale"})
    saved["state"][saved["param_groups"][1]["params"][0]]["exp_avg"] = torch.zeros(3, 7)
    with pytest.raises(ValueError, match="shape mismatch"):
        migrate_optimizer(optimizer, saved, old_named, new_named)
