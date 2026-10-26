"""Exact within-step reuse of a fully frozen gLM2 prefix, without weight casts.

The original backbone still handles embedding, masks, positions and unpadding.
During replay only frozen blocks' forward methods are temporarily bypassed;
the cached boundary tensor replaces the last frozen block's output. Trainable
suffix blocks and their activation checkpointing execute normally. Nothing is
cached across optimizer steps or serialized into model checkpoints.
"""
from contextlib import contextmanager

import torch

from .frozen_embedding_cache import _rng, _same_rng


class FrozenPrefixCache:
    def __init__(self, plm):
        self.plm = plm
        layers = list(plm.encoder.layers)
        self.layers = []
        for layer in layers:
            if any(p.requires_grad for p in layer.parameters()):
                break
            self.layers.append(layer)
        if not self.layers or len(self.layers) == len(layers):
            raise ValueError("Prefix reuse requires frozen prefix and trainable suffix blocks")
        suffix_ids = {id(p) for layer in layers[len(self.layers):] for p in layer.parameters()}
        frozen = [p for p in plm.parameters() if id(p) not in suffix_ids]
        if any(p.requires_grad for p in frozen):
            raise ValueError("Prefix reuse requires frozen embedding and prefix parameters")
        self.versions = [(p, p._version) for p in frozen]
        self.versions += [(b, b._version) for layer in self.layers for b in layer.buffers()]
        self.entries, self.originals = [], []
        self.replaying, self.active = False, False
        self.cursor, self.bytes = 0, 0
        self.boundary = None

    def install(self):
        for index, layer in enumerate(self.layers):
            # Restore the exact instance state: an inherited forward must not
            # become a permanently stored bound method after leaving the scope.
            self.originals.append((layer, layer.__dict__.get('forward'), 'forward' in layer.__dict__))
            original = layer.forward

            def forward(x, *args, _index=index, _original=original, **kwargs):
                if not self.active:
                    raise RuntimeError("Frozen prefix called outside a guarded encoding")
                last = _index == len(self.layers) - 1
                if self.replaying:
                    if not last:
                        return x
                    value = self.entries[self.cursor][3]
                    if x.shape != value.shape or x.dtype != value.dtype or x.requires_grad:
                        raise RuntimeError("Frozen prefix boundary shape/dtype/autograd mismatch")
                    return value.to(x.device, non_blocking=value.is_pinned())
                value = _original(x, *args, **kwargs)
                if last:
                    if value.requires_grad or value.ndim not in (2, 3):
                        raise RuntimeError("Frozen prefix boundary must be detached [B,L,D] or [N,D]")
                    if not _same_rng(self.before_rng, _rng(x.device)):
                        raise RuntimeError("Frozen prefix consumed RNG; reuse is unsafe")
                    stored = torch.empty(value.shape, dtype=value.dtype, device='cpu',
                                         pin_memory=value.device.type == 'cuda')
                    stored.copy_(value.detach())
                    self.boundary = stored
                return value

            layer.forward = forward

    def encode(self, ids, attention_mask):
        if self.active or attention_mask is None:
            raise RuntimeError("Prefix cache requires non-nested encoding and explicit masks")
        signature = (ids.device, ids.dtype, torch.is_autocast_enabled(ids.device.type),
                     torch.get_autocast_dtype(ids.device.type))
        tokens, mask = ids.detach().cpu(), attention_mask.detach().cpu().bool()
        if self.replaying:
            if self.cursor >= len(self.entries):
                raise RuntimeError("Frozen prefix replay has extra calls")
            expected, saved_ids, saved_mask, _ = self.entries[self.cursor]
            if signature != expected or not torch.equal(tokens, saved_ids) or not torch.equal(mask, saved_mask):
                raise RuntimeError("Frozen prefix input/order/precision mismatch")
        self.active, self.boundary = True, None
        self.before_rng = _rng(ids.device)
        try:
            value = self.plm(ids, attention_mask)
        finally:
            self.active = False
        if self.replaying:
            self.cursor += 1
        else:
            if self.boundary is None:
                raise RuntimeError("Frozen prefix boundary was not visited")
            self.entries.append((signature, tokens.clone(), mask.clone(), self.boundary))
            self.bytes += self.boundary.numel() * self.boundary.element_size()
            self.boundary = None
        return value

    def check_versions(self):
        if any(t._version != version or (isinstance(t, torch.nn.Parameter) and t.requires_grad)
               for t, version in self.versions):
            raise RuntimeError("Frozen prefix changed within the step")

    def start_replay(self):
        if self.replaying:
            raise RuntimeError("Frozen prefix replay already started")
        self.check_versions()
        self.replaying = True

    def finish(self):
        self.check_versions()
        if not self.replaying or self.cursor != len(self.entries):
            raise RuntimeError("Frozen prefix replay did not consume all entries")
        return dict(frozen_prefix_captured_calls=len(self.entries), frozen_prefix_reused_calls=self.cursor,
                    frozen_prefix_bytes=self.bytes, frozen_prefix_blocks=len(self.layers))

    def close(self):
        for layer, original, existed in reversed(self.originals):
            if existed:
                layer.forward = original
            else:
                del layer.forward
        self.originals.clear()
        self.entries.clear()
        self.boundary = None


@contextmanager
def frozen_prefix_replay(model, storage='off'):
    if storage == 'off':
        yield None
        return
    if storage != 'cpu':
        raise ValueError("frozen_prefix_cache must be 'off' or 'cpu'")
    if getattr(model, '_frozen_prefix_cache', None) is not None:
        raise RuntimeError("Nested frozen prefix caches are unsupported")
    cache = FrozenPrefixCache(model.plm)
    model._frozen_prefix_cache = cache
    try:
        cache.install()
        yield cache
    finally:
        model._frozen_prefix_cache = None
        cache.close()
