"""Lossless, CPU-backed reuse of frozen encoders within one gradient-cache step.

Only raw residue embeddings are stored. Trainable fusion/projections/gates must
run again during replay so their gradients and gLM2 LoRA gradients are retained.
The cache is deliberately positional, not keyed by a protein ID: crops, masks,
padding, dtype and encoder identity must match the corresponding capture call.
"""
from contextlib import contextmanager

import torch


def _rng(device):
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _same_rng(before, after):
    return all(a is None and b is None or a is not None and b is not None and torch.equal(a, b)
               for a, b in zip(before, after))


class FrozenEmbeddingCache:
    def __init__(self, modules):
        if not modules:
            raise ValueError("Frozen embedding reuse requires frozen encoder branches")
        self.modules = modules
        self.versions = {}
        for name, module in modules.items():
            if any(m.training for m in module.modules()) or any(p.requires_grad for p in module.parameters()):
                raise ValueError("Cached encoder must be frozen and in eval mode: " + name)
            self.versions[name] = [(tensor, tensor._version) for tensor in
                                   [*module.parameters(), *module.buffers()]]
        self.entries = []
        self.replaying = False
        self.cursor = 0
        self.bytes = 0
        self.prepared = False
        self.capture_cursor = 0
        self.bucket_positions = 0

    def prepare_buckets(self, inputs, device, bucket_size):
        """Encode cropped/masked proteins by length, then restore call order.

        Only frozen encoder calls are regrouped. Each input is the exact
        concatenation of chain A and B in an existing training microbatch.
        Heads, negative mining and gradient accumulation retain that order.
        Padding is restored to [B,L,D], including zero-valued padded residues.
        Changing GEMM shapes can change floating point rounding; this option
        requires a separate numerical/negative-mining acceptance experiment.
        """
        if self.entries or self.replaying or self.prepared:
            raise RuntimeError("Buckets must be prepared before capture")
        if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size < 1:
            raise ValueError("Expected positive frozen encoder bucket size")
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        rows = []
        normalized = []
        for call, (ids, mask) in enumerate(inputs):
            ids, mask = ids.detach().cpu().clone(), mask.detach().cpu().bool().clone()
            lengths = mask.sum(1)
            expected = torch.arange(mask.shape[1])[None, :] < lengths[:, None]
            if not torch.equal(mask, expected) or bool((lengths == 0).any()):
                raise ValueError("Buckets require nonempty right-padded sequences")
            normalized.append((ids, mask))
            rows.extend((int(length), call, row) for row, length in enumerate(lengths))
        rows.sort(key=lambda item: item[0])  # Stable sorting, no RNG consumed.
        values = {}
        for name, module in self.modules.items():
            stored = None
            for start in range(0, len(rows), bucket_size):
                selected = rows[start:start + bucket_size]
                width = selected[-1][0]
                ids = torch.zeros(len(selected), width, dtype=normalized[0][0].dtype)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for i, (length, call, row) in enumerate(selected):
                    ids[i, :length] = normalized[call][0][row, :length]
                    mask[i, :length] = True
                with torch.no_grad():
                    before = _rng(device)
                    value = module(ids.to(device), mask.to(device))
                    if not _same_rng(before, _rng(device)):
                        raise RuntimeError("Frozen encoder consumed RNG during bucketing: " + name)
                if value.requires_grad or value.ndim != 3 or value.shape[:2] != ids.shape:
                    raise RuntimeError("Expected detached [B,L,D] frozen residue embeddings")
                if stored is None:
                    stored = [torch.zeros((*x.shape, value.shape[-1]), dtype=value.dtype,
                                          pin_memory=device.type == "cuda") for x, _ in normalized]
                cpu = value.detach().cpu()
                for i, (length, call, row) in enumerate(selected):
                    stored[call][row, :length].copy_(cpu[i, :length])
                self.bucket_positions += len(selected) * width
            values[name] = stored or []
        for call, (ids, mask) in enumerate(normalized):
            for name in self.modules:
                signature = (name, device, ids.dtype, torch.is_autocast_enabled(device.type),
                             torch.get_autocast_dtype(device.type))
                value = values[name][call]
                self.entries.append((signature, ids, mask, value))
                self.bytes += value.numel() * value.element_size()
        self.prepared = True

    def start_replay(self):
        if self.replaying:
            raise RuntimeError("Frozen cache replay already started")
        if self.prepared and self.capture_cursor != len(self.entries):
            raise RuntimeError("Frozen bucket capture did not consume every prepared embedding")
        for name, tensors in self.versions.items():
            if any(tensor._version != version for tensor, version in tensors):
                raise RuntimeError("Frozen encoder changed during capture: " + name)
        self.replaying = True

    def encode(self, name, module, input_ids, attention_mask):
        if self.modules.get(name) is not module:
            raise RuntimeError("Frozen cache encoder identity changed")
        # The regular forward normalizes integer masks to bool, embed_pair does
        # not. Compare their logical values while preserving the token IDs.
        mask = attention_mask.bool()
        signature = (name, input_ids.device, input_ids.dtype,
                     torch.is_autocast_enabled(input_ids.device.type),
                     torch.get_autocast_dtype(input_ids.device.type))
        if not self.replaying and not self.prepared:
            before = _rng(input_ids.device)
            value = module(input_ids, attention_mask)
            if not _same_rng(before, _rng(input_ids.device)):
                raise RuntimeError("Frozen encoder consumed RNG; reuse would change replay: " + name)
            if value.requires_grad or value.ndim != 3 or value.shape[:2] != input_ids.shape:
                raise RuntimeError("Expected detached [B,L,D] frozen residue embeddings")
            # Blocking D2H makes completion explicit; pinned memory permits the
            # later H2D copy to be asynchronous. Never cast the cached values.
            stored = torch.empty(value.shape, dtype=value.dtype, device="cpu",
                                 pin_memory=value.device.type == "cuda")
            stored.copy_(value.detach())
            self.entries.append((signature, input_ids.detach().cpu().clone(),
                                 mask.detach().cpu().clone(), stored))
            self.bytes += stored.numel() * stored.element_size()
            return value
        cursor = self.cursor if self.replaying else self.capture_cursor
        if cursor >= len(self.entries):
            raise RuntimeError("Frozen replay has more calls than capture")
        expected, tokens, saved_mask, value = self.entries[cursor]
        if (signature != expected or not torch.equal(tokens, input_ids.detach().cpu())
                or not torch.equal(saved_mask, mask.detach().cpu())):
            raise RuntimeError("Frozen replay input/order/precision mismatch")
        if self.replaying:
            self.cursor += 1
        else:
            self.capture_cursor += 1
        return value.to(input_ids.device, non_blocking=value.is_pinned())

    def finish(self):
        if not self.replaying or self.cursor != len(self.entries):
            raise RuntimeError("Frozen replay did not consume every captured embedding")
        result = dict(frozen_cache_captured_calls=len(self.entries),
                      frozen_cache_reused_calls=self.cursor, frozen_cache_bytes=self.bytes)
        if self.prepared:
            result['frozen_cache_bucket_positions'] = self.bucket_positions
        return result


@contextmanager
def frozen_embedding_replay(model, storage="off"):
    if storage == "off":
        yield None
        return
    if storage != "cpu":
        raise ValueError("frozen_embedding_cache must be 'off' or 'cpu'")
    if getattr(model, "_frozen_embedding_cache", None) is not None:
        raise RuntimeError("Nested frozen embedding caches are not supported")
    if not hasattr(model, "frozen_encoder_modules"):
        raise ValueError("Model does not expose frozen encoder branches")
    cache = FrozenEmbeddingCache(model.frozen_encoder_modules())
    model._frozen_embedding_cache = cache
    try:
        yield cache
    finally:
        model._frozen_embedding_cache = None
        cache.entries.clear()
