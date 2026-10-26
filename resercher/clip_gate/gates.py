"""Gate extraction and interventions at fixed encoded residue representations."""
import numpy as np
import torch
from torch.nn import functional as F

from .common import stable_seed


def raw_gate(head, encoded, mask):
    valid = mask.bool()[..., None]
    values = encoded.masked_fill(~valid, 0)
    weights = head.residue_pool.score(head.residue_pool.norm(values)).float().sigmoid()
    weights = weights.masked_fill(~valid, 0)
    if not torch.isfinite(weights).all() or (weights.sum(1) <= 0).any():
        raise FloatingPointError("Invalid or zero gate mass")
    return weights


def project(head, encoded, mask, weights):
    """Match production accumulation, cast and normalization order exactly."""
    values = encoded.masked_fill(~mask.bool()[..., None], 0)
    weights = weights.masked_fill(~mask.bool()[..., None], 0)
    pooled = (values.float() * weights).sum(1) / weights.sum(1).clamp_min(torch.finfo(torch.float32).tiny)
    return F.normalize(head.head(pooled.to(encoded.dtype)), dim=-1)


def permute_gate(weights, mask, sequence_id, branch, seed, replicate):
    if weights.shape[0] != 1:
        raise ValueError("Research extraction uses one full sequence per batch")
    positions = torch.where(mask[0].bool())[0]
    rng = np.random.default_rng(stable_seed(seed, sequence_id, branch, replicate))
    order = torch.as_tensor(rng.permutation(len(positions)), device=weights.device)
    result = weights.clone()
    result[0, positions] = weights[0, positions[order]]
    return result


def extract_branch(head, encoded, mask, sequence_id, branch, seed, shuffles, need_vectors):
    weights = raw_gate(head, encoded, mask)
    raw = weights[0, mask[0].bool(), 0].cpu().numpy().astype(np.float64)
    normalized = raw / raw.sum()
    vectors = {}
    if need_vectors:
        learned = project(head, encoded, mask, weights)
        native = head(encoded, mask)
        torch.testing.assert_close(learned, native, rtol=0, atol=0,
                                   msg="Gate replay must exactly equal the native CLIP head")
        vectors["learned"] = learned[0].float().cpu().numpy()
        vectors["uniform"] = project(head, encoded, mask, torch.ones_like(weights))[0].float().cpu().numpy()
        for replicate in range(shuffles):
            shuffled = permute_gate(weights, mask, sequence_id, branch, seed, replicate)
            vectors[f"shuffle_{replicate:03d}"] = project(head, encoded, mask, shuffled)[0].float().cpu().numpy()
        if any(not np.isfinite(vector).all() for vector in vectors.values()):
            raise FloatingPointError("Nonfinite CLIP vector; do not silently change inference precision")
    return raw, normalized, normalized * len(raw), vectors
