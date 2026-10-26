"""Overlapping sequence windows before the released contact Transformer.

Windows start on a fixed stride grid. One final partial window covers the tail;
chains shorter than the window have one patch. Padding never adds patches.
These are sequence regions, not annotations of biological motifs.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PatchSpec:
    window: int = 16
    stride: int = 4

    def __post_init__(self):
        if (type(self.window) is not int or type(self.stride) is not int
                or not 1 <= self.stride <= self.window):
            raise ValueError("Require integer 1 <= stride <= window, without uncovered residues")

    def count(self, length):
        if length < 1:
            raise ValueError("A protein must contain at least one residue")
        return 1 + (max(0, length - self.window) + self.stride - 1) // self.stride

    def padding(self, length):
        return (self.count(length) - 1) * self.stride + self.window - length


def patch_mask(residue_mask, spec):
    """[B,L] right-padded residue mask -> [B,P] patch mask."""
    mask = residue_mask.bool()
    if mask.ndim != 2:
        raise ValueError("Expected [B,L] residue mask")
    lengths = mask.sum(-1)
    # Do not infer sequence lengths from structural-coordinate validity masks.
    expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
    if torch.any(lengths == 0) or not torch.equal(mask, expected):
        raise ValueError("Expected nonempty, right-padded sequences")
    counts = 1 + ((lengths - spec.window).clamp_min(0) + spec.stride - 1) // spec.stride
    return torch.arange(spec.count(mask.shape[1]), device=mask.device)[None] < counts[:, None]


def pool_embeddings(embeddings, residue_mask, spec):
    """Mask-aware window mean of contextual PLM embeddings: [B,L,D] -> [B,P,D]."""
    if embeddings.ndim != 3 or embeddings.shape[:2] != residue_mask.shape:
        raise ValueError("Embedding and residue-mask shapes differ")
    valid = patch_mask(residue_mask, spec)
    padding = spec.padding(embeddings.shape[1])
    # FP32 accumulation avoids length-dependent BF16 rounding in window means.
    values = embeddings.float().masked_fill(~residue_mask.bool()[..., None], 0).transpose(1, 2)
    values = F.avg_pool1d(F.pad(values, (0, padding)), spec.window, spec.stride)
    counts = F.avg_pool1d(F.pad(residue_mask.float()[:, None], (0, padding)),
                          spec.window, spec.stride)
    pooled = (values / counts.clamp_min(1 / spec.window)).transpose(1, 2)
    return pooled.to(embeddings.dtype).masked_fill(~valid[..., None], 0), valid


def pool_matrix_max(values, spec, pad_value):
    """[B,L1,L2] -> [B,P1,P2], on exactly the embedding window grid."""
    if values.ndim != 3:
        raise ValueError("Expected [B,L1,L2] matrix")
    padding = (0, spec.padding(values.shape[2]), 0, spec.padding(values.shape[1]))
    return F.max_pool2d(F.pad(values.float()[:, None], padding, value=pad_value),
                        spec.window, spec.stride)[:, 0]


def patch_targets(target, observed, sequence_mask1, sequence_mask2, spec):
    """Existential contact labels; unknown structure is never a negative label.

    A patch is positive if any observed residue pair contacts. A zero patch is
    supervised only when ALL real residue pairs in it have observed coordinates.
    Partial terminal patches are judged on their real residues, not padding.
    Overlapping patch pairs receive equal weight, not independent-sample status.
    """
    shape = (len(sequence_mask1), sequence_mask1.shape[1], sequence_mask2.shape[1])
    if target.shape != shape or observed.shape != shape:
        raise ValueError("Targets, coordinate masks and token masks must align")
    real = sequence_mask1.bool()[:, :, None] & sequence_mask2.bool()[:, None, :]
    positive = pool_matrix_max((target.bool() & observed.bool() & real).float(), spec, 0).bool()
    unknown = pool_matrix_max((real & ~observed.bool()).float(), spec, 0).bool()
    valid = patch_mask(sequence_mask1, spec)[:, :, None] & patch_mask(sequence_mask2, spec)[:, None, :]
    return positive & valid, valid & (positive | ~unknown)


def patch_coverage_targets(target, observed, sequence_mask1, sequence_mask2, spec):
    """Directional participating-residue fractions [B,P1,P2,2].

    Supervision is conditional on observed contact AND complete coordinates for
    every real residue pair in the window. Padding is excluded from denominators.
    Count participating residues, not contact edges; swapping chains swaps channels.
    """
    positive, _ = patch_targets(target, observed, sequence_mask1, sequence_mask2, spec)
    real = sequence_mask1.bool()[:, :, None] & sequence_mask2.bool()[:, None, :]
    contacts = (target.bool() & observed.bool() & real).float()[:, None]
    contacts = F.pad(contacts, (0, spec.padding(target.shape[2]),
                                0, spec.padding(target.shape[1])))
    a = F.max_pool2d(contacts, (1, spec.window), (1, spec.stride))
    a = F.avg_pool2d(a, (spec.window, 1), (spec.stride, 1), divisor_override=1)[:, 0]
    b = F.max_pool2d(contacts, (spec.window, 1), (spec.stride, 1))
    b = F.avg_pool2d(b, (1, spec.window), (1, spec.stride), divisor_override=1)[:, 0]
    lengths = []
    for mask in (sequence_mask1, sequence_mask2):
        count = F.avg_pool1d(F.pad(mask.float()[:, None], (0, spec.padding(mask.shape[1]))),
                            spec.window, spec.stride)[:, 0] * spec.window
        lengths.append(count.clamp_min(1))
    fractions = torch.stack((a / lengths[0][:, :, None], b / lengths[1][:, None, :]), -1)
    unknown = pool_matrix_max((real & ~observed.bool()).float(), spec, 0).bool()
    supervised = positive & ~unknown
    return fractions, supervised


def coverage_loss_sum(logits, targets, supervised):
    """Sum of per-structure means; caller divides by global eligible structures.

    Only the leading structurally labeled positive protein pairs are supervised.
    Mined negative pairs never acquire fabricated geometric labels.
    """
    logits = logits[:len(targets)].float()
    if logits.shape != targets.shape or supervised.shape != targets.shape[:-1]:
        raise ValueError("Coverage prediction/target/mask shapes differ")
    counts = supervised.flatten(1).sum(1)
    batch_index = torch.where(supervised)[0]
    losses = F.binary_cross_entropy_with_logits(logits[supervised], targets[supervised].float(),
                                                reduction="none").mean(-1)
    return (losses / counts[batch_index].clamp_min(1)).sum()


class PatchContactHead(nn.Module):
    """Reuse every released head parameter; pool BEFORE its Transformer."""

    def __init__(self, core, spec, coverage=False, seed=0, gating=None, dim=None):
        super().__init__()
        self.core, self.spec = core, spec
        self.coverage_mix = None
        self.patch_gate = None
        if gating is not None:
            from .gated_modules import PatchResidualGate
            if dim is None:
                raise ValueError('Patch gating requires the residue embedding dimension')
            self.patch_gate = PatchResidualGate(dim, **gating)
        if coverage:
            # Preserve augmentation/dropout RNG and every existing head parameter.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 2909)
                self.coverage_mix = nn.Linear(core.num_heads, 2)
                nn.init.normal_(self.coverage_mix.weight, std=0.01)
                nn.init.zeros_(self.coverage_mix.bias)

    def forward(self, embed1, embed2, mask1, mask2):
        a, ma = pool_embeddings(embed1, mask1, self.spec)
        b, mb = pool_embeddings(embed2, mask2, self.spec)
        if self.patch_gate is not None:
            a, b = self.patch_gate(a, ma), self.patch_gate(b, mb)
        return self.core(a, b, ma, mb)

    def forward_with_coverage(self, embed1, embed2, mask1, mask2):
        if self.coverage_mix is None:
            raise ValueError("Coverage output is disabled")
        a, ma = pool_embeddings(embed1, mask1, self.spec)
        b, mb = pool_embeddings(embed2, mask2, self.spec)
        if self.patch_gate is not None:
            a, b = self.patch_gate(a, ma), self.patch_gate(b, mb)
        logits, valid, features = self.core(a, b, ma, mb, return_pair_features=True)
        return logits, valid, self.coverage_mix(features)


class PatchTargetCollator:
    """Transform positive supervision only; the original model mines negatives."""

    def __init__(self, collator, spec, coverage=False):
        self.collator, self.spec = collator, spec
        self.coverage = coverage

    def __call__(self, rows):
        batch = self.collator(rows)
        if self.coverage:
            batch["coverage_target"], batch["coverage_valid_mask"] = patch_coverage_targets(
                batch["contact_target"], batch["contact_target_valid_mask"],
                batch["seq1_tokens"]["attention_mask"], batch["seq2_tokens"]["attention_mask"], self.spec)
        target, valid = patch_targets(batch["contact_target"], batch["contact_target_valid_mask"],
                batch["seq1_tokens"]["attention_mask"], batch["seq2_tokens"]["attention_mask"], self.spec)
        batch["contact_target"], batch["contact_target_valid_mask"] = target, valid
        return batch


def structural_patch_metrics(logits, row, spec, prediction_is_patch, coverage_logits=None,
                             coverage_small_threshold=0.25):
    """Same-grid comparison of residue-max-pooled and directly predicted patches.

    Metrics are computed per protein pair. K is the number of positive PATCH
    pairs, and therefore must not be compared with residue precision@K.
    """
    from sklearn.metrics import average_precision_score

    if not 0 <= coverage_small_threshold <= 1:
        raise ValueError('Coverage group threshold must be between zero and one')

    n, m = len(row["seq1"]), len(row["seq2"])
    device = logits.device
    va = torch.tensor(row.get("valid1", [True] * n), dtype=torch.bool, device=device)
    vb = torch.tensor(row.get("valid2", [True] * m), dtype=torch.bool, device=device)
    observed = va[None, :, None] & vb[None, None, :]
    target = torch.zeros((1, n, m), dtype=torch.bool, device=device)
    coords = [(int(x), int(y)) for x, y in row.get("contacts", []) if 0 <= x < n and 0 <= y < m]
    if coords:
        x, y = zip(*coords)
        target[0, list(x), list(y)] = True
    ones1 = torch.ones((1, n), dtype=torch.bool, device=device)
    ones2 = torch.ones((1, m), dtype=torch.bool, device=device)
    labels, valid = patch_targets(target, observed, ones1, ones2, spec)
    if prediction_is_patch:
        scores = logits[None, :spec.count(n), :spec.count(m)].float()
    else:
        scores = pool_matrix_max(logits[None, :n, :m], spec, -torch.inf)
    if scores.shape != labels.shape:
        raise ValueError("Prediction/label patch grids differ")
    values, truth = scores[valid], labels[valid]
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite patch predictions at valid supervised positions")
    k = int(truth.sum())
    result = dict(patch_precision_at_k=float(truth[values.topk(k).indices].float().mean()) if k else None,
                patch_average_precision=float(average_precision_score(truth.cpu().numpy(), values.cpu().numpy())) if k else None,
                patch_positive_count=k, patch_valid_count=int(valid.sum()),
                patch_total_count=spec.count(n) * spec.count(m),
                patch_random_ap=k / int(valid.sum()) if valid.any() else None)
    if coverage_logits is not None:
        fractions, coverage_mask = patch_coverage_targets(target, observed, ones1, ones2, spec)
        raw = coverage_logits[None, :spec.count(n), :spec.count(m)].float()
        if raw.shape != fractions.shape or not torch.isfinite(raw[coverage_mask]).all():
            raise ValueError("Invalid coverage evaluation predictions")
        pred = raw.sigmoid()
        error = (pred - fractions).abs()[coverage_mask]
        result.update(coverage_positive_patches=len(error),
                      coverage_mae=float(error.mean()) if len(error) else None,
                      coverage_mae_a=float(error[:, 0].mean()) if len(error) else None,
                      coverage_mae_b=float(error[:, 1].mean()) if len(error) else None)
        small = coverage_mask & (fractions.prod(-1).sqrt() <= coverage_small_threshold)
        result.update(coverage_small_threshold=coverage_small_threshold,
                      coverage_small_positive_patches=int(small.sum()),
                      coverage_small_mae=float((pred - fractions).abs()[small].mean()) if small.any() else None,
                      small_patch_existence_recall_at_half=float((scores.sigmoid()[small] >= 0.5).float().mean()) if small.any() else None)
    return result
