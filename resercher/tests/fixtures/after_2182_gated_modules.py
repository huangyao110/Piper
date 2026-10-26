"""Trainable gates around frozen aligned residue representations."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidueGatedPool(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x, mask):
        # Remove invalid values BEFORE LayerNorm: NaN padding must not poison
        # gradients even when its downstream contribution is masked to zero.
        valid = mask.bool()[..., None]
        values = x.masked_fill(~valid, 0)
        weights = self.score(self.norm(values)).float().sigmoid().masked_fill(~valid, 0)
        numerator = (values.float() * weights).sum(1)
        denominator = weights.sum(1).clamp_min(torch.finfo(torch.float32).tiny)
        return (numerator / denominator).to(x.dtype)


class GatedContrastiveHead(nn.Module):
    def __init__(self, core, dim, hidden_dim=64):
        super().__init__()
        # Keep inherited head_q.head.* / head_k.head.* names unchanged.
        self.head = core.head
        self.residue_pool = ResidueGatedPool(dim, hidden_dim)

    def forward(self, x, mask):
        return F.normalize(self.head(self.residue_pool(x, mask)), dim=-1)


class PatchResidualGate(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        self.projection = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, patches, mask):
        valid = mask.bool()[..., None]
        values = patches.masked_fill(~valid, 0)
        normalized = self.norm(values)
        residual = self.beta * self.gate(normalized).sigmoid() * self.projection(normalized)
        return (values + residual.to(values.dtype)).masked_fill(~valid, 0)


class ResidueGatedContactHead(nn.Module):
    """Apply the existing channel gate per residue, then the original head.

    No pooling or label expansion: [B,L1,D], [B,L2,D] -> [B,L1,L2].
    Keeping ``core`` preserves all pretrained concat-head parameter names.
    """

    def __init__(self, core, dim, hidden_dim=128):
        super().__init__()
        self.core = core
        # Reuse the exact gate formula; only its input resolution changes.
        self.residue_gate = PatchResidualGate(dim, hidden_dim)

    def forward(self, embed1, embed2, mask1, mask2):
        a = self.residue_gate(embed1, mask1)
        b = self.residue_gate(embed2, mask2)
        return self.core(a, b, mask1, mask2)


class EnsembleResidualFusion(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        self.norm_esmc, self.norm_prott5 = nn.LayerNorm(960), nn.LayerNorm(1024)
        self.projection_esmc = nn.Linear(960, output_dim, bias=False)
        self.projection_prott5 = nn.Linear(1024, output_dim, bias=False)
        self.alpha_esmc = nn.Parameter(torch.zeros(()))
        self.alpha_prott5 = nn.Parameter(torch.zeros(()))

    def forward(self, primary, esmc, prott5, mask):
        if (primary.shape[:2] != mask.shape or esmc.shape[:2] != mask.shape
                or prott5.shape[:2] != mask.shape):
            raise ValueError('Encoder residue representations must align with the same mask')
        valid = mask.bool()[..., None]
        a = self.projection_esmc(self.norm_esmc(esmc.masked_fill(~valid, 0)))
        b = self.projection_prott5(self.norm_prott5(prott5.masked_fill(~valid, 0)))
        residual = self.alpha_esmc * a + self.alpha_prott5 * b
        return (primary.masked_fill(~valid, 0) + residual.to(primary.dtype)).masked_fill(~valid, 0)
