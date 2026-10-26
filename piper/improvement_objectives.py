"""Small, explicit auxiliary objectives for validation-gated research pilots."""
import torch
from torch.nn import functional as F


def listwise_distillation(student, teacher, temperature=2.0):
    """Column zero is the observed positive; skip teachers that rank it lower.

    The gate is a conservative heuristic, not proof that unlabelled pairs are
    true negatives. Teacher targets and the gate must never receive gradients.
    """
    if student.shape != teacher.shape or student.ndim != 2 or student.shape[1] < 2:
        raise ValueError("Expected matching [anchors, positive + candidates] tensors")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    teacher = teacher.detach()
    keep = teacher[:, 0] > teacher[:, 1:].max(1).values
    losses = F.kl_div(F.log_softmax(student / temperature, dim=-1),
                     F.softmax(teacher / temperature, dim=-1), reduction="none").sum(-1) * temperature**2
    loss = (losses * keep).sum() / keep.sum().clamp_min(1)
    return loss, int(keep.sum())


def tail_ranking(student, teacher, top_k=2, margin=0.0, negative_probability=.1):
    """Average the largest eligible pairwise losses in each sampled shortlist.

    This is a sampled top-k surrogate, not an unbiased estimate of proteome-wide
    CVaR or weighted AP. A low teacher score is only an uncertainty heuristic.
    """
    if student.shape != teacher.shape or student.ndim != 2 or top_k < 1:
        raise ValueError("Invalid shortlist tensors or top_k")
    threshold = torch.logit(student.new_tensor(negative_probability))
    allowed = teacher.detach()[:, 1:] < threshold
    losses = F.softplus(margin + student[:, 1:] - student[:, :1])
    chosen = losses.masked_fill(~allowed, -torch.inf).topk(min(top_k, losses.shape[1]), dim=1).values
    valid = torch.isfinite(chosen)
    # torch.where prevents -inf * 0 from producing NaNs in empty shortlists.
    loss = torch.where(valid, chosen, 0.).sum() / valid.sum().clamp_min(1)
    return loss, int(valid.sum())
