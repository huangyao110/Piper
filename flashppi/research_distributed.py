"""Synchronous data parallelism for FlashPPI's two-pass gradient cache.

All ranks compute the same global CLIP objective over detached features. Each
replays only its own fixed global microbatches; parameter gradients are SUMMED
once per optimizer step. There is no DDP gradient averaging to compensate for.
Fixed microbatch partitioning preserves the original contact negative pool and
example-weighted focal objective, including uneven final batches/empty ranks.
"""
from contextlib import nullcontext
import hashlib

import torch
import torch.distributed as dist

from .loss import GradCacheClipLoss, ContactLoss
from .patch_contact import coverage_loss_sum
from .frozen_embedding_cache import frozen_embedding_replay
from .frozen_prefix_cache import frozen_prefix_replay


def distributed():
    return dist.is_available() and dist.is_initialized()


def world_rank():
    return (dist.get_world_size(), dist.get_rank()) if distributed() else (1, 0)


def broadcast_trainable(model):
    """Make initial trainable state identical even if rank setup consumed RNG."""
    if distributed():
        for parameter in model.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.detach(), src=0)


def chunk_seed(seed, step, chunk):
    value = f"{seed}:{step}:{chunk}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def assigned_chunks(indices, microbatch, world_size, rank):
    return [(i // microbatch, indices[i:i + microbatch])
            for i in range(0, len(indices), microbatch)
            if (i // microbatch) % world_size == rank]


def gather_variable(tensor):
    world, rank = world_rank()
    if world == 1:
        return tensor.detach(), 0
    length = torch.tensor([len(tensor)], device=tensor.device, dtype=torch.long)
    counts = [torch.empty_like(length) for _ in range(world)]
    dist.all_gather(counts, length)
    sizes = [int(x.item()) for x in counts]
    padded = tensor.new_zeros((max(sizes), *tensor.shape[1:]))
    padded[:len(tensor)] = tensor.detach()
    outputs = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(outputs, padded)
    return torch.cat([x[:n] for x, n in zip(outputs, sizes)]), sum(sizes[:rank])


def sum_gradients(model, bucket_bytes=32 << 20):
    """One SUM reduction per bounded bucket, including empty-rank gradients."""
    if not distributed():
        return
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for parameter in bucket:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
            offset += count

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        nbytes = parameter.numel() * parameter.element_size()
        if bucket and (size + nbytes > bucket_bytes or bucket[0].dtype != parameter.dtype):
            flush()
            bucket, size = [], 0
        bucket.append(parameter)
        size += nbytes
    flush()


class GlobalGradientCache:
    def __init__(self, model, device, clip_weight=0.1, contact_weight=1.0,
                 frozen_embedding_cache="off", frozen_length_bucket_size=0,
                 frozen_prefix_cache="off"):
        self.model, self.device = model, torch.device(device)
        self.clip_weight, self.contact_weight = clip_weight, contact_weight
        self.clip_loss = GradCacheClipLoss(valid_keys_tensor=model.valid_keys,
                                           multiplier=getattr(model, "multiplier", 1_000_000))
        self.contact_loss = ContactLoss(use_focal_loss=True)
        self.coverage_weight = float(getattr(model, "coverage_weight", 0.0))
        if frozen_embedding_cache not in ("off", "cpu"):
            raise ValueError("frozen_embedding_cache must be 'off' or 'cpu'")
        self.frozen_embedding_cache = frozen_embedding_cache
        if (isinstance(frozen_length_bucket_size, bool) or not isinstance(frozen_length_bucket_size, int)
                or frozen_length_bucket_size < 0):
            raise ValueError('frozen_length_bucket_size must be a nonnegative integer')
        if frozen_length_bucket_size and frozen_embedding_cache != 'cpu':
            raise ValueError('Frozen length bucketing requires CPU embedding reuse')
        if frozen_prefix_cache not in ('off', 'cpu'):
            raise ValueError("frozen_prefix_cache must be 'off' or 'cpu'")
        self.frozen_length_bucket_size = frozen_length_bucket_size
        self.frozen_prefix_cache = frozen_prefix_cache

    def autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda" else nullcontext())

    def arguments(self, batch):
        return [batch["seq1_tokens"]["input_ids"].to(self.device),
                batch["seq2_tokens"]["input_ids"].to(self.device),
                batch["seq1_tokens"]["attention_mask"].to(self.device),
                batch["seq2_tokens"]["attention_mask"].to(self.device),
                batch["seq1_cluster"].to(self.device),
                batch["seq2_cluster"].to(self.device)]

    def step(self, batches, global_count):
        with frozen_embedding_replay(self.model, self.frozen_embedding_cache) as frozen, \
                frozen_prefix_replay(self.model, self.frozen_prefix_cache) as prefix:
            if self.frozen_length_bucket_size:
                inputs = [(torch.cat([b[k]['input_ids'] for k in ('seq1_tokens', 'seq2_tokens')]),
                           torch.cat([b[k]['attention_mask'] for k in ('seq1_tokens', 'seq2_tokens')]))
                          for b in batches]
                with torch.no_grad(), self.autocast():
                    frozen.prepare_buckets(inputs, self.device, self.frozen_length_bucket_size)
            result = self._step(batches, global_count, frozen, prefix)
            if frozen is not None:
                result.update(frozen.finish())
            if prefix is not None:
                result.update(prefix.finish())
            return result

    def _step(self, batches, global_count, frozen, prefix=None):
        world, _ = world_rank()
        cached, rng = [], []
        with torch.no_grad(), self.autocast():
            for batch in batches:
                seed = batch.get("replay_seed")
                if seed is not None:
                    torch.manual_seed(seed)
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed(seed)
                rng.append((torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
                            if self.device.type == "cuda" else None))
                cached.append(self.model.embed_pair(*self.arguments(batch)))
        width = self.model.clip_dim
        local_q = torch.cat([x[0] for x in cached]) if cached else torch.empty(0, width, device=self.device)
        local_k = torch.cat([x[1] for x in cached]) if cached else torch.empty(0, width, device=self.device)
        # Gather features in FP32 so empty ranks have the same dtype as peers.
        all_q, offset = gather_variable(local_q.float())
        all_k, _ = gather_variable(local_k.float())
        all_clusters = []
        for index in (2, 3):
            local = (torch.cat([x[index] for x in cached]) if cached
                     else torch.empty(0, dtype=torch.long, device=self.device))
            all_clusters.append(gather_variable(local)[0])
        all_q.requires_grad_()
        all_k.requires_grad_()
        clip, accuracy = self.clip_loss(all_q, all_k, self.model.logit_scale.exp(),
                                       *all_clusters, compute_top1_acc=True)
        grad_q, grad_k, grad_scale = torch.autograd.grad(
            clip, (all_q, all_k, self.model.logit_scale))
        if frozen is not None:
            frozen.start_replay()
        if prefix is not None:
            prefix.start_replay()
        cursor, local_contact, local_coverage = offset, 0.0, 0.0
        from .negative_sampling import STAT_FIELDS
        sampling_stats = ({k: 0 for k in STAT_FIELDS}
                          if getattr(self.model, 'contact_negative_sampling', None) else {})
        contact_parts = torch.zeros(2, dtype=torch.float64, device=self.device)
        coverage_counts = torch.zeros(2, dtype=torch.long, device=self.device)
        if self.coverage_weight:
            coverage_counts[0] = sum(int(b["coverage_valid_mask"].flatten(1).any(1).sum()) for b in batches)
            coverage_counts[1] = sum(int(b["coverage_valid_mask"].sum()) for b in batches)
            if distributed():
                dist.all_reduce(coverage_counts)
        for batch, saved, states in zip(batches, cached, rng):
            torch.set_rng_state(states[0])
            if states[1] is not None:
                torch.cuda.set_rng_state(states[1], self.device)
            with self.autocast():
                extra = {}
                if getattr(self.model, 'contact_negative_sampling', None):
                    extra = {k: batch[k] for k in ('seq1_identity', 'seq2_identity')}
                outputs = (self.model(*self.arguments(batch), return_coverage=True, **extra) if self.coverage_weight
                           else self.model(*self.arguments(batch), **extra))
                if extra:
                    for key, value in self.model.last_negative_sampling.items():
                        sampling_stats[key] = sampling_stats.get(key, 0) + value
                q, k, logits, _, valid = outputs[:5]
                count = len(q)
                surrogate = ((q.float() * grad_q[cursor:cursor + count]).sum()
                             + (k.float() * grad_k[cursor:cursor + count]).sum()) * self.clip_weight
                targets = batch["contact_target"].to(self.device)
                structure_mask = batch.get("contact_target_valid_mask")
                if structure_mask is not None:
                    structure_mask = structure_mask.to(self.device)
                contact = (self.contact_loss(logits, targets, valid, structure_mask)
                           * len(targets) / global_count) if logits is not None else surrogate * 0
                if extra:
                    # Diagnostics only: same focal formula and denominator as
                    # ContactLoss, without touching its autograd graph or weight.
                    from .loss import focal_loss
                    with torch.no_grad():
                        observed = valid[:len(targets)].clone()
                        if structure_mask is not None:
                            observed &= structure_mask
                        denominator = targets[observed].sum().clamp_min(1)
                        structural = focal_loss(logits[:len(targets)].detach(), targets.float(), observed)
                        negatives = logits[len(targets):].detach()
                        negative = focal_loss(negatives, torch.zeros_like(negatives), valid[len(targets):]) / denominator
                        contact_parts += torch.stack([structural, negative]).double() * len(targets) / global_count
                coverage = surrogate * 0
                if self.coverage_weight:
                    coverage = coverage_loss_sum(outputs[-1], batch["coverage_target"].to(self.device),
                                                batch["coverage_valid_mask"].to(self.device))
                    coverage = coverage / coverage_counts[0].clamp_min(1)
                loss = surrogate + self.contact_weight * contact + self.coverage_weight * coverage
            loss.backward()
            local_contact += float(contact.detach())
            local_coverage += float(coverage.detach())
            cursor += count
        # The global temperature gradient was computed identically on every
        # rank. Divide ONLY that contribution before SUM synchronization.
        contribution = grad_scale.detach() * (self.clip_weight / world)
        if self.model.logit_scale.grad is None:
            self.model.logit_scale.grad = contribution.clone()
        else:
            self.model.logit_scale.grad.add_(contribution)
        sum_gradients(self.model)
        contact_value = torch.tensor([local_contact, local_coverage], dtype=torch.float64, device=self.device)
        if distributed():
            dist.all_reduce(contact_value)
        result = dict(clip_loss=float(clip.detach()), contact_loss=contact_value[0].item(),
                      clip_top1=float(accuracy), clip_entries=len(all_q))
        if sampling_stats:
            names = sorted(sampling_stats)
            counts = torch.tensor([sampling_stats[k] for k in names], dtype=torch.long, device=self.device)
            if distributed():
                dist.all_reduce(counts)
                dist.all_reduce(contact_parts)
            result.update(zip(names, counts.tolist()))
            result.update(contact_structural_contribution=contact_parts[0].item(),
                          contact_negative_contribution=contact_parts[1].item())
        if self.coverage_weight:
            result.update(coverage_loss=contact_value[1].item(), coverage_loss_weight=self.coverage_weight,
                          coverage_structures=int(coverage_counts[0]), coverage_positive_patches=int(coverage_counts[1]))
        return result
