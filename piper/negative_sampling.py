"""Unique heteroprotein contact negatives; the CLIP objective is unchanged."""
import torch


POLICY = {
    "policy": "unique_heteroprotein_v1",
    "budget": "legacy_presentations",
    "legacy_self_budget_ratio": 1.0,
    "deduplication": "unordered_full_sequence_pair_per_microbatch",
    "fill": "clip_rank_round_robin",
    "shortfall": "use_all_unique_and_report",
}
STAT_FIELDS = ('negative_budget', 'negative_selected', 'negative_unique_available',
               'negative_hard_seed', 'negative_refill', 'negative_shortfall',
               'negative_self_pairs', 'negative_duplicate_pairs', 'negative_microbatches')


def validate_policy(config):
    policy = config.get("contact_negative_sampling")
    if policy is not None and (
        policy != POLICY or config["contact_self_neg_ratio"] != 0
        or config["contact_random_neg_ratio"] != 0
        or config["contact_hard_neg_ratio"] != 2
        or not config["swap_self_negative"]
    ):
        raise ValueError("Unexpected unique-heteroprotein sampling configuration")
    return policy


@torch.no_grad()
def unique_heteroprotein_indices(clusters1, clusters2, sequence1, sequence2,
                                valid_keys, multiplier, query, key, hard_ratio=2,
                                legacy_self_ratio=1.0):
    """Return oriented indices, with no repeated unordered full-sequence pair.

    Sequence IDs are exact, microbatch-local IDs assigned BEFORE crop and mask.
    The budget equals the old hard+self presentation count, including duplicates.
    Preserve query-local CLIP top-k priority; fill remaining slots round-robin
    down each query's ranking. No RNG is consumed and no new loss is introduced.
    A known-positive mapping in ANY representation blocks that sequence pair.
    """
    n = len(query)
    if any(len(x) != n for x in (clusters1, clusters2, sequence1, sequence2, key)):
        raise ValueError("Negative-mining identities, clusters and embeddings must align")
    a, b = sequence1.tolist(), sequence2.tolist()
    if min(a + b, default=0) < 0 or bool((clusters1 < 0).any() | (clusters2 < 0).any()):
        raise ValueError("Exact sequence IDs and known nonnegative clusters are required")
    device = query.device
    known = torch.isin(clusters1[:, None] * multiplier + clusters2[None, :], valid_keys)
    old_allowed = ~known & ~torch.eye(n, dtype=torch.bool, device=device)
    k = min(n - 1, max(1, int(hard_ratio))) if n > 1 else 0
    hard_budget = int(old_allowed.sum(1).clamp_max(k).sum())
    self_known = torch.isin(clusters1 * multiplier + clusters1, valid_keys)
    self_known |= torch.isin(clusters1, clusters1[clusters1 == clusters2])
    self_budget = int((~self_known[:min(n, int(n * legacy_self_ratio))]).sum())
    budget = hard_budget + self_budget

    def pair(x, y):
        return (min(x, y), max(x, y))

    blocked = {pair(x, y) for x, y in zip(a, b)}
    for i, j in known.nonzero().tolist():
        blocked.add(pair(a[i], b[j]))
    scores = (query @ key.T).float().cpu()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite CLIP mining scores")
    ranked = []
    available = set()
    for i in range(n):
        candidates = [j for j in range(n) if a[i] != b[j] and pair(a[i], b[j]) not in blocked]
        candidates.sort(key=lambda j: (-float(scores[i, j]), j))
        ranked.append(candidates)
        available.update(pair(a[i], b[j]) for j in candidates)
    target = min(budget, len(available))
    chosen, emitted = [], set()

    def add(i, j):
        identity = pair(a[i], b[j])
        if len(chosen) < target and identity not in emitted:
            emitted.add(identity)
            chosen.append((i, j))

    # First consider each query's existing top-k budget, avoiding monopolization
    # by the first query when reverse orientations share an unordered identity.
    for rank in range(k):
        for i in range(n):
            if rank < len(ranked[i]):
                add(i, ranked[i][rank])
    seeds = len(chosen)
    cursor = [k] * n
    while len(chosen) < target:
        before = len(chosen)
        for i in range(n):
            while cursor[i] < len(ranked[i]):
                j = ranked[i][cursor[i]]
                cursor[i] += 1
                if pair(a[i], b[j]) not in emitted:
                    add(i, j)
                    break
        if len(chosen) == before:
            raise AssertionError("Failed to exhaust the verified unique candidate pool")
    assert len(emitted) == len(chosen) and all(a[i] != b[j] for i, j in chosen)
    indices = torch.tensor(chosen, device=device, dtype=torch.long).reshape(-1, 2)
    stats = dict(negative_budget=budget, negative_selected=len(chosen),
                 negative_unique_available=len(available), negative_hard_seed=seeds,
                 negative_refill=len(chosen)-seeds, negative_shortfall=budget-len(chosen),
                 negative_self_pairs=0, negative_duplicate_pairs=0,
                 negative_microbatches=1)
    return indices[:, 0], indices[:, 1], stats
