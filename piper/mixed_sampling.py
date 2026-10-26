"""Fixed 1:1 PPI/DDI draws, independent of cluster IDs used for loss masking."""
import numpy as np

from .dataset import ClusterSampler


class BalancedKindSampler:
    def __init__(self, c1, c2, kinds, batch_size, seed, block_steps=None):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Exact 1:1 sampling requires an even global batch")
        kinds = np.asarray(kinds)
        if set(kinds) != {"ppi", "ddi"}:
            raise ValueError("Both PPI and DDI must be present, with explicit kind labels")
        self.ppi = np.flatnonzero(kinds == "ppi")
        self.ddi = np.flatnonzero(kinds == "ddi")
        c1, c2 = np.asarray(c1), np.asarray(c2)
        if len(c1) != len(kinds) or len(c2) != len(kinds) or (c1 < 0).any() or (c2 < 0).any():
            raise ValueError("Every sequence needs a nonnegative known-positive-mask cluster")
        self.ppi_sampler = ClusterSampler(c1[self.ppi], c2[self.ppi], ddi_sampling_prob=0, seed=seed)
        self.batch_size, self.seed = batch_size, seed
        if block_steps is not None and (
            isinstance(block_steps, bool) or not isinstance(block_steps, int) or block_steps < 1
        ):
            raise ValueError("Sampling block_steps must be a positive integer")
        self.block_steps = block_steps
        self.num_samples = 0

    def __iter__(self):
        if self.num_samples < 1 or self.num_samples % self.batch_size:
            raise ValueError("Balanced sampling requires a whole number of optimizer batches")
        half = self.batch_size // 2
        steps = self.num_samples // self.batch_size
        block_steps = self.block_steps if self.block_steps is not None else steps
        self.ppi_sampler.num_samples = self.num_samples // 2
        ppi_draws = self.ppi[np.asarray(list(self.ppi_sampler))].reshape(-1, half)
        rng = np.random.default_rng(self.seed + 104729)
        schedule = []
        # DDI draws and within-batch shuffles share an RNG. Fixed blocks preserve
        # completed permutations when extending the total training budget. The
        # first block is exactly the legacy schedule of the same block length.
        # Draw a full DDI block even for a partial final block: shortening this
        # draw would move the RNG before shuffling and change the shared prefix.
        # zip emits only the available PPI batches, so the budget remains exact.
        for start in range(0, steps, block_steps):
            ddi_draws = rng.choice(self.ddi, size=block_steps * half, replace=True).reshape(-1, half)
            for ppi, ddi in zip(ppi_draws[start:start + block_steps], ddi_draws):
                batch = np.r_[ppi, ddi]
                rng.shuffle(batch)
                schedule.extend(batch.tolist())
        return iter(schedule)

    def __len__(self):
        return self.num_samples
