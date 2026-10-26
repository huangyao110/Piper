"""Training-only residue masking, applied once before gradient-cache replay."""
import math

import torch


class ResidueMasker:
    """Replace canonical amino-acid tokens without changing [B, L] alignment."""

    def __init__(self, tokenizer, probability=0.15):
        self.probability = float(probability)
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Sequence mask probability must be finite and between 0 and 1")
        self.mask_id = getattr(tokenizer, "mask_token_id", None)
        self.residue_ids = []
        if self.probability:
            vocab = tokenizer.get_vocab()
            self.residue_ids = [vocab[a] for a in "ACDEFGHIKLMNPQRSTVWY" if a in vocab]
            if not self.residue_ids:
                raise ValueError("Sequence masking requires single-residue amino-acid tokens")
            forbidden = self.residue_ids + [getattr(tokenizer, k, None) for k in
                         ("pad_token_id", "unk_token_id", "cls_token_id", "eos_token_id", "sep_token_id")]
            if self.mask_id is None or self.mask_id in forbidden or self.mask_id not in vocab.values():
                raise ValueError("Sequence masking requires a distinct native mask token")

    def __call__(self, tokens, generator=None):
        if not self.probability:
            return tokens, (0, 0)
        ids, attention = tokens["input_ids"], tokens["attention_mask"]
        if ids.ndim != 2 or attention.shape != ids.shape:
            raise ValueError("Expected aligned input_ids and attention_mask with shape [B, L]")
        residues = torch.tensor(self.residue_ids, dtype=ids.dtype, device=ids.device)
        eligible = attention.bool() & torch.isin(ids, residues)
        selected = eligible & (torch.rand(ids.shape, device=ids.device, generator=generator)
                               < self.probability)
        # Keep one known residue on very short chains or rare all-masked draws.
        # Do not condition the mask on contact/distance labels.
        full = (eligible.sum(1) > 0) & (selected.sum(1) == eligible.sum(1))
        for row in full.nonzero(as_tuple=False).flatten().tolist():
            positions = eligible[row].nonzero(as_tuple=False).flatten()
            keep = torch.randint(len(positions), (), device=ids.device, generator=generator)
            selected[row, positions[keep]] = False
        output = dict(tokens)
        output["input_ids"] = ids.masked_fill(selected, self.mask_id)
        # A masked residue remains visible to attention and supervised by its
        # original structural target. Padding and unknown structure stay separate.
        return output, (int(selected.sum()), int(eligible.sum()))


class SequenceMaskingCollator:
    """Wrap a training collator after its swap, crop and target construction.

    The stored token tensors are reused unchanged by both gradient-cache passes.
    Evaluation must use its original unmasked tokenizer/collator.
    """

    def __init__(self, collator, probability=0.15):
        self.collator = collator
        self.masker = ResidueMasker(collator.tokenizer, probability)

    def __call__(self, rows):
        batch = self.collator(rows)
        if not self.masker.probability:
            return batch
        counts = torch.zeros(2, dtype=torch.long)
        for side in ("seq1_tokens", "seq2_tokens"):
            batch[side], current = self.masker(batch[side])
            counts += torch.tensor(current, dtype=torch.long)
        batch["sequence_masking_counts"] = counts  # masked, eligible; both chains
        return batch
