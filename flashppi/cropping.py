"""Training-only, contiguous interface crops; never used to crop evaluation inputs."""
import numpy as np
import torch


def valid_contacts(item):
    lengths = [len(item['seq1']), len(item['seq2'])]
    masks = []
    for side, length in enumerate(lengths, 1):
        values = item.get(f'valid{side}')
        mask = np.ones(length, dtype=bool) if values is None else np.asarray(values, dtype=bool)
        if mask.shape != (length,):
            raise ValueError('Structural validity must match sequence length')
        masks.append(mask)
    contacts = np.asarray(item.get('contacts', []), dtype=np.int64).reshape(-1, 2)
    if len(contacts):
        if (contacts < 0).any() or (contacts[:, 0] >= lengths[0]).any() or (contacts[:, 1] >= lengths[1]).any():
            raise ValueError('Contact coordinates outside sequence bounds')
        contacts = contacts[masks[0][contacts[:, 0]] & masks[1][contacts[:, 1]]]
        contacts = np.unique(contacts, axis=0)
    return contacts, masks


def interface_crop(item, max_residues, candidates=16, near_best_fraction=.9, generator=None):
    """Pick a contact anchor, then jitter two windows containing that anchor.

    Choose uniformly among candidates retaining >=90% of the best candidate's
    contacts. This is a local candidate search, not a globally optimal crop.
    Uniform contact anchors allow different interfaces to appear on repeated
    draws (larger interfaces have proportionally more anchors).
    Uses the CPU torch RNG so checkpointed RNG state reproduces crop choices.
    """
    if max_residues < 1 or candidates < 2 or not 0 < near_best_fraction <= 1:
        raise ValueError('Invalid interface crop settings')
    contacts, masks = valid_contacts(item)
    if not len(contacts):
        raise ValueError('Interface cropping requires at least one valid contact; filter the sampling pool first')
    lengths = [len(item['seq1']), len(item['seq2'])]
    windows = {(0, 0)} if max(lengths) <= max_residues else set()
    if not windows:
        anchor = contacts[int(torch.randint(len(contacts), (1,), generator=generator))]
        bounds = [(max(0, int(pos)-max_residues+1), min(int(pos), max(0, length-max_residues)))
                  for pos, length in zip(anchor, lengths)]
        centered = tuple(max(lo, min(hi, int(pos)-max_residues//2))
                         for pos, (lo, hi) in zip(anchor, bounds))
        windows.add(centered)
        for _ in range(candidates-1):
            windows.add(tuple(int(torch.randint(lo, hi+1, (1,), generator=generator)) for lo, hi in bounds))
        if all(lo == 0 for lo, hi in bounds):
            windows.add((0, 0))
    windows = sorted(windows)
    retained = [((contacts[:, 0] >= a) & (contacts[:, 0] < a+max_residues)
                 & (contacts[:, 1] >= b) & (contacts[:, 1] < b+max_residues)) for a, b in windows]
    counts = np.array([int(mask.sum()) for mask in retained])
    eligible = np.flatnonzero(counts >= near_best_fraction * counts.max())
    selected = int(eligible[int(torch.randint(len(eligible), (1,), generator=generator))]) if len(eligible)>1 else int(eligible[0])
    start1, start2 = windows[selected]
    kept = contacts[retained[selected]] - np.array([start1, start2])
    assert len(kept) > 0
    result = dict(item)
    for side, start in ((1, start1), (2, start2)):
        result[f'seq{side}'] = item[f'seq{side}'][start:start+max_residues]
        result[f'valid{side}'] = masks[side-1][start:start+max_residues].tolist()
        result[f'len{side}'] = len(result[f'seq{side}'])
    result['contacts'] = kept.tolist()
    result['crop_metadata'] = dict(start1=start1, start2=start2, original_contacts=len(contacts),
                                   retained_contacts=len(kept), retained_fraction=len(kept)/len(contacts))
    return result


def valid_contact_counts(table):
    """Count valid annotated contacts without constructing Python contact lists.

    The frozen parquet schema is list<fixed_size_list<int32,2>> and list<bool>.
    Process bounded Arrow batches to keep temporary residue/contact arrays small.
    """
    results = []
    for batch in table.select(['contacts', 'valid1', 'valid2']).to_batches(max_chunksize=1024):
        columns = [batch.column(i) for i in range(3)]
        if any(col.null_count for col in columns):
            raise ValueError('Frozen training data must have non-null contacts and validity masks')
        coords = columns[0].flatten().flatten().to_numpy(zero_copy_only=False).reshape(-1, 2)
        row_ids = np.repeat(np.arange(len(batch)), columns[0].value_lengths().to_numpy())
        good = np.ones(len(coords), dtype=bool)
        for side in range(2):
            lengths = columns[side+1].value_lengths().to_numpy()
            if len(coords) and ((coords[:, side] < 0).any() or (coords[:, side] >= lengths[row_ids]).any()):
                raise ValueError('Contact coordinates outside validity-mask bounds')
            offsets = np.r_[0, np.cumsum(lengths)]
            mask = columns[side+1].flatten().to_numpy(zero_copy_only=False)
            good &= mask[offsets[row_ids] + coords[:, side]]
        results.append(np.bincount(row_ids[good], minlength=len(batch)))
    return np.concatenate(results) if results else np.empty(0, dtype=np.int64)
