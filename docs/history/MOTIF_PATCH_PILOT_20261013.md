# GLM2 overlapping-patch contact experiment

## Question and scope

Does replacing residue tokens in the contact head with overlapping sequence
windows improve CLIP retrieval and regional interface prediction?

The user confirmed use of the **previous filtered user PPI training set**.
This is not a reproduction on the authors' official training corpus: the
configured public training dataset endpoints were inaccessible (HTTP 401).
The stopped ESMC/DDI experiment is not resumed.

## Fixed comparison

| Setting | Residue control | Patch arm |
|---|---|---|
| Initialization | Official `flashppi_noecoli`, fresh zero-output LoRA | Identical |
| Contact input | GLM2 residue embeddings | Masked mean, window 16 / stride 4 |
| Contact Transformer and projections | Released weights, trainable | Same weights, trainable |
| Contact supervision | Residue binary contact labels | Existential patch contacts |
| Protein interaction score | Sigmoid of maximum valid contact logit | Same readout over patches |
| CLIP head | Original head | Identical architecture |
| Backbone adaptation | Last 8 GLM2 layers, Q/V LoRA rank 8, alpha 16, dropout 0.05 | Same |

Both arms use seed 20261010, 100 optimizer steps, effective batch 1024,
microbatch 16, learning rate 1e-5, five warmup steps, weight decay 0.01,
gradient clipping 1, BF16 training and FP32 global CLIP logits.
Objectives remain `0.1 * CLIP + 1.0 * Focal`, with Focal alpha 0.25/gamma 2
and the existing positive-count normalization. The loss magnitudes at different
resolutions are not themselves measures of comparative quality.
Existing contact-negative mining stays hard=2, random=0, self=1, with swapped
CLIP entries and known-positive masking. Hard-negative identities may diverge
as the two models learn; the mining algorithm and candidate microbatches match.

Each eligible residue is independently masked with probability 0.15 during
training, using the existing native-mask implementation. Labels and sequence
lengths are unchanged. There is no masking during validation.

## Data, isolation and sampling

- Training: `data/ppi_finetune_split_v1/train.parquet`; 1,120,675 records,
  1,099,957 containing at least one valid contact. These are structural records,
  not 1.1 million distinct sequence pairs. Sample with replacement using the
  existing sequence-cluster sampler; 102,400 positive draws per arm.
- Fixed structural validation: all 500 pairs in
  `data/ppi_finetune_split_v1/validation.parquet`.
- Fixed interaction validation: 81 positives and 4,119 negatives in
  `data/ppi_finetune_validation_v1/interaction.parquet`.
- Retain the existing E. coli/val homology policy: either chain excludes a
  training pair when identity >30% and both sequence coverages >50%.
  Verify frozen file hashes before training, rather than rebuild the split.
- This filtering concerns fine-tuning exposure. It does not prove the official
  pretrained checkpoint never saw validation-related structures. The prior
  RF2-PPI overlap caveat remains; no new leakage-free RF2 claim is made.
- Same seeded draws, swap augmentation, 512-residue interface crops and mask
  draws in both arms; save the sample-schedule checksum to verify this.

No raw database, validation set or official test is modified. Test metrics
are not used to choose between arms in this pilot.

## Exact patch semantics

Pool **before** the contact Transformer, not after constructing a full residue
contact map. CLIP still pools the original GLM2 residue embeddings.
For length L, use `1 + ceil(max(L-16, 0)/4)` windows. Starts are 0, 4, 8, ...;
the last window may be partial. Short chains get one patch. All residues,
including the tail, are covered. A 512-residue chain yields 125 patches.

Mean pooling ignores padding, accumulates in FP32 and retains gradient flow
into GLM2 LoRA. Original segment embeddings and compact valid-token RoPE are
then applied by the released head. Patch positions represent the window grid;
this pilot does not add center-residue position encoding or extra parameters.

A patch pair is positive if **any observed residue pair** contacts. A zero is
supervised only if every real residue pair in that patch has coordinates.
Unknown structure never becomes a negative label; artificial padding is
excluded. An observed positive remains valid even with other missing residues.
Mined negative protein pairs still have all-zero targets on their valid patches,
with the same presumed-negative assumption as the existing training code.

Patch pairs are equally weighted. Overlap deliberately repeats evidence: an
interior contact can appear in 16 patch pairs. These patches are not independent
samples or established biological motifs, and residue localization is coarser.

## Evaluation and selection

Validate at steps 0, 50 and 100, with full unmasked sequences. Report CLIP AP,
protein contact-score AP, patch AP and patch precision@K. For the residue
control, max-pool logits onto the **identical** window grid and use the same
patch labels/validity masks. K counts positive patch pairs. Patch precision
must not be compared directly with the old residue precision values.

Ground-truth audit: all 500 validation structures have evaluable positive
patches. Mean positive prevalence rises from 2.06% at residue resolution to
18.69% at patch resolution; 10 structures have no valid negative patches.
These 10 structures yield AP/P@K=1 for both arms and provide no ranking evidence;
the paired difference comes from the remaining 490 structures, with dilution
from averaging across the full 500-pair cohort.
The unknown-coordinate policy excludes 186,059 patch pairs from supervision.

Save per-structure predictions. Final comparison includes a paired bootstrap
over structures (2,000 replicates, seed 20261013), not over correlated patches.
This interval does not estimate training-seed uncertainty. One seed/100 steps
is a feasibility pilot, not a convergence claim. The two-arm comparison changes
both head resolution and supervision resolution; it does not isolate them.

Separate `best_clip.json`, `best_contact.json` (patch AP), and
`best_interaction.json` records are maintained, including the step-0 baseline.
Checkpoints include trainable weights, optimizer, RNG, source and input hashes.
Save recovery checkpoints every 25 steps and `final.json` at step 100.
SIGTERM/SIGINT saves after the current optimizer step.

## Validation and execution

CPU checks: 33 tests and 4 subtests passed across patch handling, gradient-cache
replay, sequence masking, cropping and LoRA. Real GPU checks passed on an RTX
5090 with PyTorch 2.10.0+cu128: identical initial trainables/CLIP outputs,
unchanged frozen weights, LoRA updates, window=1 identity, finite forward and
backward, validation, and exact optimizer/parameter continuation after reload.
With two full-length microbatches (32 pairs), peak training allocations were
12.31 GiB (residue) and 6.66 GiB (patch). This is a microbatch pressure check,
not the measured throughput of a full 1024-pair optimizer step.

```bash
PY=artifacts/environments/flashppi_torch210_py311/bin/python
$PY tools/check_patch_experiment.py --output experiments/motif_pilot_v1_20261013/preflight
$PY tools/run_motif_experiment.py --output experiments/motif_pilot_v1_20261013
```

The preflight refuses an existing output directory; use a fresh experiment
directory when reproducing it. The launcher requires matching preflight config
hashes, runs arms sequentially, stops on failure, and writes `comparison.json`
and `REPORT.md` after both finish. Configurations:
`configs_train/motif_residue_pilot_v1.json` and
`configs_train/motif_patch16_s4_pilot_v1.json`.
