# Interface-aware training crops

The frozen split and source database are read-only. Crops are created in memory
by `flashppi/cropping.py`; validation and test continue to use full sequences.
The prediction heads and interaction scoring rule are unchanged.

## Fixed policy

`configs_train/finetune_lora_bs1024_micro16_interface_v2.json` and its
`head_control` counterpart share the following policy:

- At most 512 consecutive residues per chain; short chains retain their full sequence.
- Choose one uniformly random **valid contact pair** as an anchor. Larger interfaces
  therefore have more chances of providing an anchor; interfaces are not sampled uniformly.
- Generate 16 anchor-containing candidates: one centered window pair and 15 random
  window pairs. Include the prefix reference if it also contains the anchor. Deduplicate
  identical window pairs.
- Choose randomly among candidates with at least 90% of the best candidate's contact
  count. This is **not** a guarantee of retaining 90% of the full structure's contacts,
  nor a globally optimal search. Different draws can cover different interface regions.
- Shift contact coordinates by the two crop offsets and slice structural-validity masks
  with exactly the same offsets. Missing coordinates do not hide sequence tokens.
- Draw crop randomness from the CPU Torch RNG, whose state is saved with training checkpoints.

## Records without contact supervision

Scan all training records for at least one contact with valid coordinates at both
endpoints. Exclude zero-contact records from this joint CLIP/contact training pool,
while keeping **all original known-positive cluster pairs** in negative exclusion.
Do not reinterpret excluded interactions as negative labels. The frozen parquet is unchanged.

The pool changes from 1,120,675 to 1,099,957 rows (20,718 excluded: 20,358 Pinder,
360 CongLab). Cluster-balanced sampling remains with replacement. Two epochs now mean
2,199,914 positive draws, 2,150 optimizer steps, and a final batch of 181 per epoch.
Microbatch is 16; ordinary effective batches remain 1,024.

## Verification and reproduction

Run these commands in the `esmfold2` environment from the repository root:

```bash
python tests/test_interface_crop.py
python tests/test_finetune_training.py
python tests/test_lora.py
python tools/audit_interface_cropping.py --output runs/NEW_CROP_AUDIT
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m flashppi.finetune \
  --config configs_train/finetune_lora_bs1024_micro16_interface_v2.json \
  --smoke --output runs/NEW_CROP_SMOKE
```

The fixed smoke run uses 8 full optimizer steps and saves its resume probe at step 7.
Resume it with the same config, `--smoke --resume runs/NEW_CROP_SMOKE/resume_probe`,
and a fresh output directory. `tools/check_finetune_replay.py` compares uninterrupted
and resumed weights, optimizer and RNG state exactly.

For full-length validation use the same config with `--evaluate-only --resume .../last`
and a fresh output directory. The evaluator does not call the training cropper.

The 64,512-draw audit uses the original sampling pool as a common reference. Among
8,004 long-chain draws with valid contacts, mean contact retention changes from
64.98% to 89.47%; median from 82.57% to 98.55%. Zero retained contacts change from
988 to 0; retention below 50% changes from 2,694 to 450. The remaining low-retention
windows and the 1,228 draws with lower retention than the prefix must not be hidden:
random anchor selection trades some contact count for interface coverage, and 512
residues cannot represent every possible long interface in one crop. These figures
measure crop quality, not prediction accuracy or a proven improvement on test.

Audit outputs: `runs/interface_crop_audit_v2/population.json` and `comparison.json`.
The formal comparison runner accepts `--microbatch 16 --variant interface_v2` and
requires the crop audit, GPU smoke/resume checks and full-length validation to pass.
It is **not launched** as part of crop verification.
