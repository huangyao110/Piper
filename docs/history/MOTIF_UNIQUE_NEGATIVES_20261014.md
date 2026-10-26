# Motif contact-negative continuation

This experiment changes contact-negative sampling only. It retains window 16,
stride 4, gLM2 last-16-layer LoRA, global batch 1024, microbatch 16, 15% sequence
masking, PPI/DDI 1:1 cluster sampling, the existing Focal/CLIP/coverage losses and
weights, validation every 50 steps, and the original 1616-step total budget.

## Sampling contract

- Assign exact local protein identities from complete sequences after chain swap
  and before interface crop or token masking. Cluster IDs remain the existing
  known-interaction filter; they are not substituted for exact sequence identity.
- Exclude identical-sequence pairs from both hard mining and the self branch.
  Observed positive homomers retain their original labels and supervision.
- Exclude all observed positive pairs and known cluster interactions. If one
  sequence pair has multiple representations, a known-positive representation
  blocks every representation of that pair in the microbatch.
- Emit each unordered full-sequence pair at most once per contact microbatch.
  The selected orientation retains its own CLIP score. Sampling across later
  microbatches/steps remains with replacement, as before.
- Preserve the old hard-plus-self **presentation budget**, including slots that
  previously duplicated the same pair. Keep hard top-2 priority per query, then
  fill round-robin down the existing CLIP rankings using unused heteroprotein
  pairs. No concat reranking, random-negative branch or new RNG draw is added.
- If the finite microbatch contains too few eligible unique pairs, use all of
  them and report `negative_shortfall`. Never invent a negative label, reinsert
  self pairs, duplicate a pair or silently change loss normalization to fill it.

The CLI configuration sets `contact_self_neg_ratio=0`; the separate
`legacy_self_budget_ratio=1` only computes how many slots to replace.

## Continuation and verification

`tools/prepare_unique_negative_continuation.py` runs GPU preflight against a
saved checkpoint while the old trainer continues. It verifies unchanged
parameter names/order and tensors, Adam moments, inference outputs, frozen
backbone, gradients for all trainable branches, exact optimizer replay, and a
full actual 1024-positive optimizer batch from the unchanged sampling schedule.
All preflight updates are disposable.

After preflight passes, stop the old supervisor gracefully so its child finishes
the current optimizer step and saves. Run the same preparation tool with
`--migrate-only --preflight-checks <checks.json>` on that latest checkpoint.
The migration copies trainable weights unchanged and preserves optimizer, RNG,
step, data hashes and previous best pointers. It explicitly records the two
changed configuration fields and updated source hashes in a new run lineage.
The original checkpoint and data remain untouched.

Launch using `tools/run_motif_training.py --resume <migrated checkpoint>
--negative-preflight <continuation_checks.json> --config <v3 config> --output
<new experiment>`. The launcher verifies configuration, checkpoint and source
hashes before strict training-state restoration. Final E. coli evaluation remains
enabled at the original final step.

## Monitoring

History records budget, selected unique pairs, refill count, available pairs,
shortfall, self/duplicate counts, and structural versus synthetic-negative Focal
contributions. These diagnostic contributions sum to the existing contact loss;
they do not change its denominator or gradients. Counts do not imply equal loss
or gradient contribution. New pairs remain presumed negatives, not experimentally
confirmed noninteractors.

Regression command:

```bash
python -m pytest -q tests/test_unique_contact_negatives.py tests/test_patch_contact.py tests/test_patch_coverage.py tests/test_interface_crop.py tests/test_sequence_masking.py tests/test_research_multigpu.py
```
