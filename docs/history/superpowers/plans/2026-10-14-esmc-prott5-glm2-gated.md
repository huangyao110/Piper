# ESMC–ProtT5–gLM2 Gated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the approved gated three-encoder experiment and automatically start it only after the existing training and its final evaluation succeed.

**Architecture:** Work in `/home/zlab/.codex/worktrees/5f44/FlashPPI`, seeded with the current research sources; the original checkout and its running trainer remain unchanged. Frozen ESMC and ProtT5 produce aligned online residue embeddings; projected residual fusion feeds independently gated CLIP pooling and 3/2 patch contact prediction. A pinned supervisor waits for the predecessor, runs disposable verification, then trains from inherited final weights with a new optimizer and schedule.

**Tech Stack:** Python/PyTorch, existing FlashPPI gradient-cache trainer, Hugging Face T5 encoder, ESMC, pytest, POSIX subprocess/flock.

## Global Constraints

- Specification: `/media/zlab/lexar4t/FlashPPI/esmc-protT5-glm2-gated.md`.
- All implementation edits and commits occur in `/home/zlab/.codex/worktrees/5f44/FlashPPI`; do not modify existing files in the original checkout. Root documentation and preparation results may be updated there.
- Current predecessor: `/media/zlab/lexar4t/FlashPPI/experiments/conglab_ecoli_motif_unique_negatives_v3_20261014`; final step 1616; successful final E. coli evaluation is required before handoff.
- New run: `/media/zlab/lexar4t/FlashPPI/experiments/esmc_prott5_glm2_gated_v1_20261014`; its preparation subdirectory already exists. Do not overwrite existing run outputs.
- Frozen ESMC-300M and ProtT5-XL-UniRef50 encoder-only; gLM2 retains last-16-layer Q/V LoRA, rank 8, alpha 16, dropout 0.05.
- Same cropped sequences and same 15% masked residue positions across all encoders. ProtT5 maps each masked residue to one `<extra_id_0>` token; no span contraction. U/Z/O/B map to X only in the ProtT5 branch.
- Fused dimension 1280; ESMC dimension 960, ProtT5 dimension 1024. Frozen encoders stay eval/no_grad; fusion projections and gates remain trainable.
- CLIP uses scalar sigmoid residue gates normalized over valid residues; gate score output initialized to zero. Preserve existing `head_q.head.*` / `head_k.head.*` weight names.
- Contact mean pool window=3, stride=2, retains all patches; applies per-channel residual gate before the existing two-layer concat core. New residual scale starts at zero and residual projection is nonzero.
- New phase: fresh Adam, step 0, 1616 steps/2 sampling epochs, batch 1024, microbatch 16, length 512, warmup 81, cosine, LoRA LR 1e-5, head LR 1e-4, loss weights 0.1/1.0/0.1, validation 50, recovery save 25.
- Preserve unique heteroprotein negative sampling and PPI/DDI 1:1. No silent fallback or automatic microbatch change after failed verification.
- Coverage minimum group for new run is sqrt(fA*fB)<=1/3+1e-7; legacy default remains 0.25. Record threshold in metrics; never compare patch AP across grids as equivalent.
- Pinned pretrained files, source, input data and effective config must be checked. Preflight updates never enter actual training. Old test scores do not select the initialization checkpoint.
- Use `/media/zlab/lexar4t/FlashPPI/artifacts/environments/esmc_prott5_gated_py312/bin/python` once prepared; existing CPU tests can initially use `/media/zlab/lexar4t/FlashPPI/artifacts/environments/flashppi_torch210_py311/bin/python`.

---

### Task 1: Gated three-encoder model and existing trainer integration

**Files:** Create `flashppi/gated_modules.py`, `flashppi/frozen_prott5.py`, `tests/test_gated_ensemble.py`; modify isolated copies of `flashppi/research_model.py`, `flashppi/research_train.py`, `flashppi/patch_contact.py`, `flashppi/research_eval.py`; add `configs_train/esmc_prott5_glm2_gated_v1.json`. Add a focused warm-start utility module if needed rather than growing the trainer with loading logic.

**Interfaces:**
- `ResidueGatedPool(dim, hidden_dim=64).forward(x, mask) -> [B,D]`.
- `GatedContrastiveHead(core, dim, hidden_dim=64)` keeps `self.head=core.head`, adds `self.residue_pool`; forward pools then uses existing head and normalization.
- `PatchResidualGate(dim, hidden_dim=128).forward(patches, mask) -> [B,P,D]`.
- `EnsembleResidualFusion(output_dim=1280).forward(primary, esmc, prott5, mask) -> [B,L,1280]` with separately named ESMC/ProtT5 alpha parameters.
- `FrozenProtT5(directory, primary_tokenizer)` loads local pinned T5 encoder and tokenizer, builds validated one-residue lookup, appends EOS, strips EOS/padding and returns `[B,L,1024]`; all parameters frozen and `train()` retains eval.
- `build_model(config, keys, device)` and `load_checkpoint` retain their existing interfaces. New opt-in `fusion.source='esmc_prott5'`, `fusion.esmc_model`, `fusion.prott5_model`, `clip_pooling`, `patch_gating`, `warm_start_checkpoint` and `coverage_evaluation` drive the new behavior.
- Artifact contract: ESMC keeps existing manifest; ProtT5 has `manifest.json` containing `repository`, immutable `revision`, and `files` mapping relative filenames to SHA256. All pinned files are checked by `verify_inputs`.
- `warm_start_checkpoint` points to predecessor `motif/checkpoints/step_001616`. Inherit every source trainable tensor by exact name and shape; only recognized new gate/fusion parameters may be absent. Verify base/LoRA compatibility and source hashes. Report inherited/new parameter names in initialization metadata. Fresh optimizer is built by the existing trainer; same-architecture `--resume` continues to restore all state strictly.

- [ ] **Step 1: Add failing behavior tests.** Include uniform-gate equivalence, learned nonuniform weighting, padding invariance (including NaN padding), empty-mask behavior, gradients only on valid residues, zero residual identity, nonzero scale gradient then downstream projection gradients after an update, both fusion branches, preserved patch grid/coverage labels, tokenizer fake with shifted special tokens and noncanonical residues, frozen eval/no-grad, and strict warm-start rejection of a missing inherited parameter.

```python
def test_uniform_residue_gate_equals_masked_mean():
    x = torch.randn(2, 9, 16, requires_grad=True)
    mask = torch.arange(9)[None] < torch.tensor([4, 9])[:, None]
    pool = ResidueGatedPool(16)
    expected = x.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[:, None]
    torch.testing.assert_close(pool(x, mask), expected)
    pool(x, mask).sum().backward()
    assert torch.count_nonzero(x.grad[~mask]) == 0

def test_three_residue_grid_keeps_tail():
    spec = PatchSpec(3, 2)
    assert spec.count(512) == 256
    for length in range(1, 20):
        covered = {i for j in range(spec.count(length))
                   for i in range(j * 2, min(length, j * 2 + 3))}
        assert covered == set(range(length))
```

- [ ] **Step 2: Run focused tests before implementation and record expected missing-module failure.** Command: `python -m pytest -q tests/test_gated_ensemble.py`.
- [ ] **Step 3: Implement gates and frozen-token alignment using the specification formulas.** FP32 pooling accumulation; exclude invalid inputs before normalization or projection. At zero gate output, sigmoid gates are 0.5 and normalized pooling equals mean. For all-padding rows output finite zero. Apply output mask after gated residual transformations. Frozen modules use lazy imports so CPU tests need not load real checkpoints. Use T5 encoder only, local files, and its model-specific EOS/mask lookup. The primary masked sequence is the only token source, so no hidden clean-sequence bypass.
- [ ] **Step 4: Integrate opt-in model/config, warm start, source tracking and diagnostics.** Legacy configs must retain identical behavior. Extend SOURCE_FILES to every new training module. Add explicit source-specific alpha and patch-beta diagnostics. Add configurable coverage threshold propagated from config through evaluation without mutating legacy defaults. Config uses the global constants above and fixed absolute artifact paths; keep `initialization.source='original_glm2'` to reconstruct the pinned base, then apply inherited trainables.
- [ ] **Step 5: Run focused and existing affected regressions once.** `python -m pytest -q tests/test_gated_ensemble.py tests/test_patch_contact.py tests/test_patch_coverage.py tests/test_unique_contact_negatives.py tests/test_sequence_masking.py tests/test_research_multigpu.py`. New behavior tests must exercise real tensor operations, not merely check mocks were called. Compile/import the modified training/evaluation entrypoints.
- [ ] **Step 6: Commit only Task 1 files and write `.superpowers/sdd/task-1-report.md`.** Include RED/GREEN evidence, exact commands, inherited names and public config contract, any dependency/precision caveats, and test counts. Do not start a GPU job or download weights; the controller handles real artifacts.

### Task 2: Verified preflight and single-instance training handoff

**Files:** Create `tools/check_gated_ensemble.py`, `tools/run_gated_after_training.py`, `tests/test_gated_handoff.py`; extend `tools/evaluate_research_checkpoint.py` only if needed for the new threshold or required loader. Keep queue/process logic separate from GPU validation.

**Interfaces:**
- `tools/check_gated_ensemble.py --config PATH --output PATH [--source-checkpoint PATH] [--memory-fraction FLOAT] [--quick]`: validate the real opt-in model; optional source override is only for disposable current-checkpoint checks. Default full mode tests one real global batch with configured microbatch; quick mode is explicitly not accepted for automatic launch. Write `checks.json` only as passed after all required checks.
- `tools/run_gated_after_training.py --predecessor PATH --config PATH --output PATH [--poll-seconds 30] [--check-only]`: wait for a verified successful predecessor, perform full preflight, train via `python -u -m flashppi.research_train`, evaluate final via `tools/evaluate_research_checkpoint.py`, publish stage/status/child PID/times. `--check-only` validates queue inputs/state without spawning GPU tasks.
- Queue root can already contain `preparation/`; refuse existing training, preflight or final-evaluation output instead of overwriting it. The parent holds an `fcntl.flock` for its full lifetime. Queue manifest pins source/config hashes, predecessor identities and final checkpoint location before waiting.

- [ ] **Step 1: Write failing handoff tests with temporary directories and controlled subprocesses.** Cover training-in-progress, evaluation-in-progress, normal success, failed/stopped trainer, dead predecessor, PID reuse/start-time identity, incomplete final checkpoint/test artifacts, missing ProtT5 manifest, changed pinned config/source, duplicate queue, signal propagation and exit-code failures. Run `python -m pytest -q tests/test_gated_handoff.py` and record RED evidence.

```python
def test_incomplete_evaluation_cannot_release_handoff(tmp_path):
    # Build a predecessor whose trainer is complete but supervisor still evaluates.
    predecessor = make_predecessor(tmp_path, trainer_status="complete",
                                   supervisor_status="evaluating_ecoli")
    assert predecessor_state(predecessor)["ready"] is False
```

- [ ] **Step 2: Implement explicit predecessor success and failure transitions.** Require supervisor status `complete`, `final_step==1616`, trainer status `complete`, trainer step 1616, final checkpoint manifest/weights/state hashes, final E. coli metrics plus manifest with correct checkpoint hash, and all predecessor GPU children exited. A dead process with non-success status is failure, not a completed run. Poll at 30 seconds and write a heartbeat. SIGTERM/SIGINT forward to current child, mark stopped, and do not start the next stage.
- [ ] **Step 3: Implement real-model disposable checks.** Compare inherited weights byte-for-byte, frozen base digests before/after, all new branch gradient paths over multiple small updates, finite outputs/loss/gradients, tokenizer alignment, zero gate/fusion initialization versus a 3/2 mean baseline, padding/batch-composition invariance, saved/reloaded predictions, and exact RNG/optimizer replay under deterministic execution. Then reload pristine warm-start parameters and run a real batch of 1024 positives at microbatch 16/max_len 512 with 1:1 PPI/DDI and unchanged negative policy; require zero duplicate/self/shortfall counts. Include peak allocated/reserved memory, time, GPU/runtime versions, model/data/source/config hashes in report. Probe updates are never inherited into actual training.
- [ ] **Step 4: Wire stages and final evaluation.** Preflight must succeed with full-batch proof and matching hashes before trainer launch. Use argv lists, no shell interpolation. Verify trainer status/step after return code zero, then evaluate the new final checkpoint and verify row count/manifest before marking complete. If any stage fails, retain logs and mark failed with the exact reason; no retries or adaptive microbatch changes.
- [ ] **Step 5: Run focused queue tests and relevant existing checkpoint tests.** `python -m pytest -q tests/test_gated_handoff.py tests/test_gated_ensemble.py`; run both CLIs with `--help` and queue `--check-only` against the real still-running predecessor. Do not start or stop GPU jobs.
- [ ] **Step 6: Commit Task 2 files and write `.superpowers/sdd/task-2-report.md`.** Include test evidence, queue state schema, real preflight checks and how to interpret pending/failed conditions.

## Controller completion and operational validation

- [ ] Prepare a separate Python environment without changing existing ones; fetch the public ProtT5 encoder at a pinned revision and hash every required file; verify the real tokenizer's residue and sentinel IDs.
- [ ] Review each task's diff with a fresh reviewer before marking it complete. Resolve Critical/Important findings with tested fixes and re-review.
- [ ] Run a small real GPU preflight with a conservative process memory cap while the predecessor remains active; avoid concurrent full-batch stress training. Record this as quick verification, not launch authorization.
- [ ] Perform final whole-branch review and the focused combined regression suite when changes warrant it.
- [ ] Pin the finalized queue sources/config and start the waiting supervisor in a detached process. Read back its lock/PID/status and ensure it is waiting for predecessor completion, not already training.
- [ ] Update the root specification with actual worktree/environment/artifact locations, tests, queue PID/status path, start/stop commands and deferred full GPU preflight. Commit documentation in the isolated branch, copy final user-facing docs to the original checkout, and leave a clean reproducible worktree. Do not merge into the running checkout.
