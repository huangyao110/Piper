# Repository Guidelines

## Active RF2-PPI Isolation Rebuild (User Approval, 2026-10-24)

The latest user explicitly requested adopting the RF2-PPI isolation rules and rebuilding `train.index.parquet`. The active new protocol is `conglab_distilled_ecoli_rf2ppi_v3`, in `configs_data/conglab_distilled_ecoli_rf2ppi_v3.json` with its separately pinned `.lock.json`. Use the EXISTING build/check scripts. Do not change historical v1/v2 lock files or memberships.

Cleanup authorization on 2026-10-24 supersedes older instructions to retain obsolete output directories: the user explicitly requested deleting V2 and unused retired directories. The strict v2 output, its unpublished draft, and retired `expanded_conglab_pinder_teddymer_v1` output have been deleted; see `docs/DATA_CLEANUP_20261024.json`. Keep their small historical configuration/lock and comparison records. Retain original `conglab_ecoli_v1`, the valid legacy expansion, active v3, and original source distributions because they still have uses. Never infer permission to delete these from the obsolete-directory cleanup.

- PPI and DDI use the SAME hit predicate: identity `nident/alnlen >0.30`, reported MMseqs E-value `<1e-5`, and query coverage `>0.50` OR target coverage `>0.50`. All boundaries are strict. No unconditional exact-match exemption from significance.
- Exclude a training pair only if BOTH endpoints independently hit the reference set. A single hit is allowed; this does not guarantee C3. No requirement to hit the same test pair.
- Keep the fixed historical E. coli positive/negative endpoint reference, no val, CongLab PPI/DDI plus the eight selected distilled PPI sources, and supplied 70/70 sequence TSV sampling. Only isolation changes, not source quality/contact/mask semantics.
- Paper evidence: supplementary M3.1 pp.20–21 and M3.3 p.27 archived under `experiments/rf2ppi_isolation_review_20261024/`. The full original search command/version was not published. Our explicit MMseqs 18.8cc5c settings, dual-direction k6/direct searches, no coverage prefilter (`-c 0`), binary/input hashes and scope are local reproducible choices, not claimed author parameters.
- Build recomputes all source decisions and independent check reparses raw significance/alignment evidence and verifies BOTH-endpoint exclusions. No legacy exemption, automatic lock refresh or automatic training start.

## Historical Fresh CongLab + Distilled Train/Test v2 (2026-10-24)

Scientific review on 2026-10-24 identified a material over-filtering concern in v2. Read `docs/RF2PPI_ISOLATION_REVIEW_20261024.md` before interpreting the split as scientifically validated. Direct identity/coverage hits are not confirmed biological homologs; a diagnostic sample found only 4 of 1,000 newly hit DDI domains met E-value <1e-5 in the checked direction. This review changes no existing membership, parameters or lock. A mechanical audit pass does not certify an RF2-PPI reproduction or resolve this concern.

The user explicitly authorized a NEW `conglab_distilled_ecoli_strict_v2` protocol for fresh training. It does not supersede or rewrite historical v1 data. Use the existing `tools/train_test.py` and `tools/audit_expanded_index_split.py`; do not add another builder.

The maintained entrypoint must serve both requested functions: `--mode build` constructs the isolated split and verifies it; `--mode check` independently checks an existing completed split without changing membership. A failed check returns a nonzero exit code and records failure; never silently repair members or refresh the protocol lock during checking.

- Train includes CongLab PPI/DDI and the eight specified distilled PPI sources in `ppidb-2026.10.3`; exclude PINDER/Teddymer. Test is the unchanged historical E. coli benchmark. No validation split.
- Filter the pair CSV's corresponding full FASTA entries BEFORE training cropping. These entries may be source structure fragments/domains; do not invent canonical full-protein sequences or silently quarantine valid FASTA entries for lacking UniProt full-length metadata.
- Identity `nident/alnlen > 0.30`; PPI both coverages `>= 0.50`; DDI training-domain coverage `>= 0.50`. Any training endpoint hitting any positive/negative test sequence excludes the entire pair. Recheck all source members, without legacy exemptions or cached legacy decisions.
- Pin the new protocol in `configs_data/conglab_distilled_ecoli_strict_v2.lock.json`: MMseqs binary/version, arguments, input hashes, ordering, bidirectional k=6 search AND mandatory bidirectional full direct alignment (`prefilter-mode=2`), plus exact matches. K=6 alone misses already verified homologs; do not omit the direct passes. Reject drift; never automatically refresh this lock on rerun.
- Use the supplied 70/70 sequence cluster TSVs for BOTH PPI and DDI sampling metadata. Do not confuse sampling clusters with 30% isolation.
- Only index files, deduplication maps, evidence and documentation are built. No training starts automatically. Keep old locks, scripts, split artifacts and audit evidence intact.

## Frozen Data Filtering Protocol (User Requirement, 2026-10-22)

The historical CongLab/PINDER/Teddymer expansion remains governed by the original `conglab_ecoli_v1` protocol. The following rules apply to v1, not the separately approved fresh v2 above. Read `docs/DATA_FILTERING_POLICY.md` and `configs_data/conglab_ecoli_filter.lock.json` before any v1 filtering change.

- Do not change identity/coverage boundaries, test/val endpoint logic, reference sequences, search direction, MMseqs version, candidate prefilter parameters, known-sequence decisions, or label/mask semantics without explicit user approval. Dataset expansion, cleanup, optimization, diagnostics, or an apparently more stringent filter is not permission to change these rules.
- Preserve all 256,596 original CongLab PPI train rows and 570,709 DDI train rows. Validation/test files and their hashes are frozen. Reuse original cached decisions for sequences already evaluated by the original pipeline, including original negative search results.
- The original searches all resolved to MMseqs 18.8cc5c, k=6 and k-mer score=96. Expanded searches must explicitly pin these observed parameters and verify the logs; do not allow `-k 0` to silently become k=7. This explicit pin implements the user's instruction to retain the original actual search conditions.
- Keep the four original protocol scripts byte-for-byte pinned by the lock file. Do not replace their filtering logic with a newly written variant. Extensions must reuse the original classifier and validate the lock. Any deliberate protocol change needs approval, a new named version and a before/after report; never overwrite historical evidence.
- Diagnostic hits from k=7 or direct alignment are evidence only; they do not authorize changing this protocol or deleting old training members. Do not describe the original heuristic protocol as exhaustive homology exclusion.
- Keep `tools/` limited to maintained entry points and required dependencies. One-off analysis belongs in its experiment directory; do not add competing data builders for convenience. Check imports and callers before removing scripts.

## Project Structure & Module Organization

Piper is the current research project derived from FlashPPI. `piper/` provides public train/evaluate/split/predict entry points; `flashppi/` retains model, loss, sampling, evaluation and training implementations at checkpoint-compatible paths. Current JSON configurations live in `configs_train/`, immutable data protocols in `configs_data/`, and pretrained manifests in `configs_models/`. Original FlashPPI screening scripts and YAML examples are under `examples/upstream/`. Regression checks belong in `tests/`; the isolated biological gate study remains in `resercher/`. Historical documentation is under `docs/history/`. Local `experiments/`, data, weights and deployment bundles are excluded from publication.

## Build, Test, and Development Commands

Run commands from the repository root in a virtual environment. There is no separate build step.

- `pip install -r requirements.txt` installs the unified runtime, data-preparation and CPU test dependencies after installing the appropriate PyTorch wheels.
- `pip install --no-deps -e .` installs the Piper CLI for use from the source checkout.
- `python -m piper train --config configs_train/piper_rf2ppi_v3.example.json --output runs/piper` starts a new run only after assets/data are prepared and audited; never run this automatically during code cleanup.
- `python -m piper evaluate --checkpoint CHECKPOINT --output NEW_DIRECTORY` evaluates the fixed test benchmark.
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -m 'not integration'` runs the offline-compatible CPU regressions; integration tests require local historical artifacts.
- `python tests/test_contact_score_padding.py` runs the contact-score regression check.

Training requires the configured datasets and pinned pretrained models. Use the root README for the current research trainer. The original upstream YAML trainer has separate optional dataset/screening/logging dependencies; it is not the Piper training entry point.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Follow nearby code and use relative imports within `flashppi/`. Keep configuration fields aligned with the dataclasses in `flashppi/config.py`. Explain tensor shapes and masking assumptions where they affect correctness. No formatter or linter configuration is currently checked in.

## Testing Guidelines

Use descriptive `tests/test_*.py` filenames. The existing check is a standalone assertion-based script, requires CUDA, and loads `tattabio/flashppi2` from Hugging Face; it does not automatically test local model changes. It verifies that padding does not change a pair's score across batch compositions. Add focused regression checks for changed behavior and report hardware and model dependencies. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive subjects such as “Apply valid_mask before taking the max contact score”; no mandatory prefix convention is evident. Keep commits focused. In pull requests, describe the problem, resulting behavior, relevant issues, and validation commands with results. For model or training changes, include configuration, hardware, and relevant metrics. Keep generated checkpoints, prediction outputs, datasets, and credentials out of commits.
