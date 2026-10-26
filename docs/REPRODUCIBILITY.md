# Reproducibility and execution

## Required records

For each result retain the Git commit, exact configuration, model/input hashes, split manifest and audit, source snapshot, dependency versions, device type, CUDA/PyTorch versions, seed, world size, checkpoint step and evaluation output. Changing source mixtures, isolation predicates, encoder precision or negative sampling creates a different experimental condition.

The current original configuration is `configs_train/conglab_distilled_rf2ppi_v3_4epochs.json`. The separately named `piper_rf2ppi_v3.example.json` changes only asset storage paths for a fresh checkout. It is not a replacement configuration for an existing run.

## Starting and stopping

`python -m piper train --config CONFIG --output NEW_DIRECTORY` calls the maintained `piper.train` module. Model, training and evaluation implementations now live together in `piper/`. The 0.2 consolidation changes module paths and removes obsolete entry points; it adds no optimizer, sampling, crop, gate or loss changes. `python -m piper split` still calls the existing `tools/train_test.py`, without duplicating its logic.

To stop at a reproducible step boundary, launch with the existing `--stop-after STEP` option while retaining the full schedule in the config. An abrupt kill may discard progress since the last saved checkpoint. Do not start concurrent evaluations on a GPU that is already full.

The current long-running workstation job uses an independent runtime snapshot under `experiments/`. Organizational changes in this release do not replace that snapshot. Local process supervisors and crash diagnostics are not part of the portable source release.

## Checkpoints and exact resume

Each checkpoint contains `trainable.safetensors`, `training_state.pt`, `config.json`, and `manifest.json`. Frozen encoders and base weights are external. The training state records optimizer/RNG state, configuration, world size, input hashes and source hashes. The runtime saves selected source files alongside its output.

`--resume CHECKPOINT` requires the original configuration, inputs, source hashes and GPU world size. **Checkpoints created before the 0.2 consolidation must resume in their original frozen runtime**, not in a newly reorganized checkout. Commit `619b186465c4b4a7a8c1b5e77b81753042021f29` records the immediately preceding repository layout; a run's own saved hashes determine its required version. Source-hash checks are not relaxed or regenerated to hide this boundary.

`flashppi/` contains only a legacy import adapter: for example, `flashppi.research_model` resolves to the same module object as `piper.ensemble`. It does not provide the original source bytes required for exact resume. Parameter names and model computation are preserved for checkpoint loading and prediction. The old standalone YAML trainer (`flashppi.train`) and old finetune CLI are retired. Historical runs retain their original executables in their frozen snapshots.

`--continue-from CHECKPOINT` exists for explicitly configured new phases. It is not an escape hatch for a failed exact-resume check. Do not use it silently when a source, data or hardware change invalidates exact continuation.

Existing absolute artifact and index paths may require the same mounts on a new server. A copied checkpoint directory by itself is insufficient. The original dataset shards, manifests, aliases/weights, fixed benchmark and pinned pretrained assets must remain available.

## Environment

Use Python 3.12 and the single [requirements.txt](../requirements.txt) for training, prediction, data preparation and tests. It pins direct dependencies, including DuckDB, Gemmi and pytest. Install the appropriate CPU or CUDA PyTorch wheels first; both satisfy the same package version constraints. Package metadata, CI and the server setup script use this one list.

[ENVIRONMENT_REFERENCE.json](ENVIRONMENT_REFERENCE.json) preserves the complete installed CUDA 12.8 environment previously captured in the server lock file. It is a historical record, not an installation input. The unified requirements pin direct dependencies rather than every transitive package; archive each run's resolved environment with `python -m pip freeze`, along with the source/model/data hashes. A reference package list alone does not promise bitwise equality across GPUs or drivers.

`tools/setup_server_env.sh` creates a separate `.venv/server`, installs CUDA 12.8 PyTorch and the unified requirements, checks dependencies, and records the resolved versions in that environment's `installed-lock.txt`. `tools/run_server_training.sh CONFIG OUTPUT GPU_IDS` explicitly launches `piper.train` with torchrun. GPU topology checks use `python -m piper preflight`; checkpoint replay comparisons use `python -m piper check-replay`. These utilities must not be used to reinstall dependencies into a currently active training environment.

Frozen-encoder reuse avoids recomputation within an optimizer step. Length bucketing and prefix caching are explicit config choices. Numerical equivalence should be established for the setting being compared; lower precision or changed padding shapes can alter hard-negative ranking and hence gradients.

## Validation tiers

1. **Offline CPU regressions:** synthetic tensors, temporary source shards and fixed protocol boundary cases. Run `pytest -m 'not integration'` with CUDA hidden and offline Hugging Face flags.
2. **Local integrations:** real MMseqs end-to-end split tests, historical alignment evidence, and experiment-controller recovery checks, marked `integration`; they need an external binary and/or archives and are not CI prerequisites. For the synthetic split integration, set `PIPER_MMSEQS` to the MMseqs executable and run `python -m pytest -q tests/test_indexed_split.py -m integration`. This test-only executable selection does not change the production binary/hash lock.
3. **Real-model checks:** require the pinned weights and adequate GPU memory. CPU toy-model tests do not establish real-model throughput, numerical equivalence on every GPU, or predictive performance.

The isolated `resercher/tests` cover its biological analysis logic and run separately. Results from those historical checkpoints should not be attributed to the current fresh RF2-PPI v3 model.

## Reporting

Contact P@K at both distance cutoffs is primary. Include eligible structure counts, label/mask convention, exact checkpoint step and split hash. Report interaction metrics separately. Test is monitored every 200 steps plus the final step under the fixed 4-epoch schedule; no validation-based best model exists in this run. Repeated test inspection must be disclosed.

For comparative claims, match data membership, negative-pair definition, cutoff, K definition, masking, orientation and full-length evaluation. A different split size or lower training loss alone is not evidence of superior generalization.
