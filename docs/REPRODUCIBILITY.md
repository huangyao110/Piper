# Reproducibility and execution

## Required records

For each result retain the Git commit, exact configuration, model/input hashes, split manifest and audit, source snapshot, dependency versions, device type, CUDA/PyTorch versions, seed, world size, checkpoint step and evaluation output. Changing source mixtures, isolation predicates, encoder precision or negative sampling creates a different experimental condition.

The current original configuration is `configs_train/conglab_distilled_rf2ppi_v3_4epochs.json`. The separately named `piper_rf2ppi_v3.example.json` changes only asset storage paths for a fresh checkout. It is not a replacement configuration for an existing run.

## Starting and stopping

`python -m piper train --config CONFIG --output NEW_DIRECTORY` calls the maintained `flashppi.research_train` module. The thin Piper entry point adds no optimizer, sampling, crop, gate or loss changes. `python -m piper split` calls the existing `tools/train_test.py`, without duplicating its logic.

To stop at a reproducible step boundary, launch with the existing `--stop-after STEP` option while retaining the full schedule in the config. An abrupt kill may discard progress since the last saved checkpoint. Do not start concurrent evaluations on a GPU that is already full.

The current long-running workstation job uses an independent runtime snapshot under `experiments/`. Organizational changes in this release do not replace that snapshot. Local process supervisors and crash diagnostics are not part of the portable source release.

## Checkpoints and exact resume

Each checkpoint contains `trainable.safetensors`, `training_state.pt`, `config.json`, and `manifest.json`. Frozen encoders and base weights are external. The training state records optimizer/RNG state, configuration, world size, input hashes and source hashes. The runtime saves selected source files alongside its output.

`--resume CHECKPOINT` requires the original configuration, inputs, source hashes and GPU world size. Use that run's frozen source snapshot for historical checkpoint reproduction. The `flashppi` implementation namespace is deliberately retained: renaming every module would break this provenance and existing tooling.

`--continue-from CHECKPOINT` exists for explicitly configured new phases. It is not an escape hatch for a failed exact-resume check. Do not use it silently when a source, data or hardware change invalidates exact continuation.

Existing absolute artifact and index paths may require the same mounts on a new server. A copied checkpoint directory by itself is insufficient. The original dataset shards, manifests, aliases/weights, fixed benchmark and pinned pretrained assets must remain available.

## Environment

Use Python 3.12 and [requirements.txt](../requirements.txt) for maintained direct dependencies. [requirements-server-lock.txt](../requirements-server-lock.txt) captures the installed CUDA 12.8 reference environment; it is platform-specific, not a promise of bitwise equality on another GPU or driver. `requirements-dev.txt` adds CPU tests, DuckDB for index handling, and Gemmi for geometry checks. `requirements-finetune.txt` and `requirements-server.txt` are compatibility aliases.

`tools/setup_server_env.sh` creates a separate `.venv/server` and installs the CUDA reference stack. `tools/run_server_training.sh CONFIG OUTPUT GPU_IDS` explicitly launches the research trainer with torchrun. These utilities must not be used to reinstall dependencies into a currently active training environment.

Frozen-encoder reuse avoids recomputation within an optimizer step. Length bucketing and prefix caching are explicit config choices. Numerical equivalence should be established for the setting being compared; lower precision or changed padding shapes can alter hard-negative ranking and hence gradients.

## Validation tiers

1. **Offline CPU regressions:** synthetic tensors, temporary source shards and fixed protocol boundary cases. Run `pytest -m 'not integration'` with CUDA hidden and offline Hugging Face flags.
2. **Local integrations:** historical MMseqs evidence and experiment-controller recovery checks, marked `integration`; they need external archives and are not CI prerequisites.
3. **Real-model checks:** require the pinned weights and adequate GPU memory. CPU toy-model tests do not establish real-model throughput, numerical equivalence on every GPU, or predictive performance.

The isolated `resercher/tests` cover its biological analysis logic and run separately. Results from those historical checkpoints should not be attributed to the current fresh RF2-PPI v3 model.

## Reporting

Contact P@K at both distance cutoffs is primary. Include eligible structure counts, label/mask convention, exact checkpoint step and split hash. Report interaction metrics separately. Test is monitored every 200 steps plus the final step under the fixed 4-epoch schedule; no validation-based best model exists in this run. Repeated test inspection must be disclosed.

For comparative claims, match data membership, negative-pair definition, cutoff, K definition, masking, orientation and full-length evaluation. A different split size or lower training loss alone is not evidence of superior generalization.
