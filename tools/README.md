# Maintained tools

This directory contains the single maintained split builder, its audit/dependencies, and two server shell launchers. Model, evaluation and checkpoint utilities live in `piper/`. The four historical filters remain byte-for-byte at the paths pinned by protocol locks. Temporary diagnostics belong in local `experiments/<name>/`, not here.

## Data

| Entry point | Purpose |
|---|---|
| `train_test.py --mode build` | Build the versioned index split and audit it; current protocol is RF2-PPI v3 |
| `train_test.py --mode check` | Independently verify an existing split and raw evidence; nonzero on failure |
| `audit_expanded_index_split.py` | Independent audit implementation used by the builder/checker |
| `build_ppi_distance_archives.py` | Verify and export source structural distance labels |
| `prepare_conglab_ecoli.py`, `prepare_conglab_mixture.py`, `prepare_ppi_finetune_split.py`, `audit_conglab_ecoli.py` | Byte-pinned historical filtering code and required dependencies; not alternative current builders |

```bash
python tools/train_test.py --mode build --config configs_data/conglab_distilled_ecoli_rf2ppi_v3.json
python tools/train_test.py --mode check --config configs_data/conglab_distilled_ecoli_rf2ppi_v3.json
```

See [the active data protocol](../docs/DATA_PROTOCOL.md) and [historical v1 policy](../docs/DATA_FILTERING_POLICY.md). Checking updates audit reports and acceptance state, never split membership or lock parameters. Build/check does not start training.

## Model commands

| Entry point | Purpose |
|---|---|
| `python -m piper evaluate` | Full-length fixed E. coli evaluation of a Piper checkpoint |
| `python -m piper check-replay` | Compare trainable tensors, optimizer and RNG states after resume |
| `python -m piper preflight` | GPU topology/NCCL checks; requires free devices and the requested world size |
| `python -m piper download esmc` | Checksum-verified pinned ESM C 300M download |
| `python -m piper download official` | Pinned upstream weights and E. coli benchmark |
| `python examples/upstream/evaluate_ecoli.py` | Upstream baseline evaluation, sharing classification metrics with `piper.evaluation` |

These commands support `--help`. The old model utility scripts have been moved or merged; there is one implementation per command. Use matching frozen source snapshots for exact resume of historical checkpoints.

## Assets and execution

- `setup_server_env.sh`: create a separate CUDA 12.8 Python environment using the root `requirements.txt`; record the resolved versions inside that environment.
- `run_server_training.sh CONFIG OUTPUT GPU_IDS`: explicit torchrun launcher for `piper.train`. Default environment: `.venv/server`.

The complete model setup, including gLM2, is documented in [configs_models/README.md](../configs_models/README.md).

Standalone DDI model training/evaluation, obsolete RF2-PPI benchmark evaluation, old real-checkpoint probes, pilot configs and the gLM2 range downloader were retired on 2026-10-26. File hashes and recovery archives are recorded in the [initial cleanup](../docs/PIPER_CLEANUP_20261026.json) and [consolidation](../docs/CODE_CONSOLIDATION_20261026.json) manifests. DDI examples remain in Piper training, and the RF2-PPI isolation rules and locks are unchanged.
