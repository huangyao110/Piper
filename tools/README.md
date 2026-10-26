# Maintained tools

Piper keeps a single maintained split builder and preserves the historical source paths that are pinned by protocol locks and training checkpoints. Temporary diagnostics belong in local `experiments/<name>/`, not here.

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

## Evaluation and recovery

| Entry point | Purpose |
|---|---|
| `evaluate_research_checkpoint.py` | Full-length fixed E. coli evaluation of a Piper/research checkpoint |
| `evaluate_official_ecoli.py` | Pinned upstream baseline evaluation and shared classification metrics |
| `check_research_replay.py` | Gradient-cache/replay verification used by regression tests and checkpoint source manifests |

Current public aliases are `python -m piper evaluate` and `python -m piper split`. Both dispatch to these existing scripts. Use matching model source snapshots for historical checkpoints.

## Assets and execution

- `download_esmc_for_fusion.py`: checksum-verified, pinned ESM C 300M download.
- `download_official_flashppi.py`: pinned upstream baseline weights and E. coli benchmark.
- `download_original_glm2.py`: historical range download of gLM2 weights; requires the small config/tokenizer files first. The complete fresh setup is documented in [configs_models/README.md](../configs_models/README.md).
- `setup_server_env.sh`: create a separate CUDA 12.8 Python environment using the root `requirements.txt`; record the resolved versions inside that environment.
- `server_preflight.py`: explicit GPU topology/NCCL checks; requires free devices and the requested world size.
- `run_server_training.sh CONFIG OUTPUT GPU_IDS`: explicit torchrun launcher for the research trainer. Default environment: `.venv/server`.

Standalone DDI model training/evaluation, obsolete RF2-PPI benchmark evaluation, old real-checkpoint probes and pilot configs were retired on 2026-10-26. File hashes and local recovery archives are recorded in [the cleanup manifest](../docs/PIPER_CLEANUP_20261026.json). This does not remove DDI examples from Piper training and does not change the RF2-PPI isolation implementation.
