# Configuration directory

All shared configuration files are organized here:

| Subdirectory | Contents |
|---|---|
| `train/` | Original RF2-PPI v3 training configuration and portable example |
| `data/` | Versioned split configurations and immutable v1/v2/v3 protocol locks |
| `models/` | Pinned encoder manifests and [asset preparation instructions](models/README.md) |
| `accelerate/` | Historical CPU/multi-GPU Accelerate templates; the current Piper trainer uses torchrun |

Run from the repository root:

```bash
python -m piper split --mode check --config config/data/conglab_distilled_ecoli_rf2ppi_v3.json
python -m piper train --config config/train/piper_rf2ppi_v3.example.json --output runs/piper
```

The training command starts a new run and requires prepared, audited data and pinned models. See the [main README](../README.md) for setup and evaluation.

The 0.2.1 layout replaces `configs_train/`, `configs_data/`, `configs_models/` and `configs_accelerate/`. Every JSON/YAML configuration, model manifest and protocol lock was moved byte-for-byte. No hyperparameters, split members, MMseqs settings or expected hashes were changed. For machine-specific storage paths, copy a configuration to `config/train/*.local.json` or `config/data/*.local.json`; these local copies are ignored by Git. Never refresh a protocol lock to accept different inputs.

The [relocation record](../docs/CONFIG_LAYOUT_20261026.json) lists every moved configuration and its original SHA-256.

Historical lock records still contain their original `configs_data/...` keys. The data verifier resolves those keys to `config/data/...` and checks the original hashes; it does not rewrite the lock or fall back to an obsolete copy. The four locked filter scripts also retain their original bytes. If explicitly running the historical mixture utility, supply `--config config/data/conglab_mixture_v1.json` rather than using its old default path. The maintained split entry point remains `tools/train_test.py`.

Existing training checkpoints and unfinished data builds must use their own original configuration and frozen source snapshot for exact continuation. Their source-hash checks remain strict. Experiment snapshots, checkpoint-local `config.json` files, source-model configs and the independent `resercher/study.json` are records belonging to those runs/assets and stay in their original locations. Historical audit documents retain the paths recorded at the time.
