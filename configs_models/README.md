# Pretrained assets

The JSON manifests in this directory are byte-for-byte copies of the manifests used by the current run. They contain hashes and provenance, not weights. Do not substitute a newer model revision or regenerate a manifest when resuming a run.

| Component | Repository | Pinned revision |
|---|---|---|
| gLM2 backbone | `tattabio/gLM2_650M_UR50` | `a5937fbb770bd8bae65e3034aacd823b041dcba3` |
| ESM C 300M | `biohub/esmc-300m-2024-12` | `7f10b20ae75017b2dbc884070e03434515709a8d` |
| ProtT5 XL encoder | `Rostlab/prot_t5_xl_half_uniref50-enc` | `94a6abc029ae13029317b140b7424e012bf8dfbf` |
| FlashPPI compatible architecture | `tattabio/flashppi_noecoli` | `aebaded7533d1e27d83a855b13b6499352e82894` |
| Fixed E. coli test | `tattabio/ecoli_pdb_benchmark` | `611db1c1571c785c25f8268b7b2a39bc89f926ce` |

From the repository root, after installing dependencies:

```bash
hf download tattabio/gLM2_650M_UR50 \
    --revision a5937fbb770bd8bae65e3034aacd823b041dcba3 \
    --include '*.json' '*.py' '*.safetensors' --local-dir artifacts/glm2_650m_ur50
cp configs_models/glm2_650m_ur50.manifest.json artifacts/glm2_650m_ur50/manifest.json

python -m piper download esmc --output artifacts/esmc_300m
cp configs_models/esmc_300m.manifest.json artifacts/esmc_300m/manifest.json

hf download Rostlab/prot_t5_xl_half_uniref50-enc \
    --revision 94a6abc029ae13029317b140b7424e012bf8dfbf \
    --include '*.json' README.md spiece.model pytorch_model.bin \
    --local-dir artifacts/prot_t5_xl_half_uniref50_enc
cp configs_models/prot_t5_xl_half_uniref50_enc.manifest.json artifacts/prot_t5_xl_half_uniref50_enc/manifest.json

# Fresh initialization needs architecture code/configuration, not upstream task weights.
hf download tattabio/flashppi_noecoli --revision aebaded7533d1e27d83a855b13b6499352e82894 \
    --include '*.json' '*.py' --local-dir artifacts/flashppi_official/flashppi_noecoli
hf download tattabio/ecoli_pdb_benchmark --repo-type dataset \
    --revision 611db1c1571c785c25f8268b7b2a39bc89f926ce \
    --include 'data/*.parquet' --local-dir artifacts/flashppi_official/ecoli_pdb_benchmark
```

These commands download several GB and are intentionally not run by installation, import, tests or data checking. Use each provider's access and licensing requirements. Review the pinned remote model code before use; the gLM2 loader uses `trust_remote_code=True` with local pinned files.

Training and full benchmark evaluation validate input hashes before running. A checkpoint-only pair prediction checks the trainable tensor checksum and model-specific loader requirements; it is not a substitute for the full dataset/input audit. For a relocated existing checkpoint, preserve the original asset locations or provide equivalent mounts; do not change a checkpoint config and call it an exact resume.

`python -m piper download official` downloads the pinned upstream weights and benchmark for baseline comparisons. The two asset download scripts are consolidated in `piper/download.py`; repository revisions, file filters and checksum checks are preserved. The obsolete gLM2 range-download helper was retired; use the complete `hf download` setup above.
