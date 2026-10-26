# Upstream FlashPPI examples

These are the original proteome and cross-proteome screening scripts and training YAML examples retained from FlashPPI. They load upstream Hugging Face models. They do **not** automatically load Piper's ensemble, gates or trainable-only checkpoints.

For Piper pair prediction use `python -m piper predict` from the repository root. For an upstream baseline, consult the upstream documentation and install its optional screening dependencies (including FAISS) in a separate environment if needed. The YAML examples use the legacy `flashppi.train` entry point and its optional `datasets`/logging dependencies, rather than the current Piper trainer.

The synthetic pair in `../pairs.csv` demonstrates input format only; it has no experimental interaction label.
