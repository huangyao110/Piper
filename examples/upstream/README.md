# Upstream FlashPPI examples

These are the original proteome and cross-proteome screening scripts retained from FlashPPI, plus the pinned upstream E. coli evaluator. They load upstream Hugging Face models. They do **not** automatically load Piper's ensemble, gates or trainable-only checkpoints.

For Piper pair prediction use `python -m piper predict` from the repository root. For an upstream baseline, consult the upstream documentation and install its optional screening dependencies (including FAISS) in a separate environment if needed. The obsolete YAML trainer and its configs have been retired; Git history retains them.

Use `python examples/upstream/evaluate_ecoli.py --help` for baseline evaluation options after the editable installation. Classification metrics are shared with `piper.evaluation`; the baseline's model loading remains separate from Piper's ensemble.

The synthetic pair in `../pairs.csv` demonstrates input format only; it has no experimental interaction label.
