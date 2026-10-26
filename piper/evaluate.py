"""Explicit official E. coli test evaluation of a portable research checkpoint."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from piper.training_utils import sha256
from piper.evaluation import evaluate_test, atomic_json
from piper.ensemble import load_checkpoint
from piper.train import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.checkpoint / "config.json").read_text())
    hashes = verify_inputs(config)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    metrics = evaluate_test(model, tokenizer, config, args.output)
    atomic_json(args.output / "manifest.json", dict(checkpoint=str(args.checkpoint.resolve()),
                trainable_sha256=sha256(args.checkpoint / "trainable.safetensors"), input_hashes=hashes,
                orientation="published", sequence_length="full", filtering="none"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
