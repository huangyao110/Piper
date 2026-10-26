"""Compare trainable tensors and optimizer states after uninterrupted/resumed training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def compare(a, b, name="state"):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError(f"Tensor mismatch: {name}; max delta={(a.double() - b.double()).abs().max().item()}")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise AssertionError("Array mismatch: " + name)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("Different keys: " + name)
        for key in a:
            compare(a[key], b[key], name + "." + str(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("Different lengths: " + name)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, name + "." + str(i))
    elif a != b:
        raise AssertionError("Value mismatch: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = load_file(str(args.reference / "trainable.safetensors"))
    b = load_file(str(args.resumed / "trainable.safetensors"))
    compare(a, b, "trainable")
    sa = torch.load(args.reference / "training_state.pt", map_location="cpu", weights_only=False)
    sb = torch.load(args.resumed / "training_state.pt", map_location="cpu", weights_only=False)
    for key in ("step", "config", "world_size", "input_hashes", "source_hashes", "optimizer", "rng"):
        compare(sa[key], sb[key], key)
    result = dict(exact_match=True, step=sa["step"], tensors=len(a),
                  optimizer_exact=True, rng_exact=True, world_size=sa["world_size"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
