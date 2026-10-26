#!/usr/bin/env python3
"""CPU-only annotation; cached public downloads and all outputs stay in resercher."""
import sys
sys.dont_write_bytecode = True
import argparse
from clip_gate.common import setup_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sasa", "conservation", "assemble"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use 1 to 4 workers to leave resources for background training")
    setup_environment()
    from clip_gate.annotations import annotate, assemble
    if args.command == "assemble":
        assemble()
    else:
        annotate(args.command, args.workers)


if __name__ == "__main__":
    main()
