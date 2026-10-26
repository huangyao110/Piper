#!/usr/bin/env python3
"""All commands and output stay inside this independent research directory."""
import sys
sys.dont_write_bytecode = True

import argparse
from clip_gate.common import ROOT, read_json, setup_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "extract", "analyze", "compare"))
    parser.add_argument("--checkpoint", choices=("before_1616", "after_2182"))
    parser.add_argument("--annotations", help="Optional annotation CSV inside resercher; see ANNOTATIONS.md")
    args = parser.parse_args()
    setup_environment()
    config = read_json(ROOT / "study.json")
    if args.command in ("preflight", "extract", "analyze") and args.checkpoint is None:
        parser.error("--checkpoint is required for this command")
    if args.command == "prepare":
        from clip_gate.panel import prepare
        prepare(config)
    elif args.command in ("preflight", "extract"):
        from clip_gate import extraction
        getattr(extraction, args.command)(config, args.checkpoint)
    else:
        from clip_gate import analysis
        if args.command == "compare":
            analysis.compare(config)
        else:
            analysis.analyze(config, args.checkpoint, args.annotations)


if __name__ == "__main__":
    main()
