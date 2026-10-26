#!/usr/bin/env python3
"""All commands and output stay inside this independent research directory."""
import sys
sys.dont_write_bytecode = True

import argparse
from clip_gate.common import ROOT, read_json, setup_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "extract", "analyze", "compare",
                                            "sasa", "conservation", "assemble", "report"))
    parser.add_argument("--checkpoint", choices=("before_1616", "after_2182"))
    parser.add_argument("--annotations", help="Optional annotation CSV inside resercher; see ANNOTATIONS.md")
    parser.add_argument("--workers", type=int, default=2, help="Annotation download workers (1 to 4)")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use 1 to 4 workers to leave resources for background training")
    if args.command in ("preflight", "extract", "analyze") and args.checkpoint is None:
        parser.error("--checkpoint is required for this command")
    setup_environment()
    if args.command in ("sasa", "conservation", "assemble"):
        from clip_gate.annotations import annotate, assemble
        if args.command == "assemble":
            assemble()
        else:
            annotate(args.command, args.workers)
        return
    if args.command == "report":
        from annotation_report import main as report
        report()
        return
    config = read_json(ROOT / "study.json")
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
