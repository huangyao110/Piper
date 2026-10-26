"""Thin command routing without changing training, evaluation or isolation logic."""
import argparse
from pathlib import Path
import runpy
import sys

from . import __version__

COMMANDS = {
    "train": ("module", "flashppi.research_train"),
    "evaluate": ("script", "tools/evaluate_research_checkpoint.py"),
    "split": ("script", "tools/train_test.py"),
    "predict": ("module", "piper.predict"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="piper", description="Piper protein interaction and residue contact research toolkit.",
        epilog="Run from the repository root. Use 'piper COMMAND --help' for command options.",
    )
    parser.add_argument("--version", action="version", version=f"Piper {__version__}")
    parser.add_argument("command", choices=COMMANDS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in COMMANDS:
        parser.parse_args(arguments)
        return
    command = arguments[0]
    kind, target = COMMANDS[command]
    root = Path(__file__).resolve().parents[1]
    old_argv, old_path = sys.argv[:], sys.path[:]
    try:
        sys.argv = [target, *arguments[1:]]
        if kind == "module":
            runpy.run_module(target, run_name="__main__")
        else:
            script = root / target
            if not script.is_file():
                parser.error("This command requires the source checkout; install with pip install -e .")
            sys.path.insert(0, str(script.parent))
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv, sys.path = old_argv, old_path
