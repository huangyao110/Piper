"""Legacy imports resolve to Piper's modules; no model implementation lives here."""
from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder
import sys

import piper

__all__ = piper.__all__
_RENAMED = {
    "research_train": "train", "research_model": "ensemble",
    "research_eval": "evaluation", "research_distributed": "distributed",
    "finetune": "training_utils", "train": None, "eval": None,
}


class _AliasLoader(Loader):
    def __init__(self, target, target_spec):
        self.target = target
        self.target_spec = target_spec

    def create_module(self, spec):
        module = import_module(self.target)
        self.original_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery assigns the alias spec; preserve the canonical identity.
        module.__spec__ = self.original_spec

    def get_code(self, fullname):
        if fullname == "flashppi.finetune":
            raise ImportError("The legacy finetune CLI was retired; use python -m piper train.")
        return self.target_spec.loader.get_code(self.target)

    def get_filename(self, fullname):
        return self.target_spec.origin


class _AliasFinder(MetaPathFinder):
    _piper_legacy_alias = True

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("flashppi."):
            return None
        name = fullname.removeprefix("flashppi.")
        name = _RENAMED.get(name, name)
        if name is None or "." in name:
            return None
        canonical = f"piper.{name}"
        spec = util.find_spec(canonical)
        if spec is None:
            return None
        return util.spec_from_loader(fullname, _AliasLoader(canonical, spec), origin=spec.origin)


if not any(getattr(finder, "_piper_legacy_alias", False) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name):
    if name in __all__:
        return getattr(piper, name)
    raise AttributeError(name)
