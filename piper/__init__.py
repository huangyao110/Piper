"""Piper models, training and inference; heavy dependencies load on demand."""
from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
    "ModelConfig": "config", "TrainingConfig": "config",
    "FlashPPIModel": "model", "PiperModel": "ensemble",
    "DataCollatorForPPI": "dataset", "ClusterSampler": "dataset",
    "build_interaction_keys": "dataset", "ClipLoss": "loss",
    "ContactLoss": "loss", "GradCacheClipLoss": "loss", "PPIGradientCache": "loss",
    "load_checkpoint": "ensemble",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(f"piper.{_EXPORTS[name]}")
    value = getattr(module, "ResearchModel" if name == "PiperModel" else name)
    globals()[name] = value
    return value
