"""Paths, provenance and local-only runtime setup."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


def local_path(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT) or result == ROOT:
        raise ValueError(f"Research output must be inside {ROOT}: {result}")
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Reject symlinks for both the final and temporary paths.
    local_path(temporary)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_id(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def check_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}; expected {expected}, got {actual}")
    return actual


def setup_environment():
    sys.dont_write_bytecode = True
    for name in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        os.environ.pop(name, None)
    for name, relative in {
        "HF_HOME": "cache/huggingface", "HF_MODULES_CACHE": "cache/hf_modules",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor", "MPLCONFIGDIR": "cache/matplotlib",
        "XDG_CACHE_HOME": "cache/xdg", "TMPDIR": "tmp",
    }.items():
        directory = local_path(relative)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                      TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import tempfile
    tempfile.tempdir = os.environ["TMPDIR"]


def verify_prepared(config):
    manifest = read_json(local_path("manifests/prepared.json"))
    if manifest["study_sha256"] != sha256(ROOT / "study.json"):
        raise ValueError("study.json changed; prepare a new, explicit research version")
    for relative, digest in manifest["generated_hashes"].items():
        check_hash(local_path(relative), digest)
    for name, relative in config["checkpoints"].items():
        checkpoint = REPOSITORY / relative
        for filename, digest in manifest["checkpoints"][name]["metadata_hashes"].items():
            check_hash(checkpoint / filename, digest)
    return manifest


def activate_snapshot(name, manifest):
    directory = local_path(f"vendor/{name}")
    if "flashppi" in sys.modules:
        raise RuntimeError("Use a fresh process for each checkpoint snapshot")
    for relative, info in manifest["checkpoints"][name]["sources"].items():
        check_hash(local_path(relative), info["sha256"])
    sys.path.insert(0, str(directory))
    return directory
