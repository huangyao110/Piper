"""Download pinned official checkpoints and benchmark into this workspace."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download

SOURCES = [
    ("tattabio/ecoli_pdb_benchmark", "dataset", "611db1c1571c785c25f8268b7b2a39bc89f926ce"),
    ("tattabio/flashppi_noecoli", "model", "aebaded7533d1e27d83a855b13b6499352e82894"),
    ("tattabio/flashppi", "model", "f6238ca5e5814d8a39b7e07822e9c35d5dc00b35"),
]


def main():
    destination = ROOT / "artifacts/flashppi_official"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    checks = []
    for repo, kind, revision in SOURCES:
        local = destination / repo.split("/")[-1]
        snapshot_download(repo, repo_type=kind, revision=revision, local_dir=local,
                          allow_patterns=["*.json", "*.py", "*.md", "*.safetensors", "data/*.parquet"],
                          max_workers=4)
        for item in HfApi().list_repo_tree(repo, repo_type=kind, revision=revision, recursive=True):
            lfs = getattr(item, "lfs", None)
            if lfs is None or not (local / item.path).is_file():
                continue
            digest = hashlib.sha256()
            with (local / item.path).open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            assert actual == lfs.sha256, f"Checksum mismatch: {repo}/{item.path}"
            checks.append(dict(repo=repo, revision=revision, file=item.path,
                               size=(local / item.path).stat().st_size, sha256=actual))
            print("VERIFIED", repo, item.path, actual, flush=True)
    (destination / "verified_downloads.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
