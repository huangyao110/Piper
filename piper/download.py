"""Fetch pinned model assets and verify their upstream SHA-256 checksums."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ESMC_REPO = "biohub/esmc-300m-2024-12"
ESMC_REVISION = "7f10b20ae75017b2dbc884070e03434515709a8d"
ESMC_FILENAME = "data/weights/esmc_300m_2024_12_v0.pth"
ESMC_SHA256 = "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea"
OFFICIAL_SOURCES = [
    ("tattabio/ecoli_pdb_benchmark", "dataset", "611db1c1571c785c25f8268b7b2a39bc89f926ce"),
    ("tattabio/flashppi_noecoli", "model", "aebaded7533d1e27d83a855b13b6499352e82894"),
    ("tattabio/flashppi", "model", "f6238ca5e5814d8a39b7e07822e9c35d5dc00b35"),
]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def download_esmc(output):
    import requests
    from huggingface_hub import hf_hub_download

    output.mkdir(parents=True, exist_ok=True)
    target = output / "weights.pth"
    if not target.exists():
        staging = output / ".download_cache"
        temporary = Path(hf_hub_download(ESMC_REPO, ESMC_FILENAME,
                                        revision=ESMC_REVISION, local_dir=staging))
        if digest(temporary) != ESMC_SHA256:
            raise RuntimeError("Downloaded ESM C weight checksum mismatch")
        temporary.replace(target)
        shutil.rmtree(staging)
    if digest(target) != ESMC_SHA256:
        raise RuntimeError("Existing weights do not match pinned ESM C checkpoint")
    card = requests.get(
        f"https://huggingface.co/{ESMC_REPO}/resolve/{ESMC_REVISION}/README.md", timeout=30)
    card.raise_for_status()
    (output / "MODEL_CARD.md").write_text(card.text)
    metadata = dict(repo=ESMC_REPO, revision=ESMC_REVISION, source_file=ESMC_FILENAME,
                    sha256=ESMC_SHA256, filename=target.name, d_model=960,
                    n_heads=15, n_layers=30, software="esm==3.2.3")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def download_official(output):
    from huggingface_hub import HfApi, snapshot_download

    output.mkdir(parents=True, exist_ok=True)
    (output / "sources.json").write_text(json.dumps(OFFICIAL_SOURCES, indent=2) + "\n")
    checks = []
    for repo, kind, revision in OFFICIAL_SOURCES:
        local = output / repo.split("/")[-1]
        snapshot_download(repo, repo_type=kind, revision=revision, local_dir=local,
                          allow_patterns=["*.json", "*.py", "*.md", "*.safetensors", "data/*.parquet"],
                          max_workers=4)
        for item in HfApi().list_repo_tree(repo, repo_type=kind, revision=revision, recursive=True):
            lfs = getattr(item, "lfs", None)
            if lfs is None or not (local / item.path).is_file():
                continue
            actual = digest(local / item.path)
            if actual != lfs.sha256:
                raise RuntimeError(f"Checksum mismatch: {repo}/{item.path}")
            checks.append(dict(repo=repo, revision=revision, file=item.path,
                               size=(local / item.path).stat().st_size, sha256=actual))
            print("VERIFIED", repo, item.path, actual, flush=True)
    (output / "verified_downloads.json").write_text(json.dumps(checks, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("esmc", "official"))
    parser.add_argument("--output", type=Path, help="Destination; defaults to artifacts/esmc_300m or artifacts/flashppi_official")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if args.asset == "esmc":
        download_esmc(args.output or ROOT / "artifacts/esmc_300m")
    else:
        download_official(args.output or ROOT / "artifacts/flashppi_official")


if __name__ == "__main__":
    main()
