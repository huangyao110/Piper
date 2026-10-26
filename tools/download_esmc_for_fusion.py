"""Download the pinned public ESM C 300M checkpoint and verify its LFS hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import requests

REPO = "biohub/esmc-300m-2024-12"
REVISION = "7f10b20ae75017b2dbc884070e03434515709a8d"
FILENAME = "data/weights/esmc_300m_2024_12_v0.pth"
SHA256 = "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/esmc_300m"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "weights.pth"
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
    if not target.exists():
        # Hugging Face handles interrupted large downloads and signed CDN URLs.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import hf_hub_download
        staging = args.output / ".download_cache"
        temporary = Path(hf_hub_download(REPO, FILENAME, revision=REVISION, local_dir=staging))
        if digest(temporary) != SHA256:
            raise RuntimeError("Downloaded ESM C weight checksum mismatch")
        temporary.replace(target)
        shutil.rmtree(staging)
    if digest(target) != SHA256:
        raise RuntimeError("Existing weights do not match pinned ESM C checkpoint")
    card = requests.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/README.md", timeout=30)
    card.raise_for_status()
    (args.output / "MODEL_CARD.md").write_text(card.text)
    metadata = dict(repo=REPO, revision=REVISION, source_file=FILENAME,
                    sha256=SHA256, filename=target.name, d_model=960,
                    n_heads=15, n_layers=30, software="esm==3.2.3")
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
