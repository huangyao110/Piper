"""Module moves preserve legacy identities, source snapshots and asset checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from piper import download

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("legacy_first", [True, False])
def test_legacy_imports_share_canonical_module_and_class_identity(legacy_first):
    code = f"""
from importlib import import_module, util
pairs = [('flashppi.ddi_common', 'piper.ddi_common'),
         ('flashppi.research_model', 'piper.ensemble'),
         ('flashppi.research_eval', 'piper.evaluation'),
         ('flashppi.finetune', 'piper.training_utils')]
for legacy, canonical in pairs:
    first, second = (legacy, canonical) if {legacy_first!r} else (canonical, legacy)
    a, b = import_module(first), import_module(second)
    assert a is b
    assert a.__name__ == a.__spec__.name == canonical
import flashppi, piper
assert flashppi.FlashPPIModel is piper.FlashPPIModel
assert piper.PiperModel is import_module('piper.ensemble').ResearchModel
assert util.find_spec('flashppi.train') is None
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_new_training_snapshot_imports_without_original_checkout(tmp_path):
    from piper.train import SOURCE_FILES

    for relative in SOURCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    env = os.environ.copy()
    # Keep optional external test dependencies, but do not resolve code from this checkout.
    env["PYTHONPATH"] = os.pathsep.join(
        entry for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != ROOT)
    result = subprocess.run([sys.executable, "-m", "piper.train", "--help"],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout


def test_esmc_download_uses_pinned_revision_and_preserves_manifest(tmp_path, monkeypatch):
    payload = b"synthetic weights"
    checksum = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(download, "ESMC_SHA256", checksum)
    calls = []

    def fetch(repo, filename, *, revision, local_dir):
        calls.append((repo, filename, revision))
        target = local_dir / "download.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    monkeypatch.setattr("requests.get", lambda *args, **kwargs:
                        SimpleNamespace(text="Synthetic model card", raise_for_status=lambda: None))
    download.download_esmc(tmp_path)
    download.download_esmc(tmp_path)
    assert calls == [(download.ESMC_REPO, download.ESMC_FILENAME, download.ESMC_REVISION)]
    assert (tmp_path / "weights.pth").read_bytes() == payload
    assert not (tmp_path / ".download_cache").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == dict(repo=download.ESMC_REPO, revision=download.ESMC_REVISION,
                           source_file=download.ESMC_FILENAME, sha256=checksum,
                           filename="weights.pth", d_model=960, n_heads=15,
                           n_layers=30, software="esm==3.2.3")


def test_esmc_corrupt_existing_weight_fails_before_network(tmp_path, monkeypatch):
    (tmp_path / "weights.pth").write_bytes(b"corrupt")

    def unexpected_network(*args, **kwargs):
        raise AssertionError("Network must not be used for a corrupt existing weight")

    monkeypatch.setattr("requests.get", unexpected_network)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", unexpected_network)
    with pytest.raises(RuntimeError, match="Existing weights"):
        download.download_esmc(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_official_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    def fetch(repo, *, repo_type, revision, local_dir, allow_patterns, max_workers):
        assert (repo, repo_type, revision) == download.OFFICIAL_SOURCES[0]
        assert allow_patterns == ["*.json", "*.py", "*.md", "*.safetensors", "data/*.parquet"]
        local_dir.mkdir()
        (local_dir / "file.parquet").write_bytes(b"corrupt")

    item = SimpleNamespace(path="file.parquet", lfs=SimpleNamespace(sha256="0" * 64))
    monkeypatch.setattr("huggingface_hub.snapshot_download", fetch)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda:
                        SimpleNamespace(list_repo_tree=lambda *args, **kwargs: [item]))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download.download_official(tmp_path)
    assert not (tmp_path / "verified_downloads.json").exists()
