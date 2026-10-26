"""Standalone inference using pinned, locally copied model source."""
import importlib.metadata
import json
import subprocess
import sys
import time

from .common import (ROOT, REPOSITORY, activate_snapshot, check_hash, local_path,
                     sha256, verify_prepared, write_json)


def gpu_available(config):
    # This study never pauses, kills or otherwise manages production training.
    processes = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    free = int(subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if processes or free < config["minimum_free_gpu_mib"]:
        raise RuntimeError(f"GPU busy: compute PIDs={processes or 'none'}, free={free} MiB. "
                           "Research inference was not started; training was not touched.")
    return free


def verify_model_sources(directory):
    for name, module in list(sys.modules.items()):
        if name == "flashppi" or name.startswith("flashppi."):
            filename = getattr(module, "__file__", None)
            if filename is not None and not local_path(filename).is_relative_to(directory):
                raise RuntimeError(f"External production import detected: {name} {filename}")


def preflight(config, name):
    import torch
    from safetensors import safe_open

    manifest = verify_prepared(config)
    directory = activate_snapshot(name, manifest)
    from flashppi.research_model import load_checkpoint  # noqa: F401; isolated snapshot only
    verify_model_sources(directory)
    checkpoint = REPOSITORY / config["checkpoints"][name]
    gate_stats = {}
    with safe_open(str(checkpoint / "trainable.safetensors"), framework="pt", device="cpu") as reader:
        for branch in ("q", "k"):
            prefix = f"head_{branch}.residue_pool."
            names = [key for key in reader.keys() if key.startswith(prefix)]
            if len(names) != 6:
                raise ValueError(f"Expected six pooling gate tensors for {branch}, found {names}")
            for key in names:
                value = reader.get_tensor(key).float()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite checkpoint gate tensor: {key}")
                gate_stats[key] = dict(shape=list(value.shape), l2=float(value.norm()),
                                       max_abs=float(value.abs().max()))
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "numpy", "pyarrow", "scikit-learn", "safetensors", "accelerate")}
    report = dict(status="cpu_preflight_passed", checkpoint=name, step=manifest["checkpoints"][name]["step"],
                  python=sys.version, packages=versions, gate_tensors=gate_stats,
                  model_loaded=False, gpu_inference_tested=False,
                  note="Import and gate-parameter checks only; biological conclusions require extraction")
    write_json(f"manifests/preflight_{name}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "gate_tensors"}, indent=2))


def extract(config, name):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from .gates import extract_branch
    from .panel import write_table

    manifest = verify_prepared(config)
    free_gpu_mib = gpu_available(config)
    directory = activate_snapshot(name, manifest)
    from flashppi.research_model import load_checkpoint
    from flashppi.finetune import clear_rotary_cache
    from flashppi.lora import merged_for_inference
    verify_model_sources(directory)
    checkpoint = REPOSITORY / config["checkpoints"][name]
    # Pin pretrained weights and dynamic Python code as well as the trainable
    # checkpoint. Ignore training-data paths: no training dataset is loaded.
    source_manifest = json.loads((checkpoint / "manifest.json").read_text())
    for path, digest in source_manifest["input_hashes"].items():
        if str(REPOSITORY / "artifacts") + "/" in path:
            check_hash(path, digest)
    output = local_path(f"results/{name}")
    output.mkdir(parents=True, exist_ok=False)
    sequences = pq.read_table(local_path("data/sequences.parquet")).to_pylist()
    started = time.monotonic()
    status = dict(status="loading", checkpoint=name, total_sequences=len(sequences), completed=0)
    write_json(output / "status.json", status)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        model, tokenizer, _ = load_checkpoint(checkpoint, device="cuda:0")
        model.eval().requires_grad_(False)
        clear_rotary_cache(model)
        gate_rows, vector_rows = [], []
        with torch.inference_mode(), merged_for_inference(model), torch.autocast("cuda", dtype=torch.float16):
            for index, item in enumerate(sequences):
                sequence, sid = item["sequence"], item["sequence_id"]
                tokens = tokenizer([sequence], padding=True, truncation=False, return_tensors="pt").to("cuda:0")
                if tokens.attention_mask.sum(1).tolist() != [len(sequence)]:
                    raise ValueError("Tokenizer/residue alignment mismatch")
                if not tokens.attention_mask[0, :len(sequence)].bool().all():
                    raise ValueError("Expected right padding without special tokens")
                encoded = model.encode_protein(tokens.input_ids, tokens.attention_mask)
                if not torch.isfinite(encoded[tokens.attention_mask.bool()]).all():
                    raise FloatingPointError("Nonfinite encoded residues")
                for branch in ("q", "k"):
                    raw, normalized, relative, vectors = extract_branch(
                        getattr(model, f"head_{branch}"), encoded, tokens.attention_mask, sid, branch,
                        config["seed"], config["gate_shuffles"], item["for_interaction"])
                    for position, aa in enumerate(sequence):
                        gate_rows.append(dict(sequence_id=sid, branch=branch, position=position + 1, aa=aa,
                                              raw_gate=float(raw[position]), normalized_weight=float(normalized[position]),
                                              relative_weight=float(relative[position])))
                    for variant, vector in vectors.items():
                        vector_rows.append(dict(sequence_id=sid, branch=branch, variant=variant, vector=vector.tolist()))
                del encoded, tokens
                status.update(status="running", completed=index + 1, seconds=time.monotonic() - started)
                write_json(output / "status.json", status)
                if index % 10 == 0:
                    print(f"{name}: {index + 1}/{len(sequences)} full sequences", flush=True)
        files = [write_table(output / "gates.parquet", gate_rows), write_table(output / "vectors.parquet", vector_rows)]
        verify_model_sources(directory)
        provenance = dict(checkpoint=name, step=manifest["checkpoints"][name]["step"],
                          prepared_sha256=sha256(ROOT / "manifests/prepared.json"),
                          trainable_sha256=manifest["checkpoints"][name]["trainable_sha256"],
                          study_sha256=sha256(ROOT / "study.json"), free_gpu_mib_at_start=free_gpu_mib,
                          evaluation="Full sequence, batch 1, FP16 autocast; fixed embeddings per intervention",
                          source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "clip_gate").glob("*.py")},
                          files={p.name: sha256(p) for p in files})
        write_json(output / "provenance.json", provenance)
        status.update(status="complete", seconds=time.monotonic() - started)
        write_json(output / "status.json", status)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "status.json", status)
        raise
