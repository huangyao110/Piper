"""Portable, synchronous multi-GPU fine-tuning with optional frozen ESM C.

Launch with torchrun. This is a NEW experiment, not a bitwise continuation of
the original single-GPU run: global CLIP logits use FP32 and crop/replay seeds
are attached to global microbatches. Every validation checkpoint is retained.
"""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from .cropping import valid_contact_counts
from .dataset import ClusterSampler, DataCollatorForPPI, build_interaction_keys
from .finetune import sha256, sample_window
from .lora import save_trainable, load_trainable
from .research_distributed import (GlobalGradientCache, assigned_chunks, chunk_seed,
                                   world_rank, distributed, broadcast_trainable)
from .research_eval import atomic_json, evaluate_validation, evaluate_test
from .research_model import build_model
from .sequence_masking import SequenceMaskingCollator
from .patch_contact import PatchTargetCollator, PatchSpec
from .mixed_sampling import BalancedKindSampler
from .indexed_training import (IndexedTrainingPopulation, build_indexed_sampler,
                               verify_indexed_inputs, validate_continuation, prune_checkpoints)

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "047edbe0570eca424ea835e0a460248d2aa5077d95a024d149ac22ab78a44a49"
TEST_SHA = "07dace691cac67561d264fbbcd8deb862ad6356998b129ae57dee7d983de7f6c"
METRICS = {"clip": "clip_ap", "contact": "contact_precision_at_k", "interaction": "contact_score_ap"}
SOURCE_FILES = ["flashppi/" + x for x in ["research_train.py", "research_model.py", "research_distributed.py",
                "research_eval.py", "finetune.py", "lora.py", "loss.py", "model.py", "dataset.py", "cropping.py",
                "sequence_masking.py", "patch_contact.py", "mixed_sampling.py", "glm_initialization.py", "layers.py",
                "coverage_migration.py", "negative_sampling.py", "gated_modules.py", "frozen_prott5.py",
                "gated_warm_start.py", "residue_warm_start.py", "frozen_embedding_cache.py", "full_finetune.py",
                "indexed_data.py", "indexed_training.py", "data_protocol.py", "frozen_prefix_cache.py"]]
SOURCE_FILES += ["tools/evaluate_research_checkpoint.py", "tools/check_research_replay.py",
                 "tools/evaluate_official_ecoli.py", "configs_data/conglab_ecoli_filter.lock.json",
                 "configs_data/conglab_distilled_ecoli_rf2ppi_v3.lock.json"]


def barrier():
    if distributed():
        dist.barrier()


def verify_inputs(config):
    split = json.loads((Path(config["train"]).parent / "manifest.json").read_text())
    if config.get("initialization", {}).get("source") == "original_glm2":
        base = Path(config["initialization"]["backbone"])
        manifest = json.loads((base / "manifest.json").read_text())
        expected = {str(base / p): h for p, h in manifest["files"].items()}
        expected[str(base / "manifest.json")] = sha256(base / "manifest.json")
        for path in Path(config["model"]).iterdir():
            if path.suffix in (".py", ".json"):
                expected[str(path)] = sha256(path)
    else:
        expected = {config["model"] + "/model.safetensors": BASE_SHA}
    if config.get('indexed_split'):
        expected.update(verify_indexed_inputs(config))
    else:
        for key in ("train", "validation_structures"):
            path = Path(config[key])
            expected[str(path)] = split["outputs"][path.name]["sha256"]
        val = json.loads((Path(config["validation_interactions"]).parent / "manifest.json").read_text())
        expected[config["validation_interactions"]] = val["benchmark_sha256"]
        expected[config["test"]] = TEST_SHA
    fusion_source = config.get('fusion', {}).get('source')
    if fusion_source in ('esmc', 'esmc_prott5'):
        path = Path(config['fusion']['esmc_model' if fusion_source == 'esmc_prott5' else 'model'])
        esm = json.loads((path / "manifest.json").read_text())
        if esm["sha256"] != "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea":
            raise ValueError("Expected pinned ESM C 300M checkpoint")
        expected[str(path / esm["filename"])] = esm["sha256"]
        expected[str(path / 'manifest.json')] = sha256(path / 'manifest.json')
    if fusion_source == 'esmc_prott5':
        path = Path(config['fusion']['prott5_model'])
        t5 = json.loads((path / 'manifest.json').read_text())
        if (t5.get('repository') != 'Rostlab/prot_t5_xl_half_uniref50-enc'
                or t5.get('revision') != '94a6abc029ae13029317b140b7424e012bf8dfbf'
                or not t5.get('files')):
            raise ValueError('Expected pinned ProtT5 XL encoder-only checkpoint')
        for filename, checksum in t5['files'].items():
            relative = Path(filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('ProtT5 manifest file must be a relative artifact path')
            expected[str(path / relative)] = checksum
        expected[str(path / 'manifest.json')] = sha256(path / 'manifest.json')
    for path, value in expected.items():
        if sha256(path) != value:
            raise ValueError("Frozen input checksum mismatch: " + path)
    return expected


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def build_optimizer(model, config):
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if config.get('mode') == 'full_last_layers':
        backbone = [p for n,p in trainable if n.startswith('plm.')]
        heads = [p for n,p in trainable if not n.startswith('plm.')]
        rate = config['backbone_learning_rate']
        if not backbone or not heads or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Full-block mode requires backbone/heads and a positive backbone rate')
        if any('.lora_' in n for n,_ in trainable):
            raise ValueError('LoRA parameters remain in full-block mode')
        return torch.optim.AdamW([
            dict(params=backbone, lr=rate, base_lr=rate, name='backbone'),
            dict(params=heads, lr=config['learning_rate'], base_lr=config['learning_rate'], name='heads')],
            weight_decay=config['weight_decay'])
    if config.get('lora_learning_rate') is None:
        return torch.optim.AdamW([p for _,p in trainable],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    adapters = [p for n,p in trainable if '.lora_' in n]
    heads = [p for n,p in trainable if '.lora_' not in n]
    if not adapters or not heads:
        raise ValueError('Separate adapter/head rates require both parameter groups')
    return torch.optim.AdamW([
        dict(params=adapters,lr=config['lora_learning_rate'],base_lr=config['lora_learning_rate'],name='lora'),
        dict(params=heads,lr=config['learning_rate'],base_lr=config['learning_rate'],name='heads')],
        weight_decay=config['weight_decay'])


def set_step_learning_rate(optimizer, config, step):
    factor = min(1.0, (step + 1) / config['warmup_steps'])
    if config.get('learning_rate_schedule') == 'cosine' and step + 1 > config['warmup_steps']:
        progress = min(1., (step + 1 - config['warmup_steps']) / max(1,config['steps']-config['warmup_steps']))
        minimum = config.get('minimum_lr_ratio', 0.0)
        if not math.isfinite(minimum) or not 0 <= minimum <= 1:
            raise ValueError('minimum_lr_ratio must be between zero and one')
        factor = minimum + (1 - minimum) * .5*(1+math.cos(math.pi*progress))
    for group in optimizer.param_groups:
        group['lr'] = group.get('base_lr',config['learning_rate'])*factor


def evaluation_due(config, step, epoch_end=False):
    """Test monitoring has a fixed cadence, independent of epoch checkpoints."""
    if config.get('smoke'):
        return False
    final = step == config['steps']
    if config.get('evaluation_split', 'validation') == 'test':
        return step > 0 and (step % config['eval_interval'] == 0 or final)
    return ((step - config.get('phase_start_step', 0)) % config['eval_interval'] == 0
            or epoch_end or final or step in config.get('early_eval_steps', []))


def _offload_optimizer_state(optimizer):
    locations = []
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                locations.append((state, key, value.device))
                state[key] = value.cpu()
    return locations


@contextmanager
def test_monitoring_context(model, optimizer):
    """Give full-length evaluation GPU space, then restore Adam and all RNGs."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    was_training = model.training
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    locations = []
    try:
        with torch.random.fork_rng(devices=devices):
            locations = _offload_optimizer_state(optimizer)
            if devices:
                torch.cuda.empty_cache()
            try:
                yield
            finally:
                if devices:
                    torch.cuda.empty_cache()
                for state, key, device in locations:
                    state[key] = state[key].to(device)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        model.train(was_training)


def monitor_test(model, tokenizer, optimizer, config, output, checkpoint, step, input_hashes):
    """Record test results only; never select a checkpoint or adjust learning rates."""
    destination = output / f'test_step_{step:06d}'
    atomic_json(output / 'status.json', dict(status='testing', step=step, total_steps=config['steps']))
    optimizer.zero_grad(set_to_none=True)
    with test_monitoring_context(model, optimizer):
        metrics = evaluate_test(model, tokenizer, config, destination)
    atomic_json(destination / 'manifest.json', dict(
        step=step, checkpoint=str(checkpoint.resolve()),
        trainable_sha256=sha256(checkpoint / 'trainable.safetensors'), input_hashes=input_hashes,
        orientation='published', sequence_length='full', filtering='none',
        purpose='fixed-schedule monitoring; no model selection or early stopping'))
    with (output / 'test_history.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(step=step, **metrics)) + '\n')
    print('TEST', json.dumps(dict(step=step, **metrics)), flush=True)
    return metrics


def save_checkpoint(output, model, optimizer, config, step, best, input_hashes, source_hashes, device):
    world, rank = world_rank()
    rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
               python=random.getstate(), numpy=np.random.get_state())
    states = [None] * world
    if distributed():
        dist.all_gather_object(states, rng)
    else:
        states = [rng]
    relative = f"checkpoints/step_{step:06d}"
    destination = output / relative
    if rank == 0:
        temporary = destination.with_name(destination.name + ".incomplete")
        temporary.mkdir(parents=True, exist_ok=False)
        atomic_json(temporary / "config.json", config)
        save_trainable(model, temporary / "trainable.safetensors")
        torch.save(dict(step=step, config=config, world_size=world,
                        optimizer=cpu_state(optimizer.state_dict()), rng=states,
                        best=best, input_hashes=input_hashes, source_hashes=source_hashes), temporary / "training_state.pt")
        atomic_json(temporary / "manifest.json", dict(step=step, format="trainable-only; pinned bases required",
                    trainable_sha256=sha256(temporary / "trainable.safetensors"),
                    training_state_sha256=sha256(temporary / "training_state.pt"), input_hashes=input_hashes))
        temporary.rename(destination)
        atomic_json(output / "latest.json", dict(step=step, checkpoint=relative))
    barrier()
    return relative


def load_population(config):
    if config.get('indexed_split'):
        table = IndexedTrainingPopulation(config['indexed_split'])
        if len(table) != config['eligible_train_rows']:
            raise ValueError('Indexed training population changed')
        # All retained training positives participate, never validation/test labels.
        pairs = pq.read_table(Path(config['indexed_split']) / 'known_positive_cluster_pairs.parquet')
        keys, _ = build_interaction_keys(pairs['cluster1'].to_numpy().astype(np.int64),
                                         pairs['cluster2'].to_numpy().astype(np.int64),
                                         np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                                         multiplier=config['cluster_multiplier'])
        return table, keys, len(table)
    columns = ["seq1", "seq2", "contacts", "seq1_cluster", "seq2_cluster", "valid1", "valid2", "pair_id"]
    if config.get("balanced_ppi_ddi"):
        columns.append("kind")
    table = pq.read_table(config["train"], columns=columns)
    keys, _ = build_interaction_keys(table["seq1_cluster"].to_numpy(), table["seq2_cluster"].to_numpy(),
                                     np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                                     multiplier=config.get("cluster_multiplier", 1_000_000))
    keep = np.flatnonzero(valid_contact_counts(table) > 0)
    if len(keep) != config["eligible_train_rows"]:
        raise ValueError("Eligible training population changed")
    return table.take(keep), keys, len(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--resume", type=Path)
    restore.add_argument("--continue-from", type=Path,
                         help="Start an explicit indexed-data phase, preserving model and Adam state")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-global-batch", type=int, default=16)
    parser.add_argument("--smoke-length", type=int, default=64)
    parser.add_argument("--stress-smoke", action="store_true", help="Sample only pairs with both chains >= smoke length")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--sequence-mask-probability", type=float,
                        help="Training-only residue masking; 0 disables, 0.15 is the initial experiment")
    parser.add_argument("--stop-after", type=int, help="Save and exit after this step; keep total schedule for exact resume tests")
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    args = parser.parse_args()
    if args.stress_smoke and not args.smoke_steps:
        parser.error("--stress-smoke requires --smoke-steps")
    os.chdir(ROOT)
    config = json.loads(args.config.read_text())
    if config.get('indexed_split') and args.smoke_steps:
        parser.error('Indexed continuation smoke uses --stop-after to retain the actual batch/config')
    if config.get("frozen_embedding_cache", "off") not in ("off", "cpu"):
        parser.error("frozen_embedding_cache must be 'off' or 'cpu'")
    if config.get('frozen_prefix_cache', 'off') != 'off' and config['mode'] != 'full_last_layers':
        parser.error('Frozen prefix reuse is supported only for full_last_layers training')
    metrics_to_select = dict(METRICS)
    if config.get("contact_patch") and config.get("patch_evaluation"):
        metrics_to_select["contact"] = "patch_average_precision"
    if args.sequence_mask_probability is not None:
        config["sequence_masking"] = {"probability": args.sequence_mask_probability}
    mask_probability = float(config.get("sequence_masking", {}).get("probability", 0.0))
    if not math.isfinite(mask_probability) or not 0 <= mask_probability <= 1:
        parser.error("Sequence mask probability must be between 0 and 1")
    if args.microbatch is not None:
        config["chunk_size"] = args.microbatch
    if args.smoke_steps:
        config.update(steps=args.smoke_steps, batch_size=args.smoke_global_batch,
                      max_len=args.smoke_length, epochs=None, smoke=True, stress_smoke=args.stress_smoke)
    else:
        config["smoke"] = False
    if min(config["batch_size"], config["chunk_size"], config["steps"]) < 1:
        raise ValueError("Batch, microbatch and steps must be positive")
    if config.get('evaluation_split', 'validation') not in ('validation', 'test'):
        raise ValueError('evaluation_split must be validation or test')
    if min(config['warmup_steps'], config['eval_interval'], config['recovery_interval']) < 1:
        raise ValueError('Warmup, evaluation and recovery intervals must be positive')
    test_only = config.get('evaluation_split') == 'test'
    if test_only and (config.get('validation_structures') or config.get('validation_interactions')):
        raise ValueError('Test-only monitoring must not also configure validation inputs')
    if config.get("contact_coverage"):
        settings = config["contact_coverage"]
        if (set(settings) != {"loss_weight", "target", "supervision", "normalization"}
                or settings["target"] != "directional_participating_residue_fraction"
                or settings["supervision"] != "complete_observed_positive_patches"
                or settings["normalization"] != "global_eligible_structure_mean"
                or not math.isfinite(settings["loss_weight"]) or settings["loss_weight"] <= 0):
            raise ValueError("Invalid coverage objective configuration")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
    world, rank = world_rank()
    torch.set_num_threads(config.get("cpu_threads_per_rank", 4))
    torch.use_deterministic_algorithms(config.get("deterministic", True))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    input_hashes = verify_inputs(config) if rank == 0 else None
    hashes = [input_hashes]
    if distributed():
        dist.broadcast_object_list(hashes, src=0)
    input_hashes = hashes[0]
    source_hashes = {p: sha256(ROOT / p) for p in SOURCE_FILES}
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "config.json", config)
        atomic_json(args.output / "input_hashes.json", input_hashes)
        code = args.output / "code"
        for source in SOURCE_FILES:
            target = code / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, target)
        atomic_json(args.output / "source_sha256.json", source_hashes)
        import importlib.metadata as metadata
        versions = {}
        for package in ("torch", "transformers", "accelerate", "esm", "numpy", "pyarrow"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        atomic_json(args.output / "environment.json", dict(versions=versions, world_size=world,
                    gpu=torch.cuda.get_device_name(device), precision="BF16 train; FP32 CLIP logits; FP16 eval",
                    algorithm="microbatch replay + SUM gradient synchronization", memory_fraction=args.memory_fraction))
    barrier()
    table, keys, original_rows = load_population(config)
    architecture_config = dict(config)
    if args.resume or args.continue_from:
        architecture_config.pop('warm_start_checkpoint', None)
    model, tokenizer, loading = build_model(architecture_config, keys, device)
    if rank == 0:
        atomic_json(args.output / 'initialization.json', loading)
    broadcast_trainable(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, config)
    cache = GlobalGradientCache(model, device, config["clip_loss_weight"], config["contact_loss_weight"],
                                frozen_embedding_cache=config.get("frozen_embedding_cache", "off"),
                                frozen_length_bucket_size=config.get('frozen_length_bucket_size', 0),
                                frozen_prefix_cache=config.get('frozen_prefix_cache', 'off'))
    collator = DataCollatorForPPI(tokenizer, config["max_len"], True, False, config["crop"],
                                sequence_identity=bool(config.get("contact_negative_sampling")))
    collator = SequenceMaskingCollator(collator, mask_probability)
    if config.get("contact_patch") is not None:
        collator = PatchTargetCollator(collator, PatchSpec(**config["contact_patch"]),
                                      coverage=bool(config.get("contact_coverage")))
    if config.get('indexed_split'):
        sampler = build_indexed_sampler(config, table)
    elif config.get("balanced_ppi_ddi"):
        sampler = BalancedKindSampler(table["seq1_cluster"].to_numpy(), table["seq2_cluster"].to_numpy(),
                                      table["kind"].to_numpy(), config["batch_size"], config["seed"],
                                      block_steps=config.get("sampling_block_steps"))
    else:
        sampler = ClusterSampler(table["seq1_cluster"].to_numpy(), table["seq2_cluster"].to_numpy(),
                                 ddi_sampling_prob=0, seed=config["seed"])
    epochs = config.get("epochs")
    epoch_steps = math.ceil(len(table) / config["batch_size"])
    if epochs and config["steps"] != epochs * epoch_steps:
        raise ValueError("Step budget differs from configured complete epochs")
    phase_start = int(config.get('phase_start_step', 0))
    sampler.num_samples = ((config["steps"] - phase_start) * config["batch_size"] if config.get("balanced_ppi_ddi")
                           else epochs * len(table) if epochs else config["steps"] * config["batch_size"])
    sampled = list(sampler)
    if args.stress_smoke:
        import pyarrow.compute as pc
        long = np.flatnonzero((pc.utf8_length(table["seq1"]).to_numpy() >= config["max_len"])
                             & (pc.utf8_length(table["seq2"]).to_numpy() >= config["max_len"]))
        if not len(long):
            raise ValueError("No full-length pairs for stress smoke")
        sampled = np.random.default_rng(config["seed"]).choice(long, len(sampled), replace=True).tolist()
    start, best = phase_start, {}
    checkpoint = args.resume or args.continue_from
    if phase_start and checkpoint is None:
        raise ValueError('A continuation phase requires its predecessor or a resume checkpoint')
    if checkpoint:
        checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text())
        for filename, key in (("trainable.safetensors", "trainable_sha256"), ("training_state.pt", "training_state_sha256")):
            if sha256(checkpoint / filename) != checkpoint_manifest[key]:
                raise ValueError("Resume checkpoint checksum mismatch: " + filename)
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if state['config'] != json.loads((checkpoint / 'config.json').read_text()) or state['step'] != checkpoint_manifest['step']:
            raise ValueError('Checkpoint config/step metadata disagree')
        if args.resume and (state["config"] != config or state["input_hashes"] != input_hashes
                or state["world_size"] != world or state["source_hashes"] != source_hashes):
            raise ValueError("Resume requires identical config, pinned inputs, source and world size")
        previous_run = checkpoint.resolve().parent.parent
        if args.continue_from:
            changes = validate_continuation(state['config'], config, state['step'])
            if state['world_size'] != world or state['input_hashes'] != checkpoint_manifest['input_hashes']:
                raise ValueError('Continuation world size/input provenance mismatch')
            for path in state['input_hashes'].keys() & input_hashes.keys():
                if state['input_hashes'][path] != input_hashes[path]:
                    raise ValueError('An inherited frozen input changed: ' + path)
            old_names = list(json.loads((previous_run / 'trainable_parameters.json').read_text()))
            if old_names != [n for n, p in model.named_parameters() if p.requires_grad]:
                raise ValueError('Continuation optimizer parameter order changed')
        load_trainable(model, checkpoint / "trainable.safetensors")
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best"] if args.resume else {}
        for record in best.values():
            old_checkpoint = (previous_run / record["checkpoint"]).resolve()
            record["checkpoint"] = os.path.relpath(old_checkpoint, args.output.resolve())
        rng = state["rng"][rank]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        if rank == 0:
            atomic_json(args.output / ("resume.json" if args.resume else 'continuation.json'),
                        dict(source=str(checkpoint.resolve()), step=start,
                        config_changes=changes if args.continue_from else [],
                        optimizer_restored=True, rng_restored=True, warmup_restarted=False,
                        old_best=state['best'],
                        previous_best=best))
            for name, record in best.items():
                atomic_json(args.output / f"best_{name}.json", record)
        del state
    if start > config['steps'] or (start == config['steps'] and not (test_only and args.resume)):
        raise ValueError("Checkpoint already reached training budget")
    if rank == 0:
        sampling_details = {}
        if config.get('indexed_split'):
            kinds, sources = table['kind'].to_numpy(), table['structure_source'].to_numpy()
            weights = sampler.effective_weights
            sampling_details = dict(ppi_sequence_clusters=sampler.ppi_sequence_clusters,
                                    expected_source_fraction={})
            if hasattr(sampler, 'ddi_sequence_clusters'):
                sampling_details.update(ddi_sequence_clusters=sampler.ddi_sequence_clusters,
                                        source_group_sequence_clusters=sampler.group_cluster_counts,
                                        source_group_draws_per_batch=sampler.counts)
            else:
                sampling_details['ddi_structure_clusters'] = sampler.ddi_structure_clusters
            for kind in ('ppi', 'ddi'):
                mask = kinds == kind
                total = weights[mask].sum()
                for source in sorted(set(sources[mask])):
                    sampling_details['expected_source_fraction'][kind + '/' + source] = float(
                        0.5 * weights[mask & (sources == source)].sum() / total)
        atomic_json(args.output / "training_population.json", dict(original_rows=original_rows,
                    eligible_rows=len(table), known_positive_mask="all retained training records; no held-out labels",
                    positive_draws=len(sampled), replacement=True,
                    epoch_steps=epoch_steps if epochs else None, epochs=epochs,
                    indexed_sampling=config.get('indexed_sampling'), phase_start_step=phase_start,
                    **sampling_details))
        import hashlib
        atomic_json(args.output / "sample_schedule.json", dict(
                    sha256=hashlib.sha256(np.asarray(sampled, dtype="<i8").tobytes()).hexdigest(),
                    draws=len(sampled), seed=config["seed"],
                    sampling_block_steps=config.get("sampling_block_steps"),
                    format="int64 little-endian indices in the eligible population"))
        atomic_json(args.output / "trainable_parameters.json", {n: p.numel() for n, p in model.named_parameters() if p.requires_grad})
    stop_request = {"signal": None}

    def request_stop(signum, frame):
        stop_request["signal"] = signum
        print("STOP requested; finishing the current optimizer step and saving", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_saved = None
    if test_only and ((not args.resume and config.get('evaluate_initial') and not config['smoke'])
                      or (args.resume and evaluation_due(config, start))):
        # A resume at a test boundary re-runs monitoring before any further
        # update, including when the preceding attempt failed during test.
        if not args.resume:
            last_saved = save_checkpoint(args.output, model, optimizer, config, start, best,
                                         input_hashes, source_hashes, device)
        if rank == 0:
            monitor_test(model, tokenizer, optimizer, config, args.output,
                         args.resume if args.resume else args.output / last_saved, start, input_hashes)
        barrier()
    if not test_only and not args.resume and config.get("evaluate_initial") and not config["smoke"]:
        if rank == 0:
            atomic_json(args.output / "status.json", dict(status="initial_validation", step=start))
            metrics = evaluate_validation(model, tokenizer, config, args.output / f"val_step_{start:06d}")
            for name, key in metrics_to_select.items():
                best[name] = dict(step=start, value=metrics[key], metric=key, checkpoint=f"checkpoints/step_{start:06d}")
            print("INITIAL_VALIDATION", json.dumps(metrics), flush=True)
        values = [best]
        if distributed():
            dist.broadcast_object_list(values, src=0)
        best = values[0]
        last_saved = save_checkpoint(args.output, model, optimizer, config, start, best,
                                     input_hashes, source_hashes, device)
        if rank == 0:
            for name, record in best.items():
                atomic_json(args.output / f"best_{name}.json", record)
    if start == config['steps']:
        # The final update was saved before test; finish a failed/interrupted
        # final test without repeating optimizer updates.
        if rank == 0:
            atomic_json(args.output / 'final.json', dict(step=start, checkpoint=str(args.resume.resolve())))
            atomic_json(args.output / 'status.json', dict(status='complete', step=start,
                        total_steps=config['steps'], checkpoint=str(args.resume.resolve())))
        barrier()
        if distributed():
            dist.destroy_process_group()
        return
    for step in range(start, config["steps"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        load_tick = time.monotonic()
        left, right = sample_window(step - phase_start, config["batch_size"],
                                    len(table) if epochs and not config.get("balanced_ppi_ddi") else None)
        # Group source reads across the entire optimizer batch; preserve the
        # sampled order before assigning microbatches and their replay seeds.
        step_rows = table.take(sampled[left:right]).to_pylist() if config.get('indexed_split') else None
        selection = list(range(right - left)) if step_rows is not None else sampled[left:right]
        chunks = assigned_chunks(selection, config["chunk_size"], world, rank)
        batches = []
        for index, selected in chunks:
            seed = chunk_seed(config["seed"], step, index)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                batch = collator([step_rows[i] for i in selected] if step_rows is not None
                                 else table.take(selected).to_pylist())
            batch["replay_seed"] = seed
            batches.append(batch)
        data_seconds = time.monotonic() - load_tick
        set_step_learning_rate(optimizer, config, step)
        torch.cuda.synchronize(device)
        tick = time.monotonic()
        result = cache.step(batches, right - left)
        if config['mode'] == 'full_last_layers' and step == start:
            from .full_finetune import check_full_block_gradients
            result['backbone_gradient_norms'] = check_full_block_gradients(
                model, config['full_finetune']['last_layers'])
        norm = torch.nn.utils.clip_grad_norm_(trainable, config["max_grad_norm"], error_if_nonfinite=True)
        if not np.isfinite([result["clip_loss"], result["contact_loss"], result.get("coverage_loss", 0), float(norm)]).all():
            raise FloatingPointError("Nonfinite training loss/gradient")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError("Frozen backbone acquired a gradient")
        optimizer.step()
        torch.cuda.synchronize(device)
        result.update(step=step + 1, positive_draws=right - left, world_size=world, training_mode=config['mode'],
                      phase_step=step + 1 - phase_start, data_seconds=data_seconds,
                      learning_rates={g.get('name',str(i)):g['lr'] for i,g in enumerate(optimizer.param_groups)},
                      grad_norm=float(norm), seconds=time.monotonic() - tick,
                      allocated_gb=torch.cuda.max_memory_allocated(device) / 1e9,
                      reserved_gb=torch.cuda.max_memory_reserved(device) / 1e9)
        if mask_probability:
            counts = sum((b["sequence_masking_counts"] for b in batches),
                         torch.zeros(2, dtype=torch.long)).to(device)
            if distributed():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            masked, eligible = counts.tolist()
            result.update(sequence_masked_residues=masked, sequence_mask_eligible_residues=eligible,
                          sequence_mask_fraction=masked / eligible if eligible else 0.0)
        if model.fusion is not None:
            if model.fusion_source == 'esmc_prott5':
                result['fusion_alpha_esmc'] = float(model.fusion.alpha_esmc.detach())
                result['fusion_alpha_prott5'] = float(model.fusion.alpha_prott5.detach())
            else:
                result["fusion_alpha"] = float(model.fusion.alpha.detach())
        if getattr(model.contact_head, 'patch_gate', None) is not None:
            result['patch_beta'] = float(model.contact_head.patch_gate.beta.detach())
        if getattr(model.contact_head, 'residue_gate', None) is not None:
            result['residue_beta'] = float(model.contact_head.residue_gate.beta.detach())
        if config.get("balanced_ppi_ddi"):
            kinds = table["kind"].take(sampled[left:right]).to_pylist()
            result.update(ppi_draws=kinds.count("ppi"), ddi_draws=kinds.count("ddi"))
            if result["ppi_draws"] != result["ddi_draws"]:
                raise AssertionError("PPI/DDI optimizer batch is not exactly 1:1")
        if step_rows is not None:
            from collections import Counter
            result['source_draws'] = dict(Counter(r['kind'] + '/' + r['structure_source'] for r in step_rows))
            result['legacy_draws'] = sum(r['legacy_train'] for r in step_rows)
        if rank == 0:
            with (args.output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            atomic_json(args.output / "status.json", dict(status="training", **result))
            print("TRAIN", json.dumps(result), flush=True)
        epoch_end = bool(epochs and (step + 1) % epoch_steps == 0)
        final = step + 1 == config["steps"]
        requested = torch.tensor(int(stop_request["signal"] is not None), device=device)
        if distributed():
            dist.all_reduce(requested, op=dist.ReduceOp.MAX)
        stopping = bool(requested) or (args.stop_after is not None and step + 1 >= args.stop_after)
        evaluate = evaluation_due(config, step + 1, epoch_end)
        validate = evaluate and not test_only
        improved = []
        if validate:
            barrier()
            if rank == 0:
                metrics = evaluate_validation(model, tokenizer, config, args.output / f"val_step_{step + 1:06d}")
                for name, key in metrics_to_select.items():
                    if metrics[key] > best.get(name, {}).get("value", -float("inf")):
                        best[name] = dict(step=step + 1, value=metrics[key], metric=key,
                                          checkpoint=f"checkpoints/step_{step + 1:06d}")
                        improved.append(name)
                print("VALIDATION", json.dumps(metrics), flush=True)
            values = [best]
            if distributed():
                dist.broadcast_object_list(values, src=0)
            best = values[0]
        if evaluate or epoch_end or final or stopping or (step + 1 - phase_start) % config["recovery_interval"] == 0:
            last_saved = save_checkpoint(args.output, model, optimizer, config, step + 1, best, input_hashes, source_hashes, device)
            if rank == 0:
                for name in improved:
                    atomic_json(args.output / f"best_{name}.json", best[name])
                if final:
                    atomic_json(args.output / "final.json", dict(step=step + 1, checkpoint=last_saved))
                if epoch_end:
                    atomic_json(args.output / f"epoch_{(step + 1) // epoch_steps:02d}.json", dict(step=step + 1, checkpoint=last_saved))
                if config.get('checkpoint_retention'):
                    keep_steps = [phase_start]
                    if epochs and config['checkpoint_retention'].get('keep_epochs'):
                        keep_steps += list(range(epoch_steps, step + 2, epoch_steps))
                    prune_checkpoints(args.output, best, config['checkpoint_retention']['keep_last'],
                                      keep_steps=keep_steps)
        if evaluate and test_only:
            barrier()
            if rank == 0:
                monitor_test(model, tokenizer, optimizer, config, args.output,
                             args.output / last_saved, step + 1, input_hashes)
            barrier()
        if stopping:
            break
    if rank == 0:
        atomic_json(args.output / "status.json", dict(status="complete" if final else "stopped_at_checkpoint",
                    step=step + 1, total_steps=config["steps"], checkpoint=last_saved))
    barrier()
    if distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
