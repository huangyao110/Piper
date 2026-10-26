"""Full final-block fine-tuning and explicit migration from trained Q/V LoRA."""
import copy
import math

import torch
from torch import nn

from .lora import QVLoRALinear


def selected_blocks(model, count):
    layers = model.plm.encoder.layers
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(layers):
        raise ValueError("Invalid full fine-tuning layer count")
    return list(enumerate(layers))[len(layers) - count:]


def configure_full_finetune(model, count):
    """Freeze the embedding/prefix; train every parameter of the final blocks."""
    selected = selected_blocks(model, count)
    if any(isinstance(module, QVLoRALinear) for module in model.plm.modules()):
        raise ValueError("Merge and remove LoRA before full-block fine-tuning")
    model.plm.requires_grad_(False)
    for _, layer in selected:
        layer.requires_grad_(True)
    model.plm.encoder.gradient_checkpointing = True
    return sum(p.numel() for p in model.plm.parameters() if p.requires_grad)


def merge_lora_and_unfreeze(model, count):
    """Merge in FP32 on CPU, retaining the frozen base's parameter names."""
    selected = selected_blocks(model, count)
    if any(not isinstance(layer.attention.wqkv, QVLoRALinear) for _, layer in selected):
        raise ValueError("Every selected block must contain its trained Q/V adapter")
    if sum(isinstance(m, QVLoRALinear) for m in model.modules()) != count:
        raise ValueError("LoRA outside the selected blocks is unsupported")
    for _, layer in selected:
        old = layer.attention.wqkv
        if not old.enabled:
            raise ValueError("Cannot migrate a temporarily disabled LoRA adapter")
        for parameter in old.parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError("Nonfinite LoRA parameter")
    for _, layer in selected:
        old = layer.attention.wqkv
        replacement = nn.Linear(old.in_features, old.out_features, bias=old.bias is not None,
                                device="meta", dtype=old.weight.dtype)
        replacement.weight = nn.Parameter(old.merged_weight_cpu().to(old.weight.device))
        if old.bias is not None:
            replacement.bias = nn.Parameter(old.bias.detach().clone())
        replacement.train(old.training)
        layer.attention.wqkv = replacement
    count_parameters = configure_full_finetune(model, count)
    # ReleasedTrainingModel uses this legacy flag for backbone autograd/train
    # mode. Full-mode construction also enables it, without installing LoRA.
    model.backbone_lora = True
    return count_parameters


def full_finetune_config(source):
    if source.get("mode") != "lora" or source.get("lora_learning_rate") is None:
        raise ValueError("Migration requires LoRA with separate backbone/head rates")
    result = copy.deepcopy(source)
    result["mode"] = "full_last_layers"
    result["full_finetune"] = {"last_layers": result.pop("lora")["last_layers"]}
    result["backbone_learning_rate"] = result.pop("lora_learning_rate")
    # A migrated checkpoint is self-contained relative to the pinned bases.
    result.pop("warm_start_checkpoint", None)
    result.pop("warm_start_transition", None)
    return result


def migrate_optimizer(optimizer, saved, old_named, new_named):
    """Keep head/fusion/gate Adam history by verified name; initialize blocks fresh.

    Low-rank factor moments cannot be mapped to moments of the dense weights.
    New backbone entries remain absent until their first optimizer update.
    """
    old_groups = {group["name"]: group for group in saved["param_groups"]}
    if len(saved["param_groups"]) != 2 or set(old_groups) != {"lora", "heads"}:
        raise ValueError("Expected exactly the old lora/heads optimizer groups")
    if not all((".lora_" in name) == name.startswith("plm.") for name in old_named):
        raise ValueError("Source backbone contains non-LoRA trainable parameters")
    if any(".lora_" in name for name in new_named):
        raise ValueError("Full-mode optimizer still contains LoRA")
    old_heads = [name for name in old_named if not name.startswith("plm.")]
    new_heads = [name for name in new_named if not name.startswith("plm.")]
    if set(old_heads) != set(new_heads):
        raise ValueError("Head/fusion/gate parameter names changed")
    previous = {}
    all_ids = []
    for group_name, group in old_groups.items():
        names = [name for name in old_named if (".lora_" in name) == (group_name == "lora")]
        if len(names) != len(group["params"]):
            raise ValueError("Old optimizer group size differs from verified names")
        all_ids.extend(group["params"])
        for name, index in zip(names, group["params"]):
            entry = saved["state"].get(index, {})
            for key, tensor in entry.items():
                if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                    raise ValueError("Invalid Adam state: " + name + "." + key)
                if key != "step" and tensor.shape != old_named[name].shape:
                    raise ValueError("Adam shape mismatch: " + name)
            previous[name] = entry
    if len(set(all_ids)) != len(all_ids) or set(saved["state"]) - set(all_ids):
        raise ValueError("Ambiguous optimizer parameter IDs")
    result = optimizer.state_dict()
    if len(result["param_groups"]) != 2 or {g["name"] for g in result["param_groups"]} != {"backbone", "heads"}:
        raise ValueError("Expected full-mode backbone/heads optimizer groups")
    names_by_id = {id(parameter): name for name, parameter in new_named.items()}
    for live, group in zip(optimizer.param_groups, result["param_groups"]):
        group_name = group["name"]
        origin = old_groups["lora" if group_name == "backbone" else "heads"]
        for key, value in origin.items():
            if key not in {"params", "name"}:
                group[key] = copy.deepcopy(value)
        for parameter, index in zip(live["params"], group["params"]):
            name = names_by_id[id(parameter)]
            if name.startswith("plm.") != (group_name == "backbone"):
                raise ValueError("Parameter assigned to the wrong optimizer group")
            if group_name == "heads":
                if parameter.shape != old_named[name].shape or parameter.dtype != old_named[name].dtype:
                    raise ValueError("Inherited parameter shape/dtype changed: " + name)
                if previous[name]:
                    result["state"][index] = previous[name]
    optimizer.load_state_dict(result)
    return dict(inherited_parameters=len(new_heads), backbone_state="fresh",
                inherited_state_entries=sum(bool(previous[name]) for name in new_heads))


def check_full_block_gradients(model, count):
    result = {}
    for index, layer in selected_blocks(model, count):
        parameters = list(layer.parameters())
        if any(not p.requires_grad or p.grad is None for p in parameters):
            raise AssertionError("Missing full-block gradient in layer " + str(index))
        norm = float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in parameters])))
        if not math.isfinite(norm) or norm <= 0:
            raise AssertionError("Invalid full-block gradient in layer " + str(index))
        result[str(index)] = norm
    return result
