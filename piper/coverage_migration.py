"""Explicit one-way migration: add directional coverage, retain all learned state."""
import copy
import torch

NEW_PARAMETERS = {"contact_head.coverage_mix.weight", "contact_head.coverage_mix.bias"}


def restore_existing_parameters(model, saved):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) - set(saved) != NEW_PARAMETERS or set(saved) - set(parameters):
        raise ValueError("Migration may only add the two coverage output parameters")
    with torch.no_grad():
        for name, tensor in saved.items():
            if tensor.shape != parameters[name].shape or tensor.dtype != parameters[name].dtype:
                raise ValueError("Changed existing parameter shape/dtype: " + name)
            parameters[name].copy_(tensor.to(parameters[name].device))
            if not torch.equal(parameters[name].detach().cpu(), tensor):
                raise AssertionError("Existing parameter not preserved: " + name)


def migrate_adam_state(model, optimizer, saved, old_names):
    """Map by verified names, never by shifted positional parameter IDs."""
    named = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    current_names = set(named.values())
    if current_names - set(old_names) != NEW_PARAMETERS or set(old_names) - current_names:
        raise ValueError("Unexpected optimizer parameter migration")
    result = optimizer.state_dict()
    previous_groups = {group["name"]: group for group in saved["param_groups"]}
    state_by_name = {}
    for name, group in previous_groups.items():
        if name not in {"lora", "heads"}:
            raise ValueError("Expected named adapter/head optimizer groups")
        names = [n for n in old_names if (".lora_" in n) == (name == "lora")]
        if len(names) != len(group["params"]):
            raise ValueError("Previous optimizer group size differs from parameter names")
        for n, identifier in zip(names, group["params"]):
            if identifier in saved["state"]:
                state_by_name[n] = saved["state"][identifier]
    parameters = dict(model.named_parameters())
    for live, serialized in zip(optimizer.param_groups, result["param_groups"]):
        previous = previous_groups[serialized["name"]]
        ids = serialized["params"]
        serialized.update({k: copy.deepcopy(v) for k, v in previous.items() if k != "params"})
        for parameter, identifier in zip(live["params"], ids):
            name = named[id(parameter)]
            if name in state_by_name:
                state = state_by_name[name]
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].shape != parameters[name].shape:
                        raise ValueError("Optimizer moment shape mismatch: " + name)
                result["state"][identifier] = copy.deepcopy(state)
    optimizer.load_state_dict(result)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = named[id(parameter)]
            actual = optimizer.state.get(parameter, {})
            expected = state_by_name.get(name, {})
            if actual.keys() != expected.keys():
                raise AssertionError("Optimizer state keys changed: " + name)
            for key in expected:
                if torch.is_tensor(expected[key]):
                    if not torch.equal(actual[key].cpu(), expected[key].cpu()):
                        raise AssertionError("Optimizer state not preserved: " + name)
                elif actual[key] != expected[key]:
                    raise AssertionError("Optimizer scalar not preserved: " + name)
    return dict(preserved_parameter_states=len(state_by_name), new_parameter_states=0)
