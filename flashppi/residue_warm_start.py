"""Strict transfer from a gated patch checkpoint to residue contact prediction."""
import torch
from safetensors.torch import load_file

from .gated_warm_start import validate_warm_start_checkpoint


DROPPED = {'contact_head.coverage_mix.weight', 'contact_head.coverage_mix.bias'}


def restore_residue_parameters(model, saved):
    """Copy all learned tensors exactly, except the removed coverage readout.

    The channel gate is pointwise and shape-compatible at residue resolution.
    No other parameter may be missing, newly initialized, or silently dropped.
    """
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    mapping = {name: name.replace('contact_head.residue_gate.', 'contact_head.patch_gate.', 1)
               for name in parameters}
    expected = set(mapping.values()) | DROPPED
    if set(saved) != expected:
        raise ValueError('Residue transition parameter names differ: missing=' +
                         str(sorted(expected - set(saved))) + '; unexpected=' +
                         str(sorted(set(saved) - expected)))
    for name, source in mapping.items():
        tensor, parameter = saved[source], parameters[name]
        if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise ValueError('Residue transition shape/dtype mismatch: ' + source)
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite source tensor: ' + source)
    with torch.no_grad():
        for name, source in mapping.items():
            parameters[name].copy_(saved[source].to(parameters[name].device))
            if not torch.equal(parameters[name].detach().cpu(), saved[source]):
                raise AssertionError('Residue transition changed parameter: ' + name)
    return dict(copied_parameters=sorted(parameters),
                renamed_parameters={s: n for n, s in mapping.items() if s != n},
                dropped_parameters=sorted(DROPPED), new_parameters=[], all_copied_exact=True)


def validate_transition_config(source, config):
    for key in ('fusion', 'clip_pooling'):
        if source.get(key) != config.get(key):
            raise ValueError('Residue transition must preserve ' + key)
    if source.get('fusion', {}).get('source') != 'esmc_prott5':
        raise ValueError('Residue transition requires the trained three-encoder fusion')
    if not source.get('contact_patch') or not source.get('contact_coverage'):
        raise ValueError('Residue transition requires a patch checkpoint with coverage')
    gate = config.get('residue_contact_gating')
    if not gate or gate != source.get('patch_gating') or gate.get('type') != 'residual_channel':
        raise ValueError('Residue transition must preserve the channel gate architecture')
    if any(config.get(k) is not None for k in ('contact_patch', 'patch_gating', 'contact_coverage')):
        raise ValueError('Residue transition must remove patch pooling and coverage supervision')


def warm_start(model, directory, config):
    directory, manifest, source, state = validate_warm_start_checkpoint(directory, config)
    validate_transition_config(source, config)
    report = restore_residue_parameters(model, load_file(str(directory / 'trainable.safetensors')))
    report.update(checkpoint=str(directory.resolve()), step=state['step'],
                  transition='patch_to_residue_gated_v1',
                  source_patch=source['contact_patch'], target_resolution='residue',
                  trainable_sha256=manifest['trainable_sha256'],
                  training_state_sha256=manifest['training_state_sha256'],
                  source_hashes=state['source_hashes'],
                  optimizer='fresh; predecessor state not loaded')
    return report
