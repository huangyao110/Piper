"""Strict name-preserving initialization for the new gated training phase."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .finetune import sha256


NEW_PARAMETERS = {'fusion.alpha_esmc', 'fusion.alpha_prott5', 'contact_head.patch_gate.beta'}
NEW_PARAMETERS.update('fusion.' + name + '.' + kind
                      for name in ('norm_esmc', 'norm_prott5') for kind in ('weight', 'bias'))
NEW_PARAMETERS.update('fusion.projection_' + branch + '.weight' for branch in ('esmc', 'prott5'))
NEW_PARAMETERS.update(prefix + name + '.' + kind
                      for prefix, layers in (
                          ('head_q.residue_pool.', ('norm', 'score.0', 'score.2')),
                          ('head_k.residue_pool.', ('norm', 'score.0', 'score.2')),
                          ('contact_head.patch_gate.', ('norm', 'gate.0', 'gate.2', 'projection.0', 'projection.2')))
                      for name in layers for kind in ('weight', 'bias'))


def restore_inherited_parameters(model, saved):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    inherited = set(parameters) - NEW_PARAMETERS
    if set(saved) != inherited:
        raise ValueError('Warm start inherited parameter names differ: missing=' +
                         str(sorted(inherited - set(saved))) + '; unexpected=' +
                         str(sorted(set(saved) - inherited)))
    # Validate everything before mutating any live tensor.
    for name, tensor in saved.items():
        if tensor.shape != parameters[name].shape or tensor.dtype != parameters[name].dtype:
            raise ValueError('Inherited parameter shape/dtype mismatch: ' + name)
    with torch.no_grad():
        for name, tensor in saved.items():
            parameters[name].copy_(tensor.to(parameters[name].device))
            if not torch.equal(parameters[name].detach().cpu(), tensor):
                raise AssertionError('Inherited parameter changed during copy: ' + name)
    return dict(inherited_parameters=sorted(inherited), new_parameters=sorted(set(parameters) - inherited))


def validate_warm_start_checkpoint(directory, config):
    """Verify the saved state, architecture, source snapshot and frozen inputs."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for filename, key in (('trainable.safetensors', 'trainable_sha256'),
                          ('training_state.pt', 'training_state_sha256')):
        if sha256(directory / filename) != manifest[key]:
            raise ValueError('Warm-start checkpoint checksum mismatch: ' + filename)
    source = json.loads((directory / 'config.json').read_text())
    state = torch.load(directory / 'training_state.pt', map_location='cpu', weights_only=False)
    if state['config'] != source or state['step'] != manifest['step']:
        raise ValueError('Warm-start checkpoint metadata differs from training state')
    def base_signature(settings):
        initialization = dict(settings.get('initialization', {}))
        if initialization.get('backbone') is not None:
            initialization['backbone'] = str(Path(initialization['backbone']).resolve())
        return dict(model=str(Path(settings['model']).resolve()), initialization=initialization,
                    mode=settings.get('mode'), lora=settings.get('lora'))
    previous, current = base_signature(source), base_signature(config)
    for key in previous:
        if previous[key] != current[key]:
            raise ValueError('Warm-start base/LoRA configuration mismatch: ' + key)
    if not state.get('source_hashes'):
        raise ValueError('Warm start requires pinned predecessor source hashes')
    run = directory.parent.parent
    for relative, expected in state['source_hashes'].items():
        if sha256(run / 'code' / relative) != expected:
            raise ValueError('Warm-start source checksum mismatch: ' + relative)
    # All original model/input pins must still match, including the gLM2 base.
    for filename, expected in state['input_hashes'].items():
        if sha256(filename) != expected:
            raise ValueError('Warm-start frozen input checksum mismatch: ' + filename)
    return directory, manifest, source, state


def warm_start(model, directory, config):
    directory, manifest, source, state = validate_warm_start_checkpoint(directory, config)
    if source.get('fusion', {}).get('source', 'none') != 'none':
        raise ValueError('Warm start requires predecessor without fusion parameters')
    report = restore_inherited_parameters(model, load_file(str(directory / 'trainable.safetensors')))
    report.update(checkpoint=str(directory.resolve()), step=state['step'],
                  trainable_sha256=manifest['trainable_sha256'],
                  training_state_sha256=manifest['training_state_sha256'],
                  source_hashes=state['source_hashes'], optimizer='fresh; predecessor state not loaded')
    return report
