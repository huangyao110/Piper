"""Fresh task heads with the original, checksum-pinned gLM2 UR50 backbone.

The released FlashPPI architecture code supplies its optimized GLM2 backbone;
no tensors from a FlashPPI checkpoint are loaded. Task heads use the repository
training constructors (including contact-logit bias -3), not HF post_init.
"""
import json
from pathlib import Path
import types

from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .model import ContactHead, ContrastiveHead
from .training_utils import sha256


def load_original_glm(config):
    settings = config['initialization']
    directory = Path(settings['backbone'])
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['repository'] != 'tattabio/gLM2_650M_UR50':
        raise ValueError('Expected original gLM2 UR50, not a FlashPPI checkpoint')
    for filename, expected in manifest['files'].items():
        if sha256(directory/filename) != expected:
            raise ValueError('Original backbone file changed: '+filename)
    architecture = config['model']
    cfg = AutoConfig.from_pretrained(architecture, trust_remote_code=True, local_files_only=True)
    native = json.loads((directory/'config.json').read_text())
    for native_name, released_name in [('dim','plm_dim'),('depth','plm_depth'),('heads','plm_heads'),
                                     ('vocab_size','plm_vocab_size'),('norm_eps','plm_norm_eps')]:
        if native[native_name] != getattr(cfg, released_name):
            raise ValueError('Backbone architecture mismatch: '+native_name)
    cls = get_class_from_dynamic_module(cfg.auto_map['AutoModel'], architecture, local_files_only=True)
    module = __import__(cls.__module__, fromlist=['GLM2Backbone'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        backbone = module.GLM2Backbone(cfg)
        with safe_open(str(directory/'model.safetensors'), framework='pt', device='cpu') as weights:
            state = {key.removeprefix('glm2.'): weights.get_tensor(key)
                     for key in weights.keys() if key.startswith('glm2.')}
        backbone.load_state_dict(state, strict=True)
        del state
        model = types.SimpleNamespace(config=cfg, plm=backbone,
            head_q=ContrastiveHead(cfg.plm_dim,cfg.clip_embed_dim),
            head_k=ContrastiveHead(cfg.plm_dim,cfg.clip_embed_dim),
            contact_head=ContactHead(cfg.plm_dim,cfg.contact_embed_dim,cfg.contact_num_heads),
            logit_scale=torch.nn.Parameter(torch.log(torch.tensor(1/0.07))))
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    reference_tokenizer = AutoTokenizer.from_pretrained(architecture, trust_remote_code=True, local_files_only=True)
    if tokenizer.get_vocab() != reference_tokenizer.get_vocab():
        raise ValueError('Backbone and FlashPPI residue vocabularies differ')
    info = dict(initialization='original_glm2_fresh_task_heads', backbone_revision=manifest['revision'],
                backbone_sha256=manifest['weights_sha256'], flashppi_weights_loaded=False,
                task_head_initialization='repository constructors; contact output bias -3')
    return model, tokenizer, info
