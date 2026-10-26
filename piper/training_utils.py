"""Shared model loading and training helpers; training lives in piper.train."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from .model import FlashPPIModel

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'artifacts/hf_cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def load_released(path):
    model, info = AutoModel.from_pretrained(path, trust_remote_code=True,
                local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError(f'Checkpoint is not an exact load: {info}')
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True,local_files_only=True)
    return model, tokenizer, info

class ReleasedTrainingModel(FlashPPIModel):
    """Reuse the repository training forward with the exact released modules."""
    def __init__(self, released, keys, cfg):
        torch.nn.Module.__init__(self)
        for name in ('plm','head_q','head_k','contact_head','logit_scale'):
            setattr(self,name,getattr(released,name))
        self.register_buffer('valid_keys',keys,persistent=False)
        self.multiplier = cfg.get('cluster_multiplier', 1_000_000)
        self.backbone_lora = cfg['mode'] == 'lora'
        self._adds_special_tokens = False
        self.swap_self_negative = cfg['swap_self_negative']
        self.contact_random_neg_ratio = cfg['contact_random_neg_ratio']
        self.contact_self_neg_ratio = cfg['contact_self_neg_ratio']
        self.contact_hard_neg_ratio = cfg['contact_hard_neg_ratio']

    def encode_protein(self, input_ids, attention_mask=None):
        with nullcontext() if self.backbone_lora else torch.no_grad():
            return self.plm(input_ids, attention_mask)

    def train(self, mode=True):
        super().train(mode)
        self.plm.train(mode if self.backbone_lora else False)
        return self

def clear_rotary_cache(model):
    # Released RoPE caches do not track dtype. Rebuild after BF16/FP16 changes;
    # these are nonpersistent trigonometric caches, not learned model weights.
    for module in model.modules():
        if hasattr(module, '_cos_cached'):
            module._cos_cached = None
            module._sin_cached = None
            module._seq_len_cached = 0

def sample_window(step, batch_size, rows_per_epoch=None):
    """Keep each epoch's final partial batch; never omit or repeat draws."""
    if rows_per_epoch is None:
        return step*batch_size, (step+1)*batch_size
    epoch, within = divmod(step, math.ceil(rows_per_epoch/batch_size))
    left = epoch*rows_per_epoch+within*batch_size
    return left, min(left+batch_size, (epoch+1)*rows_per_epoch)

def validate_resume_config(state, cfg, smoke, extend=False):
    saved_config = dict(state['config'])
    if extend:
        if cfg['steps'] <= max(state['step'], saved_config['steps']):
            raise ValueError('Extended budget must exceed saved step and original budget')
        saved_config['steps'] = cfg['steps']
    if saved_config != cfg or state['smoke'] != smoke:
        raise ValueError('Resume configuration/mode differs')
