"""Q/V-only LoRA for the released gLM2 fused [Q, K, V] projection.

The frozen weight retains its original state-dict name. Adapter training state
is saved separately; inference exports merge into the original architecture.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


class QVLoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid LoRA rank, alpha, or dropout')
        if not isinstance(base, nn.Linear) or base.out_features != 3 * base.in_features:
            raise ValueError('Expected released square fused [Q,K,V] projection')
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        for part in ('q', 'v'):
            a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            b = nn.Parameter(torch.zeros(base.in_features, rank, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.register_parameter('lora_a_' + part, a)
            self.register_parameter('lora_b_' + part, b)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if not self.enabled:
            return result
        dropped = self.dropout(x)
        dq = F.linear(F.linear(dropped, self.lora_a_q), self.lora_b_q) * self.scale
        dv = F.linear(F.linear(dropped, self.lora_a_v), self.lora_b_v) * self.scale
        return result + torch.cat([dq, torch.zeros_like(dq), dv], dim=-1)

    def merged_weight_cpu(self):
        # CPU FP32 computation is shared by evaluation and export, so neither
        # low-precision merge drift nor subtract-to-unmerge roundoff accumulates.
        result = self.weight.detach().float().cpu().clone()
        n = self.in_features
        result[:n] += (self.lora_b_q.detach().float().cpu() @ self.lora_a_q.detach().float().cpu()) * self.scale
        result[2*n:] += (self.lora_b_v.detach().float().cpu() @ self.lora_a_v.detach().float().cpu()) * self.scale
        return result.to(self.weight.dtype)


def install_lora(model, config):
    settings = config['lora']
    layers = model.plm.encoder.layers
    count = settings['last_layers']
    if not 1 <= count <= len(layers):
        raise ValueError('Invalid number of LoRA layers')
    model.plm.requires_grad_(False)
    names = []
    # Adapter initialization must not change the CPU augmentation RNG stream.
    device = next(model.plm.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config['seed'])
        for i in range(len(layers) - count, len(layers)):
            layers[i].attention.wqkv = QVLoRALinear(
                layers[i].attention.wqkv, settings['rank'], settings['alpha'], settings['dropout'])
            names.append(f'plm.encoder.layers.{i}.attention.wqkv')
    return names


@contextmanager
def merged_for_inference(model):
    saved = []
    try:
        for module in model.modules():
            if isinstance(module, QVLoRALinear):
                saved.append((module, module.weight, module.enabled))
                module.weight = nn.Parameter(module.merged_weight_cpu().to(module.weight.device), requires_grad=False)
                module.enabled = False
        yield
    finally:
        for module, weight, enabled in saved:
            module.weight = weight
            module.enabled = enabled


def merged_state_dict(model):
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if '.lora_' not in k}
    for name, module in model.named_modules():
        if isinstance(module, QVLoRALinear):
            state[name + '.weight'] = module.merged_weight_cpu().contiguous()
    return state


def save_trainable(model, path):
    save_file({n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}, str(path))


def load_trainable(model, path):
    state = load_file(str(path))
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(parameters):
        raise ValueError('Adapter/head checkpoint parameter names do not match')
    with torch.no_grad():
        for name, parameter in parameters.items():
            if state[name].shape != parameter.shape:
                raise ValueError('Adapter/head shape mismatch: ' + name)
            parameter.copy_(state[name].to(parameter.device))
