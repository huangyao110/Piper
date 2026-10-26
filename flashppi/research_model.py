"""Independent research models: released FlashPPI plus optional residue fusion.

The original training entrypoint and downloaded model sources are not modified.
ESM C is frozen in this first experiment; both branches encode the SAME cropped
sequence. BOS/EOS are removed before alignment with the original contact labels.
"""
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch import nn

from .finetune import ReleasedTrainingModel, load_released, sha256
from .lora import install_lora, load_trainable
from .patch_contact import PatchContactHead, PatchSpec
from .gated_modules import EnsembleResidualFusion, GatedContrastiveHead, ResidueGatedContactHead
from .frozen_prott5 import FrozenProtT5


class ResidualFusion(nn.Module):
    def __init__(self, secondary_dim=960, output_dim=1280):
        super().__init__()
        self.norm = nn.LayerNorm(secondary_dim)
        self.projection = nn.Linear(secondary_dim, output_dim, bias=False)
        # Only the gate is zero-initialized: the projection must have a nonzero
        # output so the gate can learn on the first optimizer step.
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, primary, secondary, mask):
        residual = self.alpha * self.projection(self.norm(secondary))
        return primary + residual.to(primary.dtype).masked_fill(~mask.bool()[..., None], 0)


def align_secondary_tokens(input_ids, attention_mask, mapping, bos, eos, pad):
    """[B,L] residue tokens -> BOS + residues + EOS + right padding."""
    mask = attention_mask.bool()
    lengths = mask.sum(1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected) or torch.any(lengths == 0):
        raise ValueError("Expected nonempty, right-padded residue sequences")
    tokens = torch.full((len(input_ids), input_ids.shape[1] + 2), pad,
                        dtype=torch.long, device=input_ids.device)
    tokens[:, 0] = bos
    tokens[:, 1:-1] = mapping[input_ids].masked_fill(~mask, pad)
    tokens[torch.arange(len(tokens), device=tokens.device), lengths + 1] = eos
    return tokens


def secondary_token_mapping(primary_tokenizer, secondary_tokenizer):
    """Map residues and the native mask token between PLM vocabularies."""
    vocab = primary_tokenizer.get_vocab()
    target = secondary_tokenizer.get_vocab()
    mapping = torch.full((max(vocab.values()) + 1,), target["X"], dtype=torch.long)
    for symbol, idx in vocab.items():
        if len(symbol) == 1 and symbol.isupper() and symbol in target:
            mapping[idx] = target[symbol]
    source_mask = primary_tokenizer.mask_token_id
    if source_mask is not None:
        if secondary_tokenizer.mask_token_id is None:
            raise ValueError("Secondary PLM must support the primary mask token")
        mapping[source_mask] = secondary_tokenizer.mask_token_id
    return mapping


class FrozenESMC(nn.Module):
    def __init__(self, directory, primary_tokenizer):
        super().__init__()
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers

        directory = Path(directory)
        self.metadata = json.loads((directory / "manifest.json").read_text())
        tokenizer = get_esmc_model_tokenizers()
        self.model = ESMC(d_model=self.metadata["d_model"],
                          n_heads=self.metadata["n_heads"],
                          n_layers=self.metadata["n_layers"],
                          tokenizer=tokenizer, use_flash_attn=False)
        weights = torch.load(directory / self.metadata["filename"],
                             map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.requires_grad_(False).eval()
        self.dim = self.metadata["d_model"]
        self.bos, self.eos, self.pad = (tokenizer.cls_token_id,
                                      tokenizer.eos_token_id, tokenizer.pad_token_id)
        self.register_buffer("token_mapping", secondary_token_mapping(primary_tokenizer, tokenizer))
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def last_hidden(self, tokens):
        # The official full forward retains all 30 hidden states and computes
        # MLM logits. Use its exact blocks/norm to obtain only final embeddings.
        x = self.model.embed(tokens)
        mask = tokens != self.pad
        chain_id = torch.ones_like(tokens)
        for block in self.model.transformer.blocks:
            result = block(x, mask, None, None, chain_id)
            x = result[0] if isinstance(result, tuple) else result
        return self.model.transformer.norm(x)

    def forward(self, input_ids, attention_mask):
        tokens = align_secondary_tokens(input_ids, attention_mask, self.token_mapping,
                                        self.bos, self.eos, self.pad)
        with torch.no_grad():
            x = self.last_hidden(tokens)
        return x[:, 1:input_ids.shape[1] + 1].masked_fill(~attention_mask.bool()[..., None], 0)


class ResearchModel(ReleasedTrainingModel):
    def __init__(self, released, tokenizer, keys, config):
        super().__init__(released, keys, config)
        # Legacy flag controls autograd/train mode for either trainable backbone.
        self.backbone_lora = config['mode'] in ('lora', 'full_last_layers')
        from .negative_sampling import validate_policy
        self.contact_negative_sampling = validate_policy(config)
        self.clip_dim = released.config.clip_embed_dim
        self.coverage_weight = float(config.get("contact_coverage", {}).get("loss_weight", 0.0))
        self.fusion_source = config.get("fusion", {}).get("source", "none")
        self.fusion = None
        self.secondary = None
        self.prott5 = None
        if self.fusion_source == "esmc":
            self.secondary = FrozenESMC(config["fusion"]["model"], tokenizer)
            self.fusion = ResidualFusion(self.secondary.dim, released.config.plm_dim)
        elif self.fusion_source == "esmc_prott5":
            if released.config.plm_dim != 1280:
                raise ValueError('Three-encoder fusion requires the 1280-dimensional gLM2 backbone')
            self.secondary = FrozenESMC(config['fusion']['esmc_model'], tokenizer)
            if self.secondary.dim != 960:
                raise ValueError('Three-encoder fusion requires ESMC-300M (960 channels)')
            self.prott5 = FrozenProtT5(config['fusion']['prott5_model'], tokenizer)
            self.fusion = EnsembleResidualFusion(released.config.plm_dim)
        elif self.fusion_source == "glm_control":
            # Same trainable fusion capacity, but no second pretrained model.
            # A fixed random projection produces the same 960 input channels.
            self.fusion = ResidualFusion(960, released.config.plm_dim)
            generator = torch.Generator().manual_seed(config["seed"] + 71)
            projection = torch.randn(released.config.plm_dim, 960, generator=generator)
            self.register_buffer("control_projection", projection / released.config.plm_dim ** 0.5)
        elif self.fusion_source != "none":
            raise ValueError("Unknown fusion source")
        pooling = config.get('clip_pooling')
        if pooling is not None:
            if pooling.get('type') != 'residue_gated':
                raise ValueError('Unknown CLIP pooling type')
            for name in ('head_q', 'head_k'):
                setattr(self, name, GatedContrastiveHead(getattr(self, name), released.config.plm_dim,
                                                       pooling.get('hidden_dim', 64)))

    def encode_protein(self, input_ids, attention_mask=None):
        with nullcontext() if self.backbone_lora else torch.no_grad():
            prefix = getattr(self, '_frozen_prefix_cache', None)
            primary = (self.plm(input_ids, attention_mask) if prefix is None
                       else prefix.encode(input_ids, attention_mask))
        if self.fusion is None:
            return primary
        if self.secondary is not None:
            secondary = self.encode_frozen("esmc", self.secondary, input_ids, attention_mask)
        else:
            secondary = primary @ self.control_projection.to(primary.dtype)
        if self.prott5 is not None:
            prott5 = self.encode_frozen("prott5", self.prott5, input_ids, attention_mask)
            return self.fusion(primary, secondary, prott5, attention_mask)
        return self.fusion(primary, secondary, attention_mask)

    def frozen_encoder_modules(self):
        return {name: module for name, module in (("esmc", self.secondary), ("prott5", self.prott5))
                if module is not None}

    def encode_frozen(self, name, module, input_ids, attention_mask):
        cache = getattr(self, "_frozen_embedding_cache", None)
        if cache is None:
            return module(input_ids, attention_mask)
        return cache.encode(name, module, input_ids, attention_mask)


def build_model(config, keys, device):
    residue_gating = config.get('residue_contact_gating')
    if residue_gating is not None:
        if any(config.get(k) is not None for k in ('contact_patch', 'patch_gating', 'contact_coverage')):
            raise ValueError('Residue contact gating excludes patch pooling, patch gating and coverage')
        if residue_gating.get('type') != 'residual_channel':
            raise ValueError('Unknown residue contact gating type')
    if config.get("initialization", {}).get("source") == "original_glm2":
        from .glm_initialization import load_original_glm
        released, tokenizer, loading = load_original_glm(config)
    elif config.get("initialization", {}).get("source", "released_flashppi") == "released_flashppi":
        released, tokenizer, loading = load_released(config["model"])
    else:
        raise ValueError("Unknown initialization source")
    released.plm.requires_grad_(False)
    if config["mode"] == "lora":
        install_lora(released, config)
        released.plm.encoder.gradient_checkpointing = True
    elif config["mode"] == "full_last_layers":
        from .full_finetune import configure_full_finetune
        if config.get('lora') is not None or config.get('lora_learning_rate') is not None:
            raise ValueError('Full-block mode excludes LoRA configuration')
        configure_full_finetune(released, config['full_finetune']['last_layers'])
    elif config["mode"] != "head_only":
        raise ValueError("Research mode must be head_only, lora or full_last_layers")
    released.contact_head.transformer.gradient_checkpointing = True
    if config.get("contact_patch") is not None:
        gating = config.get('patch_gating')
        if gating is not None and gating.get('type') != 'residual_channel':
            raise ValueError('Unknown patch gating type')
        gating = {'hidden_dim': gating.get('hidden_dim', 128)} if gating is not None else None
        released.contact_head = PatchContactHead(released.contact_head, PatchSpec(**config["contact_patch"]),
                                                coverage=bool(config.get("contact_coverage")), seed=config["seed"],
                                                gating=gating, dim=released.config.plm_dim)
    elif config.get("contact_coverage"):
        raise ValueError("Coverage supervision requires patch contacts")
    elif config.get('patch_gating') is not None:
        raise ValueError('Patch gating requires patch contacts')
    if residue_gating is not None:
        released.contact_head = ResidueGatedContactHead(
            released.contact_head, released.config.plm_dim, residue_gating.get('hidden_dim', 128))
    model = ResearchModel(released, tokenizer, keys, config).to(device)
    if config.get('warm_start_checkpoint'):
        if config.get('warm_start_transition') == 'patch_to_residue_gated_v1':
            from .residue_warm_start import warm_start
        elif config.get('warm_start_transition') is None:
            from .gated_warm_start import warm_start
        else:
            raise ValueError('Unknown warm-start transition')
        loading['warm_start'] = warm_start(model, config['warm_start_checkpoint'], config)
    return model, tokenizer, loading


def load_checkpoint(directory, device="cuda"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "trainable.safetensors") != manifest["trainable_sha256"]:
        raise ValueError("Checkpoint trainable parameter checksum mismatch")
    config = json.loads((directory / "config.json").read_text())
    # A saved new-phase checkpoint already contains every trainable tensor.
    # Rebuilding it must not reload or require its predecessor checkpoint.
    architecture_config = dict(config)
    architecture_config.pop('warm_start_checkpoint', None)
    model, tokenizer, _ = build_model(architecture_config, torch.empty(0, dtype=torch.long), device)
    load_trainable(model, directory / "trainable.safetensors")
    return model.eval(), tokenizer, config
