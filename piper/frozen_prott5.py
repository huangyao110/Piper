"""Local ProtT5 encoder with one token per primary (possibly masked) residue."""
from pathlib import Path

import torch
from torch import nn


def prott5_token_mapping(primary_tokenizer, tokenizer):
    vocab = primary_tokenizer.get_vocab()
    mapping = torch.full((max(vocab.values()) + 1,), -1, dtype=torch.long)
    symbols = {}
    for symbol, index in vocab.items():
        if len(symbol) == 1 and symbol.isascii() and symbol.isupper():
            symbols[index] = 'X' if symbol in 'UZOB' else symbol
    if primary_tokenizer.mask_token_id is None:
        raise ValueError('Primary tokenizer must define its masking token')
    symbols[primary_tokenizer.mask_token_id] = '<extra_id_0>'
    for index, symbol in symbols.items():
        tokens = tokenizer.encode(symbol, add_special_tokens=False)
        if (len(tokens) != 1 or tokens[0] in
                (tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id)):
            raise ValueError('ProtT5 must encode each residue/mask as one token: ' + symbol)
        mapping[index] = tokens[0]
    # Check spaces and repeated sentinel masks do not merge or split positions.
    ordered = list(symbols) + [primary_tokenizer.mask_token_id] * 2
    text = ' '.join(symbols[index] for index in ordered)
    if tokenizer.encode(text, add_special_tokens=False) != mapping[ordered].tolist():
        raise ValueError('ProtT5 tokenizer does not preserve one token per residue in sequences')
    return mapping


class FrozenProtT5(nn.Module):
    def __init__(self, directory, primary_tokenizer):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer

        directory = str(Path(directory))
        tokenizer = T5Tokenizer.from_pretrained(directory, local_files_only=True, do_lower_case=False,
                                                legacy=True)
        self.model = T5EncoderModel.from_pretrained(directory, local_files_only=True, torch_dtype=torch.float16)
        self.dim = self.model.config.d_model
        if self.dim != 1024:
            raise ValueError('Expected 1024-dimensional ProtT5 XL encoder')
        self.eos, self.pad = tokenizer.eos_token_id, tokenizer.pad_token_id
        if self.eos is None or self.pad is None or self.eos == self.pad:
            raise ValueError('ProtT5 requires distinct EOS and padding IDs')
        self.register_buffer('token_mapping', prott5_token_mapping(primary_tokenizer, tokenizer))
        self.model.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, input_ids, attention_mask):
        mask = attention_mask.bool()
        lengths = mask.sum(1)
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        if (input_ids.shape != mask.shape or not torch.equal(mask, expected)
                or torch.any(lengths == 0)):
            raise ValueError('Expected nonempty, right-padded residue sequences')
        # Padding IDs need not belong to the residue vocabulary. Sanitize first.
        safe_ids = input_ids.masked_fill(~mask, 0)
        if torch.any(safe_ids < 0) or torch.any(safe_ids >= len(self.token_mapping)):
            raise ValueError('Primary residue token is outside the validated vocabulary')
        mapped = self.token_mapping[safe_ids]
        if torch.any(mapped[mask] < 0):
            raise ValueError('Primary special token encountered at a residue position')
        tokens = torch.full((len(input_ids), input_ids.shape[1] + 1), self.pad,
                            dtype=torch.long, device=input_ids.device)
        tokens[:, :-1] = mapped.masked_fill(~mask, self.pad)
        tokens[torch.arange(len(tokens), device=tokens.device), lengths] = self.eos
        token_mask = torch.arange(tokens.shape[1], device=tokens.device)[None] <= lengths[:, None]
        with torch.no_grad():
            encoded = self.model(input_ids=tokens, attention_mask=token_mask).last_hidden_state
        if encoded.shape != (*tokens.shape, self.dim):
            raise ValueError('ProtT5 output shape does not match its residue token grid')
        return encoded[:, :input_ids.shape[1]].masked_fill(~mask[..., None], 0)
