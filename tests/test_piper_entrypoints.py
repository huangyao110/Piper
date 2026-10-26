"""Public entry points preserve established arguments, masks and output semantics."""
import csv
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from piper.cli import main as cli
from piper.predict import main as predict


def test_training_arguments_and_help_are_forwarded_without_reinterpretation(monkeypatch):
    seen = []
    original = sys.argv[:]
    monkeypatch.setattr('piper.cli.runpy.run_module',
                        lambda target, **kw: seen.append((target, sys.argv[:], kw)))
    cli(['train', '--config', 'path with spaces/config.json', '--resume', 'saved', '--help'])
    assert seen == [('flashppi.research_train',
                     ['flashppi.research_train', '--config', 'path with spaces/config.json',
                      '--resume', 'saved', '--help'], {'run_name': '__main__'})]
    assert sys.argv == original


def test_split_uses_existing_builder_and_restores_import_path_on_failure(monkeypatch):
    before = sys.path[:]
    def run(path, **kw):
        assert Path(path).name == 'train_test.py'
        assert sys.argv[1:] == ['--mode', 'check', '--config', 'locked.json']
        raise RuntimeError('audit failed')
    monkeypatch.setattr('piper.cli.runpy.run_path', run)
    with pytest.raises(RuntimeError, match='audit failed'):
        cli(['split', '--mode', 'check', '--config', 'locked.json'])
    assert sys.path == before


class Tokens:
    def __init__(self, seqs):
        self.attention_mask = torch.tensor([[i < len(s) for i in range(max(map(len, seqs)))] for s in seqs])
        self.input_ids = self.attention_mask.long()

    def to(self, device):
        self.attention_mask = self.attention_mask.to(device)
        self.input_ids = self.input_ids.to(device)
        return self


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def encode_protein(self, ids, mask):
        return ids.float()[..., None]

    def predict_contacts(self, a, b, ma, mb):
        valid = ma[:, :, None] & mb[:, None, :]
        # Invalid padding deliberately has a much higher logit.
        return torch.where(valid, torch.tensor(0.), torch.tensor(100.)), valid

    def head_q(self, emb, mask):
        return torch.ones((len(emb), 1))

    head_k = head_q


def test_pair_prediction_masks_padding_and_refuses_overwrite(tmp_path, monkeypatch):
    import flashppi.research_model
    tokenizer = lambda seqs, **kw: Tokens(seqs)
    monkeypatch.setattr(flashppi.research_model, 'load_checkpoint',
                        lambda *a: (TinyModel(), tokenizer, {}))
    source, output = tmp_path/'input.csv', tmp_path/'output.csv'
    source.write_text('pair_id,seq1,seq2\npair0,ac,ACDE\n')
    args = ['--checkpoint', str(tmp_path), '--input', str(source), '--output', str(output), '--device', 'cpu']
    predict(args)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [dict(pair_id='pair0', contact_score='0.5', clip_score='1.0')]
    assert not output.with_suffix('.csv.incomplete').exists()
    with pytest.raises(FileExistsError):
        predict(args)


def test_invalid_input_header_fails_before_loading_checkpoint(tmp_path, monkeypatch):
    import flashppi.research_model
    def forbidden(*a):
        pytest.fail('Checkpoint loading must not occur for invalid CSV headers')
    monkeypatch.setattr(flashppi.research_model, 'load_checkpoint', forbidden)
    source = tmp_path/'invalid.csv'
    source.write_text('sequence\nACD\n')
    with pytest.raises(ValueError, match='seq1 and seq2'):
        predict(['--checkpoint', str(tmp_path), '--input', str(source),
                 '--output', str(tmp_path/'out.csv'), '--device', 'cpu'])
