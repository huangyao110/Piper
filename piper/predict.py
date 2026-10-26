"""Score sequence pairs with a Piper checkpoint, preserving residue masks and orientation."""
import argparse
import csv
import math
from pathlib import Path
from contextlib import nullcontext


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='CSV: seq1,seq2; optional pair_id')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    import torch
    from flashppi.research_model import load_checkpoint
    from flashppi.lora import merged_for_inference
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.input.open(newline="") as source:
        if not {"seq1", "seq2"}.issubset(csv.DictReader(source).fieldnames or []):
            raise ValueError("Input CSV requires seq1 and seq2 columns")
    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    device = next(model.parameters()).device
    context = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.incomplete')
    with args.input.open(newline='') as source, temporary.open('x', newline='') as target:
        reader = csv.DictReader(source)
        if not {'seq1', 'seq2'}.issubset(reader.fieldnames or []):
            raise ValueError('Input CSV requires seq1 and seq2 columns')
        writer = csv.DictWriter(target, fieldnames=['pair_id', 'contact_score', 'clip_score'])
        writer.writeheader()
        with torch.inference_mode(), merged_for_inference(model), context:
            for index, row in enumerate(reader):
                seqs = [(row[k] or "").strip().upper() for k in ('seq1', 'seq2')]
                if any(not s or any(c not in 'ACDEFGHIKLMNPQRSTVWYBXZUO' for c in s) for s in seqs):
                    raise ValueError(f'Invalid or empty protein sequence at CSV row {index + 2}')
                tokens = tokenizer(seqs, padding=True, truncation=False, return_tensors='pt').to(device)
                if tokens.attention_mask.sum(1).tolist() != list(map(len, seqs)):
                    raise ValueError('Residue/token alignment mismatch')
                emb = model.encode_protein(tokens.input_ids, tokens.attention_mask)
                ma, mb = tokens.attention_mask[:1].bool(), tokens.attention_mask[1:].bool()
                logits, valid = model.predict_contacts(emb[:1], emb[1:], ma, mb)
                if not valid.any():
                    raise ValueError('Pair has no valid residue contacts')
                contact = logits.float().masked_fill(~valid, -torch.inf).max().sigmoid().item()
                q = model.head_q(emb[:1], ma).float()
                k = model.head_k(emb[1:], mb).float()
                clip = (q * k).sum(-1).item()
                if not all(math.isfinite(x) for x in (contact, clip)):
                    raise ValueError(f'Nonfinite scores at CSV row {index + 2}')
                writer.writerow(dict(pair_id=row.get('pair_id') or index, contact_score=contact, clip_score=clip))
    temporary.replace(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
