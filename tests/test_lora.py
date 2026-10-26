"""Adapter correctness and GradCache dropout replay; CPU, no downloads."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piper.lora import QVLoRALinear, merged_for_inference, merged_state_dict, save_trainable, load_trainable
from test_finetune_training import Accelerator, Toy
from piper.loss import PPIGradientCache, GradCacheClipLoss, ContactLoss


class DropoutToy(Toy):
    def __init__(self, swap):
        super().__init__(swap)
        self.proj = QVLoRALinear(nn.Linear(3, 9), rank=2, alpha=4, dropout=.2)
        # Nonzero B exercises gradients through both matrices and dropout.
        nn.init.normal_(self.proj.lora_b_q, std=.1)
        nn.init.normal_(self.proj.lora_b_v, std=.1)

    def embed_pair(self, a, b, ma, mb, c1=None, c2=None):
        x = checkpoint(self.proj, a.float(), use_reentrant=False)
        y = checkpoint(self.proj, b.float(), use_reentrant=False)
        if self.swap_self_negative:
            x, y = torch.cat([x, y]), torch.cat([y, x])
        return x, y, None, None


class Tests(unittest.TestCase):
    def test_zero_init_k_frozen_merge_and_restore(self):
        torch.manual_seed(2)
        base = nn.Linear(4, 12)
        layer = QVLoRALinear(base, rank=2, alpha=4, dropout=.1)
        x = torch.randn(3, 4)
        self.assertTrue(torch.equal(base(x), layer(x)))
        layer(x).square().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.lora_b_q.grad.norm().item(), 0)
        with torch.no_grad():
            layer.lora_b_q.normal_(); layer.lora_b_v.normal_()
        layer.eval()
        expected = layer(x)
        self.assertTrue(torch.equal(expected[:, 4:8], base(x)[:, 4:8]))
        model = nn.Sequential(layer)
        original = layer.weight
        value = original.detach().clone()
        with merged_for_inference(model):
            torch.testing.assert_close(model(x), expected, atol=2e-6, rtol=2e-6)
            self.assertFalse(layer.enabled)
        self.assertIs(layer.weight, original)
        self.assertTrue(torch.equal(original, value))
        self.assertTrue(layer.enabled)
        state = merged_state_dict(model)
        self.assertFalse(any('lora_' in k for k in state))
        restored = nn.Sequential(nn.Linear(4, 12))
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored(x), expected, atol=2e-6, rtol=2e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trainable.safetensors'
            save_trainable(model, path)
            other = nn.Sequential(QVLoRALinear(copy.deepcopy(base), 2, 4, .1))
            load_trainable(other, path)
            for (n,p), (m,q) in zip(model.named_parameters(), other.named_parameters()):
                if p.requires_grad:
                    self.assertEqual(n,m); self.assertTrue(torch.equal(p,q))

    def test_gradcache_replays_dropout_with_checkpointing(self):
        for swap in (False, True):
            torch.manual_seed(4)
            model = DropoutToy(swap); direct = copy.deepcopy(model)
            a = torch.randn(5,3); b = torch.randn(5,3); mask = torch.ones(5,3)
            batch = {'seq1_tokens':{'input_ids':a,'attention_mask':mask},
                     'seq2_tokens':{'input_ids':b,'attention_mask':mask},
                     'contact_target':torch.zeros(5,3,3)}
            rng = torch.get_rng_state()
            PPIGradientCache(model,2,GradCacheClipLoss(),ContactLoss(),0,Accelerator(),.1).step(batch)
            torch.set_rng_state(rng)
            parts = [direct.embed_pair(a[i:i+2],b[i:i+2],mask[i:i+2],mask[i:i+2]) for i in range(0,5,2)]
            x = torch.cat([p[0] for p in parts]); y = torch.cat([p[1] for p in parts])
            (.1*GradCacheClipLoss()(x,y,direct.logit_scale.exp())).backward()
            for p,q in zip(model.parameters(),direct.parameters()):
                if p.requires_grad:
                    torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
                else:
                    self.assertIsNone(p.grad)


if __name__ == '__main__':
    unittest.main()
