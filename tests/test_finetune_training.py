"""CPU regression checks; no downloads or pretrained models required."""
import copy
from contextlib import nullcontext
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import torch
from flashppi.model import get_contact_negatives
from flashppi.dataset import DataCollatorForPPI
from flashppi.loss import ContactLoss, GradCacheClipLoss, PPIGradientCache


class Tokenizer:
    def __call__(self,seqs,**kw):
        n=kw['max_length'];mask=torch.zeros(len(seqs),n,dtype=torch.long)
        for i,s in enumerate(seqs):mask[i,:min(n,len(s))]=1
        return {'input_ids':mask.clone(),'attention_mask':mask}


class Accelerator:
    device=torch.device('cpu')
    num_processes=1
    autocast=staticmethod(nullcontext)
    no_sync=staticmethod(lambda model:nullcontext())
    backward=staticmethod(lambda loss:loss.backward())


class Toy(torch.nn.Module):
    def __init__(self,swap):
        super().__init__();self.proj=torch.nn.Linear(3,4);self.logit_scale=torch.nn.Parameter(torch.tensor(.3));self.swap_self_negative=swap
    def embed_pair(self,a,b,ma,mb,c1=None,c2=None):
        x,y=self.proj(a.float()),self.proj(b.float())
        if self.swap_self_negative:x,y=torch.cat([x,y]),torch.cat([y,x])
        return x,y,None,None
    def forward(self,*args):
        x,y,_,_=self.embed_pair(*args)
        return x,y,None,self.logit_scale.exp(),None


class Tests(unittest.TestCase):
    def test_hard_candidates_exhausted_never_select_positive(self):
        e=torch.randn(3,2,4);m=torch.ones(3,2,dtype=torch.bool);c=torch.arange(3)
        keys=(c[:,None]*10+c[None,:]).flatten()
        out=get_contact_negatives(e,e,m,m,0,0,2,c,c,keys,10,torch.randn(3,4),torch.randn(3,4))
        self.assertEqual(len(out[0]),0);self.assertEqual(out[2].dtype,torch.bool)

    def test_self_pairs_are_same_protein_and_known_homomers_excluded(self):
        a=torch.tensor([[[1.]],[[2.]]]);b=a+10;m=torch.ones(2,1,dtype=torch.bool)
        out=get_contact_negatives(a,b,m,m,0,1,0,torch.tensor([1,2]),torch.tensor([3,4]),torch.tensor([11]),10)
        torch.testing.assert_close(out[0],a[1:]);torch.testing.assert_close(out[1],a[1:])

    def test_random_ratio_cannot_wrap_to_original_pair(self):
        a=torch.arange(3.).reshape(3,1,1);m=torch.ones(3,1,dtype=torch.bool)
        out=get_contact_negatives(a,a+10,m,m,20,0,0)
        self.assertEqual(len(out[0]),6)
        self.assertTrue(torch.all(out[0]+10 != out[1]))

    def test_structure_unknown_not_negative_and_synthetic_negatives_retained(self):
        logits=torch.zeros(2,2,2,requires_grad=True)
        target=torch.tensor([[[1,0],[0,0]]],dtype=torch.bool)
        mask=torch.tensor([[[1,0],[0,0]]],dtype=torch.bool)
        loss=ContactLoss()(logits,target,torch.ones_like(logits,dtype=torch.bool),mask);loss.backward()
        self.assertEqual(float(logits.grad[0,1,1]),0)
        self.assertLess(float(logits.grad[0,0,0]),0)
        self.assertTrue(torch.all(logits.grad[1]>0))

    def test_swap_transposes_structural_mask_without_hiding_tokens(self):
        row={'seq1':'ABC','seq2':'DE','contacts':[(0,1)],'valid1':[1,0,1],'valid2':[0,1]}
        with patch('torch.rand',return_value=torch.tensor([0.])):
            batch=DataCollatorForPPI(Tokenizer(),4,True,False)([row])
        self.assertTrue(batch['contact_target'][0,1,0])
        self.assertTrue(batch['contact_target_valid_mask'][0,1,0])
        self.assertFalse(batch['contact_target_valid_mask'][0,1,1])
        self.assertEqual(int(batch['seq2_tokens']['attention_mask'].sum()),3)

    def test_gradcache_matches_direct_clip_gradient_and_scale_weight(self):
        for swap in (False,True):
            torch.manual_seed(4);model=Toy(swap);direct=copy.deepcopy(model)
            a=torch.randn(5,3);b=torch.randn(5,3);mask=torch.ones(5,3)
            batch={'seq1_tokens':{'input_ids':a,'attention_mask':mask},'seq2_tokens':{'input_ids':b,'attention_mask':mask},'contact_target':torch.zeros(5,3,3)}
            cache=PPIGradientCache(model,2,GradCacheClipLoss(),ContactLoss(),0,Accelerator(),.1)
            cache.step(batch)
            # GradCache concatenates [A_chunk,B_chunk] when swap is enabled.
            parts=[direct.embed_pair(a[i:i+2],b[i:i+2],mask[i:i+2],mask[i:i+2]) for i in range(0,5,2)]
            x=torch.cat([p[0] for p in parts]);y=torch.cat([p[1] for p in parts])
            (.1*GradCacheClipLoss()(x,y,direct.logit_scale.exp())).backward()
            for p,q in zip(model.parameters(),direct.parameters()):
                torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)

class ResumeConfigurationTests(unittest.TestCase):
    def test_two_epochs_preserve_all_draws_and_partial_batches(self):
        from flashppi.finetune import sample_window
        windows = [sample_window(step,1024,1120675) for step in range(2190)]
        self.assertEqual(windows[0][0],0)
        self.assertEqual(windows[-1][1],2241350)
        self.assertTrue(all(a[1]==b[0] for a,b in zip(windows,windows[1:])))
        self.assertEqual(windows[1094][1]-windows[1094][0],419)
        self.assertEqual(windows[2189][1]-windows[2189][0],419)
        self.assertEqual(windows[1095][0],1120675)

    def test_extension_preserves_optimizer_settings(self):
        from flashppi.finetune import validate_resume_config
        state = {'config': {'steps': 63, 'learning_rate': 1e-5}, 'step': 63, 'smoke': False}
        validate_resume_config(state, {'steps': 126, 'learning_rate': 1e-5}, False, extend=True)
        with self.assertRaises(ValueError):
            validate_resume_config(state, {'steps': 126, 'learning_rate': 2e-5}, False, extend=True)
        with self.assertRaises(ValueError):
            validate_resume_config(state, {'steps': 63, 'learning_rate': 1e-5}, False, extend=True)
        with self.assertRaises(ValueError):
            validate_resume_config(state, {'steps': 126, 'learning_rate': 1e-5}, False)


if __name__=='__main__':unittest.main()
