"""Released-checkpoint head-only or Q/V LoRA fine-tuning and fixed validation evaluation.

The official backbone and heads are reused verbatim. Architecture and inference
score remain unchanged. Run via python -m flashppi.finetune --help.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'artifacts/hf_cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from sklearn.metrics import average_precision_score, roc_auc_score
from .lora import (QVLoRALinear, install_lora, merged_for_inference, merged_state_dict,
                   save_trainable, load_trainable)
from .model import FlashPPIModel
from .dataset import DataCollatorForPPI, ClusterSampler, build_interaction_keys
from .loss import GradCacheClipLoss, ContactLoss, PPIGradientCache
from .cropping import valid_contact_counts


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


@torch.inference_mode()
def evaluate(model, tokenizer, cfg, output):
    with merged_for_inference(model):
        return _evaluate(model, tokenizer, cfg, output)


def _evaluate(model, tokenizer, cfg, output):
    """Full-length val AP plus masked contact precision@K on all structures."""
    output.mkdir(parents=True,exist_ok=True)
    positives = pq.read_table(cfg['validation_structures']).to_pylist()
    binary = pq.read_table(cfg['validation_interactions']).to_pylist()
    rows = binary + positives
    sequences = sorted({r[k] for r in rows for k in ('seq1','seq2')},key=lambda s:(len(s),s))
    model.eval()
    clear_rotary_cache(model)
    cache = {}
    with torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(sequences),4):
            seqs = sequences[start:start+4]
            tokens = tokenizer(seqs,padding=True,truncation=False,return_tensors='pt').to('cuda')
            assert tokens.attention_mask.sum(1).tolist() == list(map(len,seqs))
            encoded = model.encode_protein(tokens.input_ids,tokens.attention_mask)
            for j,s in enumerate(seqs):
                cache[s] = encoded[j,:len(s)].clone()
        print('Validation encoded',len(cache),'sequences',flush=True)
        scores = np.zeros(len(rows))
        clip_scores = np.zeros(len(binary))
        contact = []
        order = sorted(range(len(rows)),key=lambda i:(len(rows[i]['seq1'])//64,len(rows[i]['seq2'])//64))
        cursor = 0
        while cursor < len(order):
            indices=[]; la=lb=0
            for idx in order[cursor:cursor+16]:
                a,b = len(rows[idx]['seq1']),len(rows[idx]['seq2'])
                na,nb=max(la,a),max(lb,b)
                if indices and (len(indices)+1)*(na+nb)**2 > 12_000_000:
                    break
                indices.append(idx); la,lb=na,nb
            left=[cache[rows[i]['seq1']] for i in indices]
            right=[cache[rows[i]['seq2']] for i in indices]
            a,b=pad_sequence(left,batch_first=True),pad_sequence(right,batch_first=True)
            ma=torch.arange(la,device='cuda')[None,:] < torch.tensor([len(x) for x in left],device='cuda')[:,None]
            mb=torch.arange(lb,device='cuda')[None,:] < torch.tensor([len(x) for x in right],device='cuda')[:,None]
            logits,valid=model.predict_contacts(a,b,ma,mb)
            logits=logits.float().masked_fill(~valid,-torch.inf)
            sc=torch.sigmoid(logits.flatten(1).max(1).values).cpu().tolist()
            q,k=model.head_q(a,ma).float(),model.head_k(b,mb).float()
            for j,i in enumerate(indices):
                r=rows[i]; scores[i]=sc[j]
                if i < len(binary):
                    clip_scores[i]=float((q[j]*k[j]).sum())
                else:
                    n,m=len(r['seq1']),len(r['seq2'])
                    structural=torch.tensor(r['valid1'],device='cuda')[:,None] & torch.tensor(r['valid2'],device='cuda')[None,:]
                    coords={(int(x),int(y)) for x,y in r['contacts'] if r['valid1'][x] and r['valid2'][y]}
                    if coords:
                        values=logits[j,:n,:m].masked_fill(~structural,-torch.inf).flatten()
                        top=values.topk(len(coords)).indices.cpu().tolist()
                        precision=sum((x//m,x%m) in coords for x in top)/len(coords)
                        contact.append({'pair_id':r['pair_id'],'precision_at_k':precision,'k':len(coords)})
            cursor+=len(indices)
    y=np.array([r['label'] for r in binary]); bs=scores[:len(binary)]
    metrics={'interaction_rows':len(binary),'positive':int(y.sum()),'negative':int((y==0).sum()),
             'random_ap':float(y.mean()),'contact_score_ap':float(average_precision_score(y,bs)),
             'contact_score_auroc':float(roc_auc_score(y,bs)),
             'clip_ap':float(average_precision_score(y,clip_scores)),
             'contact_structures':len(positives),'contact_evaluable':len(contact),
             'contact_precision_at_k':float(np.mean([r['precision_at_k'] for r in contact])),
             'evaluation':'full length; published orientation; FP32 weights, FP16 autocast; sigmoid(max valid logits)',
             'interaction_sha256':sha256(cfg['validation_interactions']),
             'structures_sha256':sha256(cfg['validation_structures'])}
    write_json(output/'metrics.json',metrics)
    write_json(output/'contact_metrics.json',contact)
    import csv
    with (output/'predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['pair_id','taxid','label','contact_score','clip_score'])
        for r,s,c in zip(binary,bs,clip_scores):
            w.writerow([r['pair_id'],r['taxid'],r['label'],s,c])
    print('VALIDATION',json.dumps(metrics),flush=True)
    clear_rotary_cache(model)
    return metrics


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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs_train/finetune_head_v1.json')
    p.add_argument('--smoke',action='store_true',help='Four optimizer steps and checkpoint round-trip; no formal training')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--stress-smoke',action='store_true',help='Smoke using pairs with both sequences >= max_len')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--extend-steps',type=int,help='Resume with a larger total step budget; all other configuration remains fixed')
    p.add_argument('--recovery-interval',type=int,help='Override recovery checkpoint frequency without changing the training configuration')
    args=p.parse_args()
    if args.recovery_interval is not None and args.recovery_interval <= 0:
        p.error('--recovery-interval must be positive')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory: '+str(args.output))
    cfg=json.loads(Path(args.config).read_text())
    recovery_interval = args.recovery_interval if args.recovery_interval is not None else cfg.get('recovery_interval')
    if args.extend_steps is not None:
        if args.resume is None or args.smoke or args.evaluate_only or args.extend_steps <= cfg['steps']:
            raise ValueError('--extend-steps requires a training resume and a larger total step budget')
        cfg['steps'] = args.extend_steps
    if cfg['mode'] not in ('head_only', 'lora'):
        raise ValueError('Expected head_only or lora mode')
    if args.stress_smoke and not args.smoke:
        raise ValueError('--stress-smoke requires --smoke')
    lora_mode = cfg['mode'] == 'lora'
    split_manifest = json.loads((Path(cfg['train']).parent/'manifest.json').read_text())
    for key in ('train','validation_structures'):
        path = Path(cfg[key])
        expected = split_manifest['outputs'][path.name]['sha256']
        if sha256(path) != expected:
            raise ValueError('Frozen split checksum mismatch: '+str(path))
    if sha256(Path(cfg['model'])/'model.safetensors') != '047edbe0570eca424ea835e0a460248d2aa5077d95a024d149ac22ab78a44a49':
        raise ValueError('Expected the pinned official no-E. coli checkpoint')
    if not args.smoke:
        val_manifest = json.loads((Path(cfg['validation_interactions']).parent/'manifest.json').read_text())
        if sha256(cfg['validation_interactions']) != val_manifest['benchmark_sha256']:
            raise ValueError('Frozen validation benchmark checksum mismatch')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(cfg.get('deterministic', False))
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    accelerator=Accelerator(mixed_precision='bf16')
    if accelerator.num_processes != 1 or accelerator.device.type != 'cuda':
        raise RuntimeError('This fixed configuration requires one CUDA GPU')
    load_path = cfg['model'] if lora_mode and args.resume and not args.evaluate_only else (args.resume or cfg['model'])
    model,tokenizer,loading=load_released(str(load_path))
    model.cuda()
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',cfg)
    import transformers, accelerate
    write_json(args.output/'environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,
        'accelerate':accelerate.__version__,'numpy':np.__version__,'training_precision':'bf16',
        'cuda_allocator_config':os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
        'config_sha256':sha256(args.config),'entrypoint_sha256':sha256(__file__),
        'recovery_interval':recovery_interval})
    write_json(args.output/'loading_info.json',loading)
    if args.evaluate_only:
        evaluate(model,tokenizer,cfg,args.output)
        return
    model.plm.requires_grad_(False)
    adapter_modules = install_lora(model,cfg) if lora_mode else []
    if lora_mode and args.resume:
        load_trainable(model,args.resume/'trainable.safetensors')
    if lora_mode:
        model.plm.encoder.gradient_checkpointing = True
    model.contact_head.transformer.gradient_checkpointing=True
    write_json(args.output/'trainable_parameters.json', {
        'mode':cfg['mode'],'adapter_modules':adapter_modules,
        'lora_parameters':sum(p.numel() for n,p in model.named_parameters() if '.lora_' in n),
        'total_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    cols=['seq1','seq2','contacts','seq1_cluster','seq2_cluster','valid1','valid2','pair_id']
    table=pq.read_table(cfg['train'],columns=cols)
    c1=table['seq1_cluster'].to_numpy();c2=table['seq2_cluster'].to_numpy()
    keys,multiplier=build_interaction_keys(c1,c2,np.array([],dtype=np.int64),np.array([],dtype=np.int64))
    if cfg.get('crop'):
        # Build negative-exclusion keys BEFORE filtering. Missing structure
        # labels do not make a known biological interaction a safe negative.
        counts = valid_contact_counts(table)
        keep = np.flatnonzero(counts > 0)
        if not len(keep): raise ValueError('No valid contact-supervised training rows')
        write_json(args.output/'training_population.json', {
            'original_rows':len(table),'eligible_rows':len(keep),'excluded_no_valid_contact':int((counts==0).sum()),
            'known_positive_mask':'all original training rows retained',
            'crop':cfg['crop']})
        if cfg.get('eligible_train_rows',len(keep)) != len(keep):
            raise ValueError('Eligible training population differs from fixed config')
        table=table.take(keep)
        c1=table['seq1_cluster'].to_numpy();c2=table['seq2_cluster'].to_numpy()
    wrapper=ReleasedTrainingModel(model,keys.cuda(),cfg).cuda()
    optimizer=torch.optim.AdamW([x for x in wrapper.parameters() if x.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    collator=DataCollatorForPPI(tokenizer,cfg['max_len'],swap_augment=True,adds_special_tokens=False,crop_config=cfg.get('crop'))
    sampler=ClusterSampler(c1,c2,ddi_sampling_prob=0,seed=seed)
    total_steps=cfg['smoke_steps'] if args.smoke else cfg['steps']
    batch_size=cfg['smoke_batch_size'] if args.smoke else cfg['batch_size']
    epochs = cfg.get('epochs') if not args.smoke else None
    steps_per_epoch = math.ceil(len(table)/batch_size) if epochs else None
    if epochs and total_steps != epochs*steps_per_epoch:
        raise ValueError('Configured steps must equal epochs * ceil(train_rows / batch_size)')
    sampler.num_samples=epochs*len(table) if epochs else total_steps*batch_size
    sampled=list(sampler)
    if args.stress_smoke:
        import pyarrow.compute as pc
        eligible = np.flatnonzero((pc.utf8_length(table['seq1']).to_numpy() >= cfg['max_len']) &
                                  (pc.utf8_length(table['seq2']).to_numpy() >= cfg['max_len']))
        if not len(eligible):
            raise ValueError('No full-length pairs available for stress smoke')
        sampled = np.random.default_rng(seed).choice(eligible,len(sampled),replace=True).tolist()
    gc=PPIGradientCache(wrapper,cfg['chunk_size'],GradCacheClipLoss(valid_keys_tensor=keys.cuda()),
                        ContactLoss(use_focal_loss=True),cfg['contact_loss_weight'],accelerator,cfg['clip_loss_weight'])
    # Exact wrapper parity before any updates, with dropout disabled.
    row=table.slice(sampled[0],1).to_pylist()[0]
    fixed=DataCollatorForPPI(tokenizer,64,False,False)([row])
    a={k:v.cuda() for k,v in fixed['seq1_tokens'].items()};b={k:v.cuda() for k,v in fixed['seq2_tokens'].items()}
    wrapper.eval();model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        original=model(a['input_ids'],b['input_ids'],a['attention_mask'],b['attention_mask'])
        out=wrapper(a['input_ids'],b['input_ids'],a['attention_mask'],b['attention_mask'])
        delta=float((original.contact_map-torch.sigmoid(out[2])).abs().max())
    if delta != 0:
        raise AssertionError(f'Initial prediction changed: {delta}')
    clear_rotary_cache(model)
    if lora_mode and not args.resume:
        adapters = [m for m in model.modules() if isinstance(m,QVLoRALinear)]
        for module in adapters: module.enabled=False
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            plain=model(a['input_ids'],b['input_ids'],a['attention_mask'],b['attention_mask']).contact_map
        for module in adapters: module.enabled=True
        if not torch.equal(plain,original.contact_map):
            raise AssertionError('Zero-initialized LoRA changed initial predictions')
    clear_rotary_cache(model)
    backbone_before={n:hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for n,t in model.plm.named_parameters() if not t.requires_grad}
    adapter_before={n:t.detach().clone() for n,t in model.plm.named_parameters() if '.lora_' in n}
    head_before=model.contact_head.output_mix.weight.detach().clone()
    start=0;best=-1.;history=[]
    if args.resume:
        state=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=False)
        validate_resume_config(state,cfg,args.smoke,extend=args.extend_steps is not None)
        optimizer.load_state_dict(state['optimizer']);start=state['step'];best=state['best_ap']
        previous_best_path = args.resume.parent/'best.json'
        if previous_best_path.exists():
            previous_best = json.loads(previous_best_path.read_text())
            previous_best['checkpoint'] = str((args.resume.parent/previous_best['checkpoint']).resolve())
            write_json(args.output/'best.json',previous_best)
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    def save(path, step):
        path.mkdir()
        if lora_mode:
            model.save_pretrained(path,state_dict=merged_state_dict(model),safe_serialization=True,max_shard_size='5GB')
            save_trainable(model,path/'trainable.safetensors')
            write_json(path/'adapter_config.json',{'base_model':cfg['model'],'lora':cfg['lora'],
                'format':'model.safetensors is merged for official inference; trainable.safetensors contains unmerged adapters and heads'})
        else:
            model.save_pretrained(path,safe_serialization=True,max_shard_size='5GB')
        tokenizer.save_pretrained(path)
        torch.save({'optimizer':optimizer.state_dict(),'step':step,'config':cfg,'smoke':args.smoke,'best_ap':best,
                    'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                    'recovery_interval':recovery_interval},path/'training_state.pt')
    for step in range(start,total_steps):
        wrapper.train();optimizer.zero_grad(set_to_none=True)
        left,right = sample_window(step,batch_size,len(table) if epochs else None)
        batch=collator(table.take(sampled[left:right]).to_pylist())
        warmup=min(1.,(step+1)/cfg['warmup_steps'])
        for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*warmup
        tick=time.time();cl,ct,scale,acc=gc.step(batch)
        norm=torch.nn.utils.clip_grad_norm_([x for x in wrapper.parameters() if x.requires_grad],cfg['max_grad_norm'],error_if_nonfinite=True)
        if not np.isfinite([cl,ct,scale,float(norm)]).all():raise FloatingPointError('Nonfinite training')
        if any(x.grad is not None for x in model.plm.parameters() if not x.requires_grad):raise AssertionError('Frozen backbone acquired gradient')
        adapter_grad_norm = float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for n,p in model.plm.named_parameters() if '.lora_' in n and p.grad is not None]))) if lora_mode else 0.0
        if lora_mode and (not np.isfinite(adapter_grad_norm) or adapter_grad_norm <= 0):
            raise AssertionError('LoRA did not receive a finite nonzero gradient')
        optimizer.step()
        rec={'step':step+1,'clip_loss':cl,'contact_loss':ct,'total_loss':cfg['clip_loss_weight']*cl+cfg['contact_loss_weight']*ct,
             'grad_norm':float(norm),'seconds':time.time()-tick,'lr':optimizer.param_groups[0]['lr'],'lora_grad_norm_after_clip':adapter_grad_norm,
             'positive_draws_this_step':right-left,'epoch':(step//steps_per_epoch+1) if epochs else None}
        rec.update(peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                   peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9)
        if cfg.get('crop'):
            metadata=batch['crop_metadata']
            rec['crop_min_contacts']=min(m['retained_contacts'] for m in metadata)
            rec['crop_mean_retained_fraction']=float(np.mean([m['retained_fraction'] for m in metadata]))
        history.append(rec);print('TRAIN',json.dumps(rec),flush=True)
        write_json(args.output/'history.json',history)
        if args.smoke and step+1 == cfg.get('smoke_resume_step', total_steps//2):
            save(args.output/'resume_probe',step+1)
        epoch_end = bool(epochs and (step+1)%steps_per_epoch == 0)
        if not args.smoke and ((step+1)%cfg['eval_interval']==0 or step+1==total_steps or epoch_end or step+1 in cfg.get('early_eval_steps', [])):
            metrics=evaluate(model,tokenizer,cfg,args.output/f'val_step_{step+1:06d}')
            if metrics['contact_score_ap']>best:
                best=metrics['contact_score_ap']
                save(args.output/f'best_step_{step+1:06d}',step+1)
                write_json(args.output/'best.json',{'step':step+1,'ap':best,'checkpoint':f'best_step_{step+1:06d}'})
        if epoch_end:
            save(args.output/f'epoch_{(step+1)//steps_per_epoch:02d}',step+1)
        if not args.smoke and recovery_interval is not None and (step+1)%recovery_interval==0 and step+1<total_steps:
            # Replace only our own completed recovery checkpoint after the new
            # checkpoint is fully written; retain best and epoch checkpoints.
            temporary=args.output/'latest_new'
            save(temporary,step+1)
            latest=args.output/'latest'
            previous=args.output/'latest_previous'
            if latest.exists(): latest.rename(previous)
            temporary.rename(latest)
            if previous.exists(): shutil.rmtree(previous)
    save(args.output/'last',total_steps)
    backbone_after={n:hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for n,t in model.plm.named_parameters() if not t.requires_grad}
    adapter_updated = any(not torch.equal(adapter_before[n],t) for n,t in model.plm.named_parameters() if n in adapter_before)
    if lora_mode and not adapter_updated:raise AssertionError('LoRA parameters did not update')
    if backbone_before != backbone_after:raise AssertionError('Frozen backbone changed')
    if not history and args.resume:
        raise ValueError('Resume checkpoint already reached configured step limit')
    if torch.equal(head_before,model.contact_head.output_mix.weight):raise AssertionError('Contact head did not update')
    model.eval()
    clear_rotary_cache(model)
    with torch.no_grad(),merged_for_inference(model),torch.autocast('cuda',dtype=torch.float16):
        expected=model(a['input_ids'],b['input_ids'],a['attention_mask'],b['attention_mask']).contact_map.float().cpu()
    reloaded,_,reload_info=load_released(str(args.output/'last'));reloaded.cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        actual=reloaded(a['input_ids'],b['input_ids'],a['attention_mask'],b['attention_mask']).contact_map.float().cpu()
    reload_delta=float((expected-actual).abs().max())
    if reload_delta != 0:raise AssertionError(f'Save/reload changed prediction: {reload_delta}')
    write_json(args.output/'checks.json',{'initial_wrapper_max_delta':delta,'save_reload_max_delta':reload_delta,
        'frozen_backbone_unchanged':True,'contact_head_updated':True,'finite_gradients':True,
        'mode':cfg['mode'],'lora_updated':adapter_updated,'lora_modules':adapter_modules,
        'stress_full_length':args.stress_smoke,'peak_reserved_cuda_gb':torch.cuda.max_memory_reserved()/1e9,
        'steps':total_steps,'gpu':torch.cuda.get_device_name(0),'peak_cuda_gb':torch.cuda.max_memory_allocated()/1e9,
        'model_sha256':sha256(Path(cfg['model'])/'model.safetensors'),
        'train_sha256':sha256(cfg['train']),'reload_loading_info':reload_info,
        'contact_objective':'example-weighted mean of microbatch focal losses; fixed chunk size',
        'train_truncation':cfg.get('crop', 'first 512 residues'), 'validation':'always full length'})

if __name__=='__main__':main()
