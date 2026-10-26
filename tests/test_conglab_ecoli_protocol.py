"""Boundary and pairing regressions for the E.coli-only source audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import pyarrow as pa
from audit_conglab_ecoli import classify_hit, matched_sides, unordered_origin


def test_exact_boundaries_and_domain_coverage():
    # Strict identity >30%, strict both-chain coverage >50%.
    assert classify_hit([30,100,1,100,150,1,100,120]) == (False,False,False)
    assert classify_hit([31,100,1,100,200,1,100,120]) == (False,True,True)
    assert classify_hit([31,100,1,100,150,1,100,125]) == (True,True,True)
    # A domain covers 80% of its own length but only 10% of a test protein.
    assert classify_hit([31,100,1,100,1000,1,100,125]) == (False,True,True)


def test_two_chains_means_both_endpoints_not_either():
    table=pa.table({'seq1_id':['a','a','c'],'seq2_id':['b','c','d']})
    a,b=matched_sides(table,{'a','b'})
    assert (~(a&b)).tolist() == [False,True,True]
    assert (~(a|b)).tolist() == [False,False,True]


def test_original_chain_case_is_preserved():
    assert unordered_origin('1abc_A:1abc_b') == unordered_origin('1abc_b:1abc_A')
    assert unordered_origin('1abc_A:1abc_b') != unordered_origin('1abc_A:1abc_B')


def test_separate_lora_head_rates_and_cosine():
    import torch
    from flashppi.research_train import build_optimizer,set_step_learning_rate
    model=torch.nn.Module()
    model.adapter=torch.nn.Module()
    model.adapter.lora_b_q=torch.nn.Parameter(torch.ones(2,2))
    model.head=torch.nn.Linear(2,1)
    config=dict(learning_rate=5e-4,lora_learning_rate=1e-5,weight_decay=.1,
                learning_rate_schedule='cosine',steps=100,warmup_steps=5)
    optimizer=build_optimizer(model,config)
    set_step_learning_rate(optimizer,config,0)
    assert abs(optimizer.param_groups[0]['lr']-2e-6)<1e-12
    assert abs(optimizer.param_groups[1]['lr']-1e-4)<1e-12
    set_step_learning_rate(optimizer,config,4)
    assert optimizer.param_groups[0]['lr']==1e-5
    assert optimizer.param_groups[1]['lr']==5e-4
    set_step_learning_rate(optimizer,config,99)
    assert all(g['lr']==0 for g in optimizer.param_groups)
