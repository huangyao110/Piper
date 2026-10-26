"""The user-approved old filtering protocol must not drift during expansion."""
import copy
import json
from pathlib import Path
import sys

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flashppi.data_protocol import (PROTOCOL_ID, read_lock, validate_config,
    restore_known_decisions, locked_command, verify_search_log)
from flashppi.indexed_data import file_sha256


def configuration():
    return dict(filter_protocol=PROTOCOL_ID, test_exclusion='both_chains',
                val_exclusion='either_chain', ppi_coverage='both_coverage_50',
                ddi_coverage='candidate_coverage_80', preserve_all_legacy=True)


@pytest.mark.parametrize('key,value', [
    ('test_exclusion','either_chain'), ('val_exclusion','both_chains'),
    ('ppi_coverage','candidate_coverage_80'), ('ddi_coverage','both_coverage_50'),
    ('preserve_all_legacy',False), ('filter_protocol','another_version'),
    ('kmer_length',7), ('search_parameters',{})])
def test_filter_changes_are_rejected(key, value):
    config = {**configuration(), key: value}
    with pytest.raises(ValueError, match='protocol'):
        validate_config(config)


def test_missing_protocol_is_rejected():
    config = configuration()
    config.pop('filter_protocol')
    with pytest.raises(ValueError, match='protocol'):
        validate_config(config)


def test_pin_is_derived_from_original_observed_search():
    lock = read_lock()
    assert lock['search']['kmer_length'] == 6
    assert lock['search']['kmer_score'] == 96
    assert lock['search']['mmseqs_version'] == '18.8cc5c'
    previous = dict(command=['mmseqs','easy-search','<TEMP>/q','<TEMP>/t','<TEMP>/a','<TEMP>/tmp',
                            '--min-seq-id','0.3','--max-seqs','100','-s','7.5'])
    original = copy.deepcopy(previous)
    command = locked_command(previous,'pinned-mmseqs','q','t','a','tmp',500)
    assert previous == original
    assert command[command.index('-k')+1] == '6'
    assert command[command.index('--k-score')+1] == 'seq:96,prof:2147483647'
    assert command[command.index('--max-seqs')+1] == '500'
    assert command[command.index('--min-seq-id')+1] == '0.3'


def test_original_filter_scripts_are_unchanged():
    files=read_lock()['baseline_files']
    for name, expected in files.items():
        if name.startswith('tools/'):
            assert file_sha256(ROOT/name) == expected, name


def test_known_positive_and_negative_decisions_both_survive_expansion():
    decisions = pa.table({'seq_id':['old_hit','old_nonhit','outside_val_scope'],
                          'test_known':[True,True,True],'val_known':[True,True,False],
                          'test_hit':[True,False,False],'val_hit':[True,False,False]})
    # Expanded search misses a former hit and newly detects an old non-hit.
    observed = {'old_nonhit','outside_val_scope','novel_hit'}
    assert restore_known_decisions(observed, decisions, 'test') == {'old_hit','novel_hit'}
    assert restore_known_decisions(observed, decisions, 'val') == {
        'old_hit','outside_val_scope','novel_hit'}


def test_actual_search_drift_is_rejected(tmp_path):
    path = tmp_path / 'search.log'
    path.write_text('MMseqs Version: 18.8cc5c\nIndex table k-mer threshold: 96 at k-mer size 6\n')
    verify_search_log(path)
    for text in ['MMseqs Version: 18.8cc5c\nIndex table k-mer threshold: 101 at k-mer size 7\n',
                 'MMseqs Version: another\nIndex table k-mer threshold: 96 at k-mer size 6\n',
                 'no completed prefilter output']:
        path.write_text(text)
        with pytest.raises(ValueError):
            verify_search_log(path)


def test_strict_integer_boundaries_and_directional_domain_coverage():
    from flashppi.data_protocol import strict_hit, strict_pair_decision
    # Exactly 50% on both sides counts; exactly 30% identity does not.
    f = ['ref','train','30','100','1','50','100','101','150','1000']
    assert not strict_hit(f, 'ppi', 'test_to_train')
    f[2] = '31'
    f[7:] = ['1','50','100']
    assert strict_hit(f, 'ppi', 'test_to_train')
    f[2] = '30'
    assert not strict_hit(f, 'ppi', 'test_to_train')
    # A 100-residue training domain aligned over 50 residues of a 1000-aa
    # reference passes DDI coverage but not PPI bidirectional coverage.
    f = ['ref','train','31','50','1','50','1000','1','50','100']
    assert strict_hit(f, 'ddi', 'test_to_train')
    assert not strict_hit(f, 'ppi', 'test_to_train')
    swapped = [f[1],f[0],*f[2:4],*f[7:10],*f[4:7]]
    assert strict_hit(swapped, 'ddi', 'train_to_test')
    assert not strict_hit(swapped, 'ppi', 'train_to_test')
    f[8] = '49'
    assert not strict_hit(f, 'ddi', 'test_to_train')
    assert strict_pair_decision(True, False) == 'test_either_endpoint_homology'
    assert strict_pair_decision(False, True) == 'test_either_endpoint_homology'
    assert strict_pair_decision(False, False) == 'candidate'


def test_strict_lock_rejects_parameter_source_and_scope_drift():
    from flashppi.data_protocol import read_strict_lock, validate_strict_config, strict_command
    lock = read_strict_lock()
    config = copy.deepcopy(lock['required_config'])
    validate_strict_config(config)
    for change in [dict(threads=4),dict(validation='val'),dict(coverage_basis='cropped_sequences'),
                   dict(sensitivity=7),dict(allowed_sources={'ppi':['pinder'],'ddi':['teddymer']}),
                   dict(preserve_all_legacy=True),dict(test_exclusion='both_chains')]:
        with pytest.raises(ValueError):
            validate_strict_config({**config,**change})
    a = strict_command('mmseqs','q','t','out','tmp',772,'ddi','test_to_train')
    b = strict_command('mmseqs','q','t','out','tmp',123,'ddi','train_to_test')
    assert a[a.index('--cov-mode')+1] == '1'
    assert b[b.index('--cov-mode')+1] == '2'
    assert a[a.index('--max-seqs')+1] == '772'
    assert b[b.index('--max-seqs')+1] == '123'
    assert a[a.index('-k')+1] == '6'
    assert a[a.index('--shuffle')+1] == '0'
    for direction in ('test_to_train_nofilter','train_to_test_nofilter'):
        command = strict_command('mmseqs','q','t','out','tmp',772,'ddi',direction)
        assert command[command.index('--prefilter-mode')+1]=='2'


@pytest.mark.parametrize('kind', ['ppi', 'ddi'])
@pytest.mark.parametrize('direction', ['test_to_train', 'train_to_test_nofilter'])
def test_rf2ppi_significance_coverage_or_and_strict_boundaries(kind, direction):
    from flashppi.data_protocol import rf2ppi_hit
    # 51/100 on the short sequence, 51/1000 on the long one: either side suffices.
    f=['q','t','31','51','1','51','100','1','51','1000','9.99e-6','40']
    assert rf2ppi_hit(f,kind,direction)
    swapped=[f[1],f[0],*f[2:4],*f[7:10],*f[4:7],*f[10:]]
    assert rf2ppi_hit(swapped,kind,direction)
    f[5]='50'; f[8]='50'; f[3]='50'
    assert not rf2ppi_hit(f,kind,direction)  # exactly 50% fails
    f[5]='51'; f[8]='51'; f[3]='100'; f[2]='30'
    assert not rf2ppi_hit(f,kind,direction)  # exactly 30% fails
    f[2]='31'; f[10]='1e-5'
    assert not rf2ppi_hit(f,kind,direction)  # exactly 1e-5 fails
    f[10]='9.99e-6'
    assert rf2ppi_hit(f,kind,direction)
    # Even a 100% identical short match must pass statistical significance.
    f=['q','t','5','5','1','5','5','1','5','5','0.1','10']
    assert not rf2ppi_hit(f,kind,direction)
    for invalid in ('nan','inf','-1'):
        f[10]=invalid
        with pytest.raises(ValueError):
            rf2ppi_hit(f,kind,direction)


def test_rf2ppi_empty_mmseqs_results_are_not_significant_hits():
    from flashppi.data_protocol import rf2ppi_hit
    # Observed with MMseqs 18.8cc5c, A*139 versus the complete E. coli reference.
    # A zero-residue alignment has no defined identity or significance.
    f=['q','t','0','0','0','1','139','0','0','30','6.868E-322','4']
    for kind in ('ppi','ddi'):
        for direction in ('test_to_train','train_to_test_nofilter'):
            assert not rf2ppi_hit(f,kind,direction)
    f[10]='nan'
    assert not rf2ppi_hit(f,'ppi','train_to_test_nofilter')
    for pos,value in ((2,'1'),(3,'1'),(4,'1'),(5,'2'),(6,'0'),(9,'0')):
        malformed=f.copy();malformed[pos]=value
        with pytest.raises(ValueError):
            rf2ppi_hit(malformed,'ppi','train_to_test_nofilter')


def test_rf2ppi_pair_rule_and_locked_search_settings():
    from flashppi.data_protocol import (RF2_PROTOCOL_ID, read_strict_lock, strict_pair_decision,
                                       validate_strict_config, strict_command)
    for h1,h2 in ((False,False),(True,False),(False,True),(True,True)):
        assert (strict_pair_decision(h1,h2,RF2_PROTOCOL_ID)!='candidate') == (h1 and h2)
    lock=read_strict_lock(RF2_PROTOCOL_ID)
    config=copy.deepcopy(lock['required_config'])
    validate_strict_config(config)
    for key,value in [('test_exclusion','either_chain'),('ppi_coverage','both_coverage_gt_50'),
                      ('ddi_coverage','training_domain_coverage_gte_50'),('evalue','none')]:
        with pytest.raises(ValueError):
            validate_strict_config({**config,key:value})
    for direction in lock['search']['directions']:
        command=strict_command('mmseqs','q','t','out','tmp',772,'ppi',direction,RF2_PROTOCOL_ID)
        assert command[command.index('-c')+1]=='0'  # cannot prefilter with coverage AND
        assert command[command.index('-e')+1]=='0.00001'
        assert command[command.index('--format-output')+1].endswith(',evalue,bits')
        assert command[command.index('--max-seqs')+1]=='772'


@pytest.mark.integration
def test_strict_direct_pass_recovers_historical_verified_homologs(tmp_path):
    """Regression against an observed k-mer miss, using real CPU MMseqs.

    Requires the local historical diagnostic evidence and frozen E. coli file;
    does not download datasets and does not require CUDA/model weights.
    """
    import subprocess
    import pyarrow.parquet as pq
    from flashppi.data_protocol import strict_command, strict_hit, reference_fasta, verify_strict_search_log
    folder=ROOT/'experiments/homology_consistency_20261022'
    candidate=folder/'disputed_domains.fasta'
    evidence=folder/'nofilter_accepted.parquet'
    test=ROOT/'artifacts/flashppi_official/ecoli_pdb_benchmark/data/train-00000-of-00001.parquet'
    binary=Path('/home/zlab/miniconda3/bin/mmseqs')
    if not all(p.exists() for p in (candidate,evidence,test,binary)):
        pytest.skip('Local historical evidence/frozen benchmark/MMseqs required')
    known={r['seq_id'] for r in pq.read_table(evidence).to_pylist() if r['category']=='test'}
    rows=pq.read_table(test,columns=['seq1','seq2'])
    seqs=set(rows['seq1'].to_pylist()+rows['seq2'].to_pylist())
    reference=tmp_path/'reference.fasta';reference.write_text(reference_fasta(seqs))
    count=sum(s.startswith('>') for s in candidate.open())
    hits=set()
    for direction in ('test_to_train_nofilter','train_to_test_nofilter'):
        query,target,n=(reference,candidate,count) if direction.startswith('test_to_train') else (candidate,reference,len(seqs))
        path=tmp_path/(direction+'.tsv');log=tmp_path/(direction+'.log')
        cmd=strict_command(binary,query,target,path,tmp_path/(direction+'_tmp'),n,'ddi',direction)
        with log.open('w') as handle:
            subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,check=True)
        verify_strict_search_log(log,direction)
        for line in path.open():
            f=line.rstrip('\n').split('\t')
            if strict_hit(f,'ddi',direction):
                hits.add(f[1] if direction.startswith('test_to_train') else f[0])
    assert known <= hits
