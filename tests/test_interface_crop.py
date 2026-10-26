"""Interface crop coordinates, masks, sampling diversity and reproducibility."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from piper.cropping import interface_crop, valid_contacts, valid_contact_counts
from piper.dataset import DataCollatorForPPI, build_interaction_keys
from test_finetune_training import Tokenizer


class Tests(unittest.TestCase):
    def row(self):
        return dict(seq1='ACDEFGHIKL'*100,seq2='MNPQRSTVWY'*120,
                    contacts=[[a,b] for a in range(800,810) for b in range(900,910)]+[[0,0]],
                    valid1=[False]+[True]*999,valid2=[False]+[True]*1199,
                    seq1_cluster=3,seq2_cluster=7)

    def test_far_interface_coordinates_masks_and_no_mutation(self):
        row=self.row();original=copy.deepcopy(row)
        crop=interface_crop(row,512,generator=torch.Generator().manual_seed(1))
        meta=crop['crop_metadata'];a,b=meta['start1'],meta['start2']
        self.assertEqual(row,original)
        self.assertEqual(crop['seq1'],row['seq1'][a:a+512])
        self.assertEqual(crop['seq2'],row['seq2'][b:b+512])
        self.assertEqual(crop['valid1'],row['valid1'][a:a+512])
        self.assertEqual(crop['valid2'],row['valid2'][b:b+512])
        expected={(x-a,y-b) for x,y in row['contacts'] if a<=x<a+512 and b<=y<b+512 and row['valid1'][x] and row['valid2'][y]}
        self.assertEqual(set(map(tuple,crop['contacts'])),expected)
        self.assertEqual(meta['original_contacts'],100)
        self.assertEqual(meta['retained_contacts'],100)

    def test_short_chain_and_boundary_contact(self):
        row=dict(seq1='A'*10,seq2='G'*1000,contacts=[[9,999]],valid1=[True]*10,valid2=[True]*1000)
        crop=interface_crop(row,512)
        self.assertEqual(crop['seq1'],row['seq1'])
        self.assertEqual(crop['crop_metadata']['start1'],0)
        self.assertEqual(crop['contacts'],[[9,511]])

    def test_replay_and_multiple_interfaces(self):
        row=self.row();row['contacts']=[[10,10],[990,1190]]
        torch.manual_seed(42);state=torch.get_rng_state()
        first=[interface_crop(row,128) for _ in range(30)]
        torch.set_rng_state(state)
        self.assertEqual(first,[interface_crop(row,128) for _ in range(30)])
        self.assertTrue(any(c['crop_metadata']['start1']<100 for c in first))
        self.assertTrue(any(c['crop_metadata']['start1']>800 for c in first))
        self.assertTrue(all(len(c['contacts'])>0 for c in first))

    def test_empty_invalid_duplicate_and_out_of_bounds(self):
        row=self.row();row['contacts']=[[0,0]]
        with self.assertRaises(ValueError):interface_crop(row,512)
        row['contacts']=[[1000,1]]
        with self.assertRaises(ValueError):interface_crop(row,512)
        row['contacts']=[[800,900]]*3
        self.assertEqual(interface_crop(row,512)['crop_metadata']['original_contacts'],1)

    def test_collator_swap_and_special_token_budget(self):
        row=self.row();cfg=dict(mode='interface',candidates=16,near_best_fraction=.9)
        with patch('torch.rand',return_value=torch.tensor([0.])):
            out=DataCollatorForPPI(Tokenizer(),514,True,True,crop_config=cfg)([row])
        meta=out['crop_metadata'][0]
        expected={(y-meta['start1'],x-meta['start2']) for x,y in row['contacts'] if x>0 and y>0}
        actual=set(map(tuple,out['contact_target'][0].nonzero().tolist()))
        self.assertEqual(actual,expected)
        self.assertEqual(out['seq1_cluster'].item(),7)
        self.assertEqual(out['contact_target'].shape,(1,512,512))
        self.assertTrue(out['contact_target_valid_mask'][out['contact_target']].all())

    def test_vector_counts_slice_and_excluded_pairs_stay_known(self):
        schema=pa.schema([('contacts',pa.list_(pa.list_(pa.int32(),2))),('valid1',pa.list_(pa.bool_())),('valid2',pa.list_(pa.bool_()))])
        rows=[dict(contacts=[[0,0],[1,1]],valid1=[True,False],valid2=[True,True]),
              dict(contacts=[],valid1=[True],valid2=[True]),
              dict(contacts=[[0,0]],valid1=[True],valid2=[True])]
        table=pa.Table.from_pylist(rows,schema=schema)
        np.testing.assert_array_equal(valid_contact_counts(table),[1,0,1])
        np.testing.assert_array_equal(valid_contact_counts(table.slice(1,2)),[0,1])
        keys,mult=build_interaction_keys(np.array([1,3,5]),np.array([2,4,6]),np.array([],dtype=int),np.array([],dtype=int))
        self.assertIn(3*mult+4,keys.tolist())


if __name__=='__main__':unittest.main()
