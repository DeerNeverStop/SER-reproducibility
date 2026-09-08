"""Synthetic metadata/mutation tests; no audio, model, prediction or GPU use."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from v3.final_program_20260907 import plan


def synthetic_metadata():
    metas = {}
    for corpus in plan.CORPUS_ORDER:
        design = plan.DESIGNS[corpus]
        people = [f'SYN_{i:03d}' for i in range(design['speakers'])]
        sex = {s:'F' if i%2 == 0 else 'M' for i,s in enumerate(people)}
        prompts = [f'S{p}' for p in range(1,13 if corpus=='cremad' else 11 if corpus=='subesco' else 3)]
        rows, cells = {}, {}
        for speaker in people:
            for prompt in prompts:
                for label in range(design['n_classes']):
                    path = f'SYNTHETIC_ONLY/{speaker}_{prompt}_{label}.wav'
                    row = dict(corpus=corpus,relative_path=path,speaker=speaker,sex=sex[speaker],
                               sex_stratum=sex[speaker],sentence=prompt,label_index=label,
                               label=plan.CORPORA[corpus].labels[label],
                               take='XX' if corpus=='cremad' else 'T1' if corpus=='subesco' else 'R1',
                               intensity='XX' if corpus=='cremad' else '' if corpus=='subesco' else 'normal',
                               bytes=1,sha256='0'*64,sample_index=len(rows))
                    rows[path]=row;cells[speaker,prompt,label]=row
        metas[corpus]=dict(rows=rows,cells=cells,sex=sex,speakers=people,prompts=prompts,hygiene={})
    return metas, {'synthetic': True, 'no_audio_or_real_model': True}


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas,cls.inputs=synthetic_metadata()
        with patch.object(plan,'_load_metadata',return_value=(cls.metas,cls.inputs)):
            cls.sample=plan.generate(Path('.'))

    def validate(self, value):
        with patch.object(plan,'_load_metadata',return_value=(self.metas,self.inputs)):
            return plan.validate(value,Path('.'))

    def test_exact_native_matrix_and_role_budgets(self):
        self.assertTrue(self.validate(self.sample))
        units=self.sample['units']
        self.assertEqual(Counter(u['phase'] for u in units),Counter(formal=384,pilot=4))
        for corpus,classes,fit_size,query_size in [('cremad',6,576,288),('subesco',7,224,112),('ravdess',8,64,64)]:
            primary=[u for u in units if u['phase']=='formal' and u['corpus']==corpus and u['arm']=='A']
            self.assertEqual(len(primary),120)
            for u in primary:
                self.assertEqual(u['n_classes'],classes)
                self.assertEqual(len(u['fit']),fit_size)
                self.assertEqual(len(u['report']['A']),query_size)
                self.assertEqual(len(u['report']['B']),query_size)
                self.assertEqual({self.metas[corpus]['rows'][p]['label_index'] for p in u['fit']},set(range(classes)))
                self.assertEqual(u['prediction_epochs'],list(range(1,16)))
        controls=[u for u in units if u['phase']=='formal' and u['arm']=='B']
        self.assertEqual(len(controls),24)
        self.assertTrue(all(u['corpus']=='cremad' and u['fold']==0 and not u['selection_enabled'] and u['prediction_epochs']==[15] for u in controls))

    def test_all_people_outer_exactly_once_per_draw(self):
        for corpus in plan.CORPUS_ORDER:
            for draw in range(24):
                selected=[u for u in self.sample['units'] if u['phase']=='formal' and u['arm']=='A' and u['corpus']==corpus and u['draw']==draw]
                self.assertEqual(Counter(s for u in selected for s in u['test_speakers']),Counter(self.metas[corpus]['speakers']))

    def test_pair_seeds_and_slot_person_rank_are_matched(self):
        units=self.sample['units']
        for b in [u for u in units if u['arm']=='B']:
            a=next(u for u in units if u['pair_id']==b['pair_id'] and u['arm']=='A')
            self.assertEqual(a['seeds'],b['seeds'])
            self.assertEqual(a['report_batches'],b['report_batches'])
            pairs=set()
            for pa,pb in zip(a['fit'],b['fit']):
                ra,rb=self.metas['cremad']['rows'][pa],self.metas['cremad']['rows'][pb]
                self.assertEqual(tuple(ra[k] for k in ('sex_stratum','sentence','label_index','take','intensity')),
                                 tuple(rb[k] for k in ('sex_stratum','sentence','label_index','take','intensity')))
                pairs.add((ra['speaker'],rb['speaker']))
            self.assertEqual(len(pairs),24)
            self.assertEqual(len({p[0] for p in pairs}),24)
            self.assertEqual(len({p[1] for p in pairs}),24)

    def test_no_outer_record_in_validation_batch(self):
        for u in self.sample['units']:
            outer=set(u['report']['outer'])
            for role in ('A','B'):
                self.assertFalse(outer & {p for batch in u['report_batches'][role] for p in batch})
                self.assertEqual(u['report_batches'][role], [u['report'][role][i:i+16] for i in range(0,len(u['report'][role]),16)])

    def test_main_and_pilot_randomization_are_separate(self):
        for pilot in [u for u in self.sample['units'] if u['phase']=='pilot']:
            formal=next(u for u in self.sample['units'] if u['phase']=='formal' and
                        (u['corpus'],u['draw'],u['fold'],u['arm'])==(pilot['corpus'],0,0,pilot['arm']))
            self.assertNotEqual(pilot['unit_id'],formal['unit_id'])
            self.assertTrue(all(pilot['seeds'][k]!=formal['seeds'][k] for k in pilot['seeds']))

    def test_semantic_mutations_fail_even_with_new_self_hash(self):
        mutations=[
            lambda p:p.__setitem__('program','WRONG'),
            lambda p:p['analysis'].__setitem__('family_size',1),
            lambda p:p['analysis'].__setitem__('bootstrap',True),
            lambda p:p['config'].__setitem__('epochs',14),
            lambda p:p['units'][0]['seeds'].__setitem__('crop',7),
            lambda p:p['units'][0].__setitem__('unit_id','WRONG'),
            lambda p:p['units'][0].__setitem__('seen_group','outer'),
            lambda p:p['units'][0].__setitem__('prediction_epochs',
                [15] if p['units'][0]['arm']=='A' else list(range(1,16))),
            lambda p:p['units'][0].__setitem__('arm_execution_order',list(reversed(p['units'][0]['arm_execution_order']))),
        ]
        for mutate in mutations:
            changed=deepcopy(self.sample);mutate(changed)
            changed['plan_sha256']=plan.digest({k:v for k,v in changed.items() if k!='plan_sha256'})
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):self.validate(changed)

    def test_partial_slot_swap_is_rejected_despite_same_class_quota(self):
        changed=deepcopy(self.sample)
        u=next(u for u in changed['units'] if u['phase']=='formal' and u['corpus']=='cremad' and u['fold']==1)
        # Same sex, prompt, class, take; swap only two person positions at one slot.
        u['fit'][0],u['fit'][24]=u['fit'][24],u['fit'][0]
        changed['plan_sha256']=plan.digest({k:v for k,v in changed.items() if k!='plan_sha256'})
        with self.assertRaises(ValueError):self.validate(changed)

    def test_outer_person_missing_class_is_rejected(self):
        changed=deepcopy(self.sample);u=changed['units'][0]
        rows=self.metas[u['corpus']]['rows'];person=rows[u['report']['outer'][0]]['speaker']
        u['report']['outer']=[p for p in u['report']['outer'] if not (rows[p]['speaker']==person and rows[p]['label_index']==0)]
        u['report_batches']['outer']=[u['report']['outer'][i:i+16] for i in range(0,len(u['report']['outer']),16)]
        with self.assertRaisesRegex(ValueError,'lacks native class'):self.validate(changed)

    def test_mixed_outer_batch_is_rejected(self):
        changed=deepcopy(self.sample);u=changed['units'][0]
        u['report_batches']['A'][0][0]=u['report']['outer'][0]
        with self.assertRaisesRegex(ValueError,'batch manifest'):self.validate(changed)

    def test_row_metadata_and_duplicate_units_are_rejected(self):
        changed=deepcopy(self.sample)
        row=next(iter(changed['rows']['ravdess'].values()));row['label_index']=0
        row['sha256']='f'*64
        changed['plan_sha256']=plan.digest({k:v for k,v in changed.items() if k!='plan_sha256'})
        with self.assertRaises(ValueError):self.validate(changed)
        changed=deepcopy(self.sample);changed['units'][-1]=deepcopy(changed['units'][0])
        with self.assertRaisesRegex(ValueError,'uniqueness'):self.validate(changed)

    def test_repeat_generation_is_byte_semantically_identical(self):
        with patch.object(plan,'_load_metadata',return_value=(self.metas,self.inputs)):
            second=plan.generate(Path('.'))
        self.assertEqual(self.sample,second)


class CropTests(unittest.TestCase):
    def test_integer_uniform_and_short_audio(self):
        for epoch in (1,7,15):
            for slot in (0,1,575):
                encoded=json.dumps([plan.PROGRAM,123,epoch,slot],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
                h=int(hashlib.sha256(encoded).hexdigest()[:16],16)
                for length in (0,12000,47999,48000,48001,93071,160000):
                    got=plan.crop_start(123,epoch,slot,length)
                    self.assertEqual(got,h*(max(0,length-48000)+1)//2**64)
                    self.assertTrue(0<=got<=max(0,length-48000))
        self.assertEqual({plan.crop_start(123,1,s,48001) for s in range(100)},{0,1})

    def test_invalid_crop_coordinates(self):
        for coords in ((True,1,0,50000),(1,0,0,50000),(1,16,0,50000),(1,1,-1,50000),(1,1,0,-1)):
            with self.assertRaises(ValueError):plan.crop_start(*coords)


if __name__=='__main__':unittest.main()
