"""Portable, CPU-only materialization of the prospectively narrowed supplement.

No training is admitted by generating this plan. AutoDL technical gates and an
exact source/environment lock are separately required by run.py.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path

from .engine import canonical, digest, file_sha, require, validate_unit

PROGRAM = 'SER27-AUTODL-SUPPLEMENT-EXECUTION-1'
REFERENCE_BYTE_SHA = '6bee99d32fe5932a6dfe6d22708ac2db6ec445b03afc339be50c649e3cf37c14'
REFERENCE_SEMANTIC_SHA = '393109434af0bfb6d18205f8f3713aa5e08d08f0f0ac3ffc4a63f8e635422c25'
CORPORA = ('cremad', 'subesco', 'ravdess')
MODELS = ('wavlm_base_plus', 'hubert_base')
PANEL_KEYS = ('corpus', 'draw', 'fold', 'n_classes', 'group_speakers',
              'test_speakers', 'fit_prompts', 'query_prompts', 'fit', 'report', 'report_batches')
ROW_KEYS = ('corpus', 'relative_path', 'bytes', 'sha256', 'speaker', 'label',
            'label_index', 'sentence', 'take', 'intensity', 'sex_stratum')


def write_new(path, value, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b'\n'
    if compressed:
        data = gzip.compress(data, compresslevel=9, mtime=0)
    with path.open('xb') as handle:
        handle.write(data)


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)


def seal(value, key):
    require(key not in value, 'already sealed')
    value[key] = digest(value)
    return value


def check_seal(value, key):
    require(key in value and digest({k: v for k, v in value.items() if k != key}) == value[key],
            f'{key} differs')


def export_reference(original):
    require(file_sha(original) == REFERENCE_BYTE_SHA, 'original reference bytes differ')
    ref = read_json(original)
    require(ref['plan_sha256'] == REFERENCE_SEMANTIC_SHA, 'original reference semantic identity differs')
    anchors = []
    for unit in ref['units']:
        if unit['phase'] == 'formal' and unit['arm'] == 'A':
            anchor = {k: unit[k] for k in PANEL_KEYS}
            anchor['reference_unit_id'] = unit['unit_id']
            anchors.append(anchor)
    require(len(anchors) == 360, 'anchor inventory differs')
    used = {c: set() for c in CORPORA}
    for a in anchors:
        for paths in [a['fit'], *a['report'].values()]:
            used[a['corpus']].update(paths)
    rows = {c: {p: {k: ref['rows'][c][p][k] for k in ROW_KEYS}
                for p in sorted(used[c])} for c in CORPORA}
    return seal(dict(schema='ser-supplement-portable-reference-1',
                     reference_plan_byte_sha256=REFERENCE_BYTE_SHA,
                     reference_plan_semantic_sha256=REFERENCE_SEMANTIC_SHA,
                     anchors=anchors, rows=rows), 'reference_sha256')


def seed(corpus, draw, fold, stream, phase='formal'):
    return int(digest([PROGRAM, phase, corpus, draw, fold, stream])[:16], 16) % (2**31 - 1)


def family():
    tests = [dict(id=f'hubert_{c}_{e}', corpus=c, endpoint=e, model='hubert_base', window=15,
                  comparison='absolute_seen_minus_unseen')
             for c in CORPORA for e in ('D_CE', 'J')]
    tests += [dict(id=f'window_{c}_{e}', corpus=c, endpoint=e, model='wavlm_base_plus',
                   comparison='paired_window45_minus_window15')
              for c in ('subesco', 'ravdess') for e in ('L_CE', 'L_J')]
    return tests


def make_unit(anchor, model, epochs, phase='formal'):
    u = copy.deepcopy({k: anchor[k] for k in PANEL_KEYS})
    c, d, f = u['corpus'], u['draw'], u['fold']
    u.update(unit_id=f'ads2_{phase}_{c}_d{d:02d}_f{f}_{model}_A', phase=phase,
             backend_required='autodl', model=model, arm='A', seen_group='A', unseen_group='B',
             reference_unit_id=anchor['reference_unit_id'], reference_panel_sha256=digest(anchor),
             seeds={s: seed(c, d, f, s, phase) for s in ('head', 'order', 'crop', 'torch_training')},
             windows=[15] if epochs == 15 else [15, 45],
             config=dict(model=model, trainable_layers='top4+head', epochs=epochs,
                         batch_size=16, fp16=True, lr_encoder=5e-5, lr_head=1e-3,
                         weight_decay=0.01, crop_seconds=3, eval_cap_seconds=10,
                         early_stopping=False, hyperparameter_search=False),
             permanent_checkpoint_sample=(d == 0 and f == CORPORA.index(c))
                 or (d == 12 and f == (CORPORA.index(c) + 3) % 5))
    return seal(u, 'unit_sha256')


def generate(reference):
    check_seal(reference, 'reference_sha256')
    require(reference['reference_plan_byte_sha256'] == REFERENCE_BYTE_SHA, 'reference provenance differs')
    anchors = {(a['corpus'], a['draw'], a['fold']): a for a in reference['anchors']}
    require(len(anchors) == 360, 'reference anchor count differs')
    units = []
    for draw in range(24):
        for fold in range(5):
            for corpus in CORPORA:
                models = sorted(MODELS, key=lambda m: digest([PROGRAM, 'model_order', corpus, draw, fold, m]))
                for model in models:
                    epochs = 45 if model == 'wavlm_base_plus' and corpus != 'cremad' else 15
                    units.append(make_unit(anchors[(corpus, draw, fold)], model, epochs))
    pilots = []
    # All corpora/backbones are probed; WavLM long probes additionally get an
    # independently trained short companion with exactly the same stochastic seeds.
    for corpus in CORPORA:
        anchor = anchors[(corpus, 23, 4)]
        for model in MODELS:
            epochs = 45 if model == 'wavlm_base_plus' and corpus != 'cremad' else 15
            u = make_unit(anchor, model, epochs, phase='technical_pilot')
            u.pop('unit_sha256')
            u['permanent_checkpoint_sample'] = True
            pilots.append(seal(u, 'unit_sha256'))
            if epochs == 45:
                short = copy.deepcopy(u)
                short.pop('unit_sha256')
                short['unit_id'] += '_independent15'
                short['windows'] = [15]
                short['config']['epochs'] = 15
                pilots.append(seal(short, 'unit_sha256'))
    plan = dict(schema='ser-autodl-supplement-plan-2', program=PROGRAM,
                status='materialized_not_gpu_qualified', execution_ready=False,
                reference_sha256=reference['reference_sha256'],
                rows=reference['rows'], units=units, pilots=pilots,
                formal_units=720, formal_epoch_passes=18000,
                stats=dict(family=family(), replicate='24 draw means after equal five-fold means',
                           test='two-sided one-sample t', df=23, multiplicity='Holm', alpha=0.05,
                           family_size=10, intervals='pointwise95%', missing='reject incomplete main analysis',
                           historical_local_results_enter_primary_pairs=False),
                retention=dict(sample_ids=[u['unit_id'] for u in units if u['permanent_checkpoint_sample']],
                               all_logits_permanent=True, all_saved_states_reload_before_completion=True,
                               non_sample_deletion_requires_verified_off_instance_backup_ack=True,
                               max_units_waiting_backup=8, min_free_gib=20),
                execution=dict(backend='autodl', gpu_count=1, preferred_gpu='5090-p',
                               python='3.11', torch='2.11.0+cu128', torchaudio='2.11.0+cu128',
                               num_threads=4, num_interop_threads=2,
                               no_local_gpu=True, no_training_resume=True,
                               formal_admission_requires_pilots=True))
    seal(plan, 'plan_sha256')
    validate_plan(plan)
    return plan


def validate_plan(plan):
    check_seal(plan, 'plan_sha256')
    require(plan['program'] == PROGRAM and plan['formal_units'] == len(plan['units']) == 720,
            'wrong program/inventory')
    require(len({u['unit_id'] for u in plan['units'] + plan['pilots']}) == 728, 'duplicate/missing IDs')
    counts = Counter((u['corpus'], u['model']) for u in plan['units'])
    require(counts == Counter({(c, m): 120 for c in CORPORA for m in MODELS}), 'factor coverage differs')
    require(Counter(u['config']['epochs'] for u in plan['units']) == {15: 480, 45: 240}, 'epoch matrix differs')
    require(sum(u['config']['epochs'] for u in plan['units']) == 18000, 'workload differs')
    for u in plan['units'] + plan['pilots']:
        check_seal(u, 'unit_sha256')
        validate_unit(u, plan['rows'][u['corpus']])
        require(u['backend_required'] == 'autodl', 'backend differs')
        require(len(u['fit']) == {'cremad': 576, 'subesco': 224, 'ravdess': 64}[u['corpus']], 'fit budget differs')
    indexed = {(u['corpus'], u['draw'], u['fold'], u['model']): u for u in plan['units']}
    for u in plan['units']:
        v = indexed[(u['corpus'], u['draw'], u['fold'], 'wavlm_base_plus')]
        require(all(u[k] == v[k] for k in (*PANEL_KEYS, 'seeds', 'reference_panel_sha256')),
                'matched model context differs')
    require(len(plan['retention']['sample_ids']) == 12, 'fixed restoration sample differs')
    require(plan['stats']['family'] == family(), 'fixed inferential family differs')


def verify_audio(plan, roots):
    validate_plan(plan)
    counts, byte_count = {}, 0
    for corpus, rows in plan['rows'].items():
        root = Path(roots[corpus]).resolve()
        for relative, row in rows.items():
            path = (root / relative).resolve()
            require(path.is_relative_to(root), 'audio path escapes root')
            require(path.stat().st_size == row['bytes'] and file_sha(path) == row['sha256'], 'audio bytes differ')
            byte_count += row['bytes']
        counts[corpus] = len(rows)
    return dict(audio_counts=counts, audio_bytes=byte_count, plan_sha256=plan['plan_sha256'])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export-reference')
    e.add_argument('--original', required=True)
    e.add_argument('--out', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--reference', required=True)
    g.add_argument('--out', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--plan', required=True)
    v.add_argument('--audio-roots')
    args = p.parse_args()
    if args.command == 'export-reference':
        value = export_reference(args.original)
        write_new(args.out, value, compressed=True)
        print(json.dumps(dict(reference_sha256=value['reference_sha256'], anchors=len(value['anchors']),
                              rows={c: len(r) for c, r in value['rows'].items()})))
    elif args.command == 'generate':
        value = generate(read_json(args.reference))
        write_new(args.out, value, compressed=True)
        print(json.dumps({k: value[k] for k in ('plan_sha256', 'formal_units', 'formal_epoch_passes', 'execution_ready')}))
    else:
        value = read_json(args.plan)
        validate_plan(value)
        print(json.dumps(verify_audio(value, read_json(args.audio_roots)) if args.audio_roots else
                         dict(valid=True, plan_sha256=value['plan_sha256'])))


if __name__ == '__main__':
    main()
