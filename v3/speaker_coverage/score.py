"""Score only the complete independently verified 720-unit formal block."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from v3.data_design.core_plan import read_metadata, require, file_sha, digest, write_json

MODELS = ('cnn', 'ridge_wavlm', 'wavlm_ft')
POLICIES = ('U', 'R', 'C')
CHECKPOINTS = ('best', 'last')


def uar(correct, support):
    correct, support = np.asarray(correct), np.asarray(support)
    require(correct.shape == support.shape == (6,) and (support > 0).all()
            and (correct >= 0).all() and (correct <= support).all(), 'incomplete fixed-six-class counts')
    return float((correct / support).mean() * 100)


def repetitions(model):
    return 2 if model == 'wavlm_ft' else 3


def collect(plan, root, meta):
    speakers = sorted(meta['speakers'])
    require(len(speakers) == 91, 'all 91 held-out test people are required')
    by_path = {r['relative_path']:r for r in meta['clean']}
    stats = {}
    closure = {}
    for unit in plan['units']:
        directory = Path(root) / 'formal/units' / unit['unit_id']
        done = json.loads((directory/'DONE').read_text(encoding='utf-8'))
        require(done['unit_id'] == unit['unit_id'] and done['plan_sha256'] == plan['plan_sha256']
                and done['phase'] == 'formal', 'scoring closure identity mismatch')
        closure[unit['unit_id']] = file_sha(directory/'DONE')
        name = f"attempts/{done['attempt']:04d}/predictions.npz"
        path = directory / name
        require(file_sha(path) == done['artifacts'][name], 'predictions changed after verification')
        with np.load(path, allow_pickle=False) as archive:
            paths, labels = archive['paths'].tolist(), archive['labels']
            require(paths == unit['test'] and labels.tolist() == [int(by_path[p]['label_index']) for p in paths],
                    'prediction truth or ordering differs from manifest')
            guesses = {'best':archive['logits'].argmax(1), 'last':archive['last_logits'].argmax(1)}
        repeat = unit['seed_index'] if unit['model'] == 'wavlm_ft' else unit['draw']
        groups = defaultdict(list)
        for position, path in enumerate(paths):
            groups[by_path[path]['speaker']].append(position)
        for speaker, positions in groups.items():
            query_paths = {paths[i] for i in positions}
            for checkpoint in CHECKPOINTS:
                key = unit['model'], unit['policy'], checkpoint, repeat, speaker
                if key not in stats:
                    stats[key] = [np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64), set(), set(), set()]
                correct, support, rotations, folds, seen = stats[key]
                require(unit['rotation'] not in rotations and not (seen & query_paths),
                        'duplicate speaker/query rotation or recording')
                rotations.add(unit['rotation']); folds.add(unit['fold']); seen.update(query_paths)
                truth, guess = labels[positions], guesses[checkpoint][positions]
                support += np.bincount(truth, minlength=6)
                correct += np.bincount(truth[truth == guess], minlength=6)
    expected = {(m,p,c,r,s) for m in MODELS for p in POLICIES for c in CHECKPOINTS
                for r in range(repetitions(m)) for s in speakers}
    require(set(stats) == expected, 'incomplete model/policy/checkpoint/repetition/person grid')
    values, detail, fold_by_person = {}, [], {}
    for model in MODELS:
        for policy in POLICIES:
            for checkpoint in CHECKPOINTS:
                matrix = np.empty((repetitions(model), len(speakers)), dtype=np.float64)
                for repeat in range(repetitions(model)):
                    for index, speaker in enumerate(speakers):
                        correct, support, rotations, folds, paths = stats[model,policy,checkpoint,repeat,speaker]
                        require(rotations == set(range(6)) and len(folds) == 1, 'six-rotation aggregation incomplete')
                        fold = next(iter(folds))
                        require(speaker not in fold_by_person or fold_by_person[speaker] == fold, 'speaker fold changed')
                        fold_by_person[speaker] = fold
                        matrix[repeat,index] = uar(correct,support)
                        detail.append({'model':model,'policy':policy,'checkpoint':checkpoint,
                            'repeat':repeat,'repeat_type':'training_seed' if model=='wavlm_ft' else 'panel_draw',
                            'speaker':speaker,'fold':fold,'uar_percent':float(matrix[repeat,index]),
                            'query_records':len(paths),'minimum_class_support':int(support.min())})
                values[model,policy,checkpoint] = matrix
    return speakers, values, detail, fold_by_person, closure


def paired_estimate(first, second, indices):
    require(first.shape == second.shape and first.ndim == 2 and first.shape[1] == indices.shape[1],
            'paired speaker/repetition dimensions differ')
    difference = (first-second).mean(axis=0)
    low, high = np.percentile(difference[indices].mean(axis=1), [2.5,97.5], method='linear')
    return {'difference_pp':float(difference.mean()),'ci95_low_pp':float(low),'ci95_high_pp':float(high),
            'repeat_differences_pp':(first-second).mean(axis=1).tolist()}


def summarize(speakers, values, folds):
    indices = np.random.default_rng(20260906).integers(0,len(speakers),size=(10000,len(speakers)))
    policies, contrasts, by_fold, people = [], [], [], []
    for model in MODELS:
        for checkpoint in CHECKPOINTS:
            for policy in POLICIES:
                matrix = values[model,policy,checkpoint]
                means = matrix.mean(axis=0)
                policies.append({'model':model,'policy':policy,'checkpoint':checkpoint,
                    'n_speakers':len(speakers),'n_repetitions':len(matrix),
                    'mean_speaker_uar_percent':float(means.mean()),
                    'q25_speaker_uar_percent_descriptive':float(np.quantile(means,.25,method='linear')),
                    'repeat_population_means_percent':matrix.mean(axis=1).tolist()})
                for i,speaker in enumerate(speakers):
                    people.append({'model':model,'policy':policy,'checkpoint':checkpoint,'speaker':speaker,
                                   'fold':folds[speaker],'uar_percent':float(means[i])})
            for policy in ('C','R'):
                contrast = policy+'-U'
                estimate = paired_estimate(values[model,policy,checkpoint], values[model,'U',checkpoint],indices)
                row = {'model':model,'checkpoint':checkpoint,'contrast':contrast,
                    'role':'primary' if (model,checkpoint,policy)==('cnn','best','C') else 'secondary_or_sensitivity',
                    'n_speakers':len(speakers),**estimate}
                contrasts.append(row)
                difference=(values[model,policy,checkpoint]-values[model,'U',checkpoint]).mean(axis=0)
                for fold in range(5):
                    selected=[i for i,speaker in enumerate(speakers) if folds[speaker]==fold]
                    by_fold.append({'model':model,'checkpoint':checkpoint,'contrast':contrast,'fold':fold,
                                    'n_speakers':len(selected),'difference_pp_descriptive':float(difference[selected].mean())})
    return policies,contrasts,by_fold,people


def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--outroot',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    require(args.repo.resolve() == Path(__file__).resolve().parents[2], 'executed checkout differs from --repo')
    from v3.speaker_coverage.verify import verify_result
    audits={block:verify_result(args.repo,args.plan,args.outroot,phase='formal',block=block,consolidate=True)
            for block in ('core','ft')}
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    require(len(plan['units']) == 720 and len({u['unit_id'] for u in plan['units']}) == 720,'incomplete plan')
    meta=read_metadata(args.repo)
    require(all(meta['input'][k] == plan['input'][k] for k in ('manifest_path','manifest_sha256','demographics_path','demographics_sha256')), 'metadata identity mismatch')
    speakers,values,detail,folds,closure=collect(plan,args.outroot,meta)
    policies,contrasts,by_fold,people=summarize(speakers,values,folds)
    # Check every committed marker again before publishing computed tables.
    require(all(file_sha(args.outroot/'formal/units'/uid/'DONE')==sha for uid,sha in closure.items()),
            'closure changed during scoring')
    args.out.mkdir(parents=True,exist_ok=True)
    tables={'policy_means.csv':policies,'contrasts.csv':contrasts,'fold_contrasts.csv':by_fold,
            'per_speaker.csv':people,'per_repeat_speaker.csv':detail}
    for filename,rows in tables.items(): write_csv(args.out/filename,rows)
    report={'schema':'ser-speaker-coverage-analysis-1','plan_sha256':plan['plan_sha256'],
        'source_sha256':file_sha(Path(__file__)),'verified_formal_units':720,'excluded_pilot_units':19,
        'result_audits':audits,'done_mapping_sha256':digest(closure),'primary':next(r for r in contrasts if r['role']=='primary'),
        'policies':policies,'contrasts':contrasts,'bootstrap_repetitions':10000,'bootstrap_seed':20260906,
        'interval_scope':'paired people, conditional on these fitted models/reference corpus; shared training sets and few training seeds are not independent replicates',
        'aggregation':'pool six rotations by person/class, compute six-class UAR, average three draws or two FT training seeds per person, then equal-person mean',
        'all_comparisons_reported':True,'equivalence_claim':False,'causal_mediation_claim':False,
        'table_sha256':{name:file_sha(args.out/name) for name in tables}}
    write_json(args.out/'results.json',report)
    print(json.dumps({'status':'complete_720_verified','primary':report['primary']}))


if __name__=='__main__': main()
