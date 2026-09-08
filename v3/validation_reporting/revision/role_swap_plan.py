"""Metadata-only executable design; no audio, prediction, model or GPU access."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from v3.data_design.core_plan import digest, file_sha, read_metadata, require, write_json

PROGRAM = 'SER26-ROLE-SWAP-DESIGN-1'
DRAWS, FOLDS = 24, 5
CONFIG = dict(model='wavlm_base_plus_ft', trainable_layers='top4+head',
              lr_encoder=5e-5, lr_head=1e-3, weight_decay=.01, epochs=15,
              batch_size=16, fp16=True, crop_seconds=3., eval_cap_seconds=10.,
              checkpoint='last', early_stopping=False, hyperparameter_search=False)


def ranked(values, *parts):
    return sorted(values, key=lambda value: (digest([PROGRAM, *parts, value]), value))


def seed(*parts):
    return int(digest([PROGRAM, *parts])[:16], 16) % (2**31 - 1)


def _generate(meta):
    pairs = []
    for draw in range(DRAWS):
        outer = {sex: ranked([s for s in meta['speakers'] if meta['sex'][s] == sex], draw, sex, 'outer')
                 for sex in ('Female', 'Male')}
        for fold in range(FOLDS):
            test = sorted(s for group in outer.values() for s in group[fold::FOLDS])
            prompts = ranked(meta['prompts'], draw, fold, 'texts')
            fit_texts, query_texts = prompts[:4], prompts[4:6]
            eligible = [s for s in meta['speakers'] if s not in test
                        and all((s, p, c) in meta['cells'] for p in fit_texts + query_texts for c in range(6))]
            groups = {'A': [], 'B': []}
            for sex in ('Female', 'Male'):
                people = ranked([s for s in eligible if meta['sex'][s] == sex], draw, fold, sex, 'groups')
                require(len(people) >= 24, 'insufficient complete people; no result-dependent resampling')
                groups['A'].extend(people[:12])
                groups['B'].extend(people[12:24])

            def paths(people, texts, complete=True):
                out = []
                for s in sorted(people):
                    for p in texts:
                        for c in range(6):
                            cell = meta['cells'].get((s, p, c))
                            require(cell is not None or not complete, 'missing required cell')
                            if cell is not None:
                                out.append(cell['relative_path'])
                return sorted(out)

            # Slot k matches metadata sex, within-sex person rank, prompt and class.
            # Independent order RNG and slot-keyed crop draws are execution requirements.
            fit_slots = []
            for sex in ('Female', 'Male'):
                ordered = {a: [s for s in groups[a] if meta['sex'][s] == sex] for a in groups}
                for person_rank in range(12):
                    for prompt in fit_texts:
                        for label in range(6):
                            fit_slots.append({'sex': sex, 'person_rank': person_rank,
                                'prompt': prompt, 'label': label,
                                **{a: meta['cells'][ordered[a][person_rank], prompt, label]['relative_path'] for a in groups}})
            report = {a: paths(groups[a], query_texts) for a in groups}
            report['outer'] = paths(test, query_texts, complete=False)
            shared_paths = sorted(p for group in report.values() for p in group)
            pair = dict(pair_id=f'swap_d{draw:02d}_f{fold}', draw=draw, fold=fold,
                group_speakers={a: sorted(s) for a, s in groups.items()}, test_speakers=test,
                eligible_speakers=sorted(eligible), fit_prompts=fit_texts, query_prompts=query_texts,
                fit_slots=fit_slots, report=report,
                shared_report_batches=[shared_paths[i:i+16] for i in range(0, len(shared_paths), 16)],
                seeds={kind: seed(draw, fold, kind) for kind in ('initialization', 'order', 'crop', 'torch_training')},
                arm_execution_order=['A', 'B'] if seed(draw, fold, 'execution_order') % 2 == 0 else ['B', 'A'])
            pairs.append(pair)
    plan = dict(program=PROGRAM, status='metadata-validated design; not frozen for GPU execution',
        latest_external_review_resolved=False, new_training_runs=0, config=CONFIG.copy(),
        input=meta['input'].copy(), inference=dict(primary='mean_draw(mean_fold(0.5*((Q_A(M_A)-Q_A(M_B))+(Q_B(M_B)-Q_B(M_A)))))',
            test='single two-sided one-sample t on 24 draw means', df=23,
            interval='95% t interval; fixed-corpus randomization scope only',
            bootstrap=False, secondary_hypothesis_tests=False),
        execution_requirements=dict(separate_order_and_crop_rng=True,
            crop_uniform_key='[PROGRAM, crop_seed, epoch_1based, fit_slot_index_0based]',
            crop_uniform='h64 = first 16 hex SHA256 digits as integer; start=(h64*(max(0,n_samples-48000)+1))//2**64',
            crop_draw_required_for_every_slot=True, report_only_after_last_checkpoint=True,
            paired_initial_full_state_hash_equal=True, paired_report_batch_paths_equal=True,
            save_last_delta_only=True, independently_reload_every_saved_delta=True,
            pilot_fits=2, pilot_in_primary=False, formal_fits=240,
            partial_epoch_resume=False, science_score_until_all_pairs_sealed=False), pairs=pairs)
    return plan


def build(meta):
    plan = _generate(meta)
    validate(plan, meta)
    plan['plan_sha256'] = digest(plan)
    return plan


def validate(plan, meta):
    pairs = plan['pairs']
    rows = {r['relative_path']: r for r in meta['clean']}
    require(plan['config'] == CONFIG and plan['input'] == meta['input'], 'fixed config/input changed')
    require(len(pairs) == 120 and {(p['draw'], p['fold']) for p in pairs} ==
            {(d, f) for d in range(DRAWS) for f in range(FOLDS)}, 'incomplete draw/fold design')
    for p in pairs:
        a, b, test = set(p['group_speakers']['A']), set(p['group_speakers']['B']), set(p['test_speakers'])
        require(len(a) == len(b) == 24 and not a & b and not (a | b) & test, 'speaker roles overlap/wrong size')
        for group in (a, b):
            require(Counter(meta['sex'][s] for s in group) == Counter(Female=12, Male=12), 'sex imbalance')
        require(len(p['fit_prompts']) == 4 and len(p['query_prompts']) == 2
                and not set(p['fit_prompts']) & set(p['query_prompts']), 'text overlap/budget')
        require(len(p['fit_slots']) == 576, 'fit budget')
        used_fit = set()
        for arm, people in [('A', a), ('B', b)]:
            fit = [s[arm] for s in p['fit_slots']]
            require(len(set(fit)) == 576, 'duplicate fit slot')
            used_fit.update(fit)
            require({rows[x]['speaker'] for x in fit} == people, 'fit speakers differ')
            require(Counter(int(rows[x]['label_index']) for x in fit) == Counter({c: 96 for c in range(6)}), 'fit class balance')
            require(len(p['report'][arm]) == len(set(p['report'][arm])) == 288, 'report budget')
            require({rows[x]['speaker'] for x in p['report'][arm]} == people, 'report people differ')
            require(Counter(int(rows[x]['label_index']) for x in p['report'][arm]) == Counter({c: 48 for c in range(6)}), 'report classes')
            for slot in p['fit_slots']:
                row = rows[slot[arm]]
                require(meta['sex'][row['speaker']] == slot['sex'] and row['sentence'] == slot['prompt']
                        and int(row['label_index']) == slot['label'], 'paired fit slot metadata differ')
        union = [x for v in p['report'].values() for x in v]
        require(len(union) == len(set(union)) and not set(union) & used_fit, 'report paths leak into fit')
        require({rows[x]['speaker'] for x in p['report']['outer']} == test, 'outer people mismatch')
        require({int(rows[x]['label_index']) for x in p['report']['outer']} == set(range(6)), 'outer missing class')
        for speaker in test:
            require({int(rows[x]['label_index']) for x in p['report']['outer']
                     if rows[x]['speaker'] == speaker} == set(range(6)), 'outer person missing class')
        require(all(rows[x]['sentence'] in p['query_prompts'] for x in union), 'report query texts differ')
        require([x for batch in p['shared_report_batches'] for x in batch] == sorted(union), 'shared report order differs')
        require(all(len(batch) == 16 for batch in p['shared_report_batches'][:-1])
                and 1 <= len(p['shared_report_batches'][-1]) <= 16, 'report batch size differs')
    for draw in range(DRAWS):
        outer = [s for p in pairs if p['draw'] == draw for s in p['test_speakers']]
        require(sorted(outer) == meta['speakers'], 'outer folds do not cover corpus exactly once per draw')
    # Replay the outcome-free deterministic design, including seeds, slot ranks,
    # execution order, endpoints and all fixed fields. This shares the generator;
    # it is mutation detection, not independent scientific validation.
    expected = _generate(meta)
    require(set(expected) <= set(plan) <= set(expected) | {'plan_sha256', 'generator_sha256'}, 'plan fields differ')
    require(all(plan[k] == v for k, v in expected.items()), 'deterministic design replay differs')
    if 'plan_sha256' in plan:
        require(plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}), 'plan semantic hash differs')
    if 'generator_sha256' in plan:
        require(plan['generator_sha256'] == file_sha(Path(__file__)), 'generator identity differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'refuse to overwrite plan')
    plan = build(read_metadata(args.repo.resolve()))
    plan['generator_sha256'] = file_sha(Path(__file__))
    # Regenerate digest after recording executable generator identity.
    plan['plan_sha256'] = digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    write_json(args.out, plan)
    print({'pairs': len(plan['pairs']), 'formal_fits': 240, 'updates': 129600,
           'status': plan['status'], 'plan_sha256': plan['plan_sha256']})


if __name__ == '__main__':
    main()
