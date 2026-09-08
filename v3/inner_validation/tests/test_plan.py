"""Synthetic-only role, quota, randomization and source-freeze contracts.

No real speech, model training, cloud access, or study predictions are used.
The synthetic SHA strings identify fixture rows, not verified audio files.
"""
import copy
import json
from collections import Counter

import pytest

from v3.inner_validation import plan as planner


ANALYSIS = dict(
    primary='draw_mean[(V_seen-T_seen)-(V_unseen-T_unseen)]', draws=24,
    selection='minimum loss checkpoint; maximum own-validation UAR config; first index ties',
    fixed_config_index=3, bootstrap_repetitions=50000, bootstrap_seed=2026090701,
)
PANEL_KEYS = (*planner.ROLES, 'fit_speakers', 'unseen_speakers', 'test_speakers',
              'fit_prompts', 'query_prompts', 'eligible_speakers')


@pytest.fixture(scope='module')
def synthetic_meta():
    speakers = [f'SYNTHETIC_{i:03d}' for i in range(70)]
    prompts = [f'P{i:02d}' for i in range(12)]
    sex = {s: 'Female' if i < 35 else 'Male' for i, s in enumerate(speakers)}
    cells, clean = {}, []
    for speaker in speakers:
        for prompt in prompts:
            for label in range(6):
                path = f'{speaker}_{prompt}_{label}.wav'
                row = dict(relative_path=path, speaker=speaker, sentence=prompt,
                           label_index=label, sha256=planner.digest(['SYNTHETIC_ONLY', path]))
                cells[speaker, prompt, label] = row
                clean.append(row)
    return dict(speakers=speakers, prompts=prompts, sex=sex, cells=cells,
                clean=clean, input={'fixture': 'synthetic_only_not_real_audio'})


@pytest.fixture(scope='module')
def valid_plan(synthetic_meta):
    units = planner.generate(synthetic_meta)
    return dict(schema='ser-dual-validation-plan-1', program=planner.PROGRAM,
                draft=False, source_commit='0' * 40, sources={},
                input=synthetic_meta['input'], base_file_sha256=planner.BASE_SHA,
                units=units, analysis=copy.deepcopy(ANALYSIS),
                pilot_units=[u['unit_id'] for u in units if u['draw'] == u['fold'] == 0])


def context(plan, draw=0, fold=0):
    return [u for u in plan['units'] if (u['draw'], u['fold']) == (draw, fold)]


def cell_paths(meta, people, prompts):
    return sorted(meta['cells'][s, p, c]['relative_path']
                  for s in people for p in prompts for c in range(6))


def test_complete_grid_common_fit_quotas_and_outer_people(valid_plan, synthetic_meta):
    planner.validate(valid_plan, synthetic_meta)
    assert len(valid_plan['units']) == 480
    for draw in range(24):
        out = []
        for fold in range(5):
            block = context(valid_plan, draw, fold)
            first = block[0]
            assert len(block) == 4
            assert len(first['fit']) == 576
            assert len(first['val_seen']) == len(first['val_unseen']) == 288
            for key in PANEL_KEYS:
                assert all(unit[key] == first[key] for unit in block)
            assert len({unit['train_seed'] for unit in block}) == 1
            for key in ('fit_speakers', 'unseen_speakers'):
                assert Counter(synthetic_meta['sex'][s] for s in first[key]) == {'Female': 12, 'Male': 12}
            out.extend(first['test_speakers'])
        assert Counter(out) == Counter(synthetic_meta['speakers'])


def test_deterministic_new_namespace_and_unique_draws(valid_plan, synthetic_meta):
    assert planner.generate(synthetic_meta) == valid_plan['units']
    assert planner.PROGRAM not in ('SER26-N14R2', 'SER26-COVERAGE-1')
    assert len({u['train_seed'] for u in valid_plan['units']}) == 120
    signatures = {tuple(tuple(context(valid_plan, d, f)[0]['test_speakers'])
                        for f in range(5)) for d in range(24)}
    assert len(signatures) == 24


@pytest.mark.parametrize('field,value', [('train_seed', 0), ('unit_id', 'not-a-frozen-unit')])
def test_wrong_unit_identity_or_seed_rejected(valid_plan, synthetic_meta, field, value):
    plan = copy.deepcopy(valid_plan)
    plan['units'][0][field] = value
    with pytest.raises(ValueError):
        planner.validate(plan, synthetic_meta)


def test_seen_unseen_identity_swap_without_fit_swap_rejected(valid_plan, synthetic_meta):
    plan = copy.deepcopy(valid_plan)
    for unit in context(plan):
        unit['val_seen'], unit['val_unseen'] = unit['val_unseen'], unit['val_seen']
    with pytest.raises(ValueError):
        planner.validate(plan, synthetic_meta)


@pytest.mark.parametrize('role', ['fit', 'val_seen', 'val_unseen'])
def test_fitting_or_validation_budget_change_rejected(valid_plan, synthetic_meta, role):
    plan = copy.deepcopy(valid_plan)
    for unit in context(plan):
        unit[role] = unit[role][1:]
    with pytest.raises(ValueError):
        planner.validate(plan, synthetic_meta)


def test_missing_role_class_rejected(valid_plan, synthetic_meta):
    meta = copy.deepcopy(synthetic_meta)
    for row in meta['clean']:
        if row['label_index'] == 5:
            row['label_index'] = 4
    with pytest.raises(ValueError):
        planner.validate(valid_plan, meta)


def test_config_must_not_change_panel_even_to_another_legal_panel(valid_plan, synthetic_meta):
    plan = copy.deepcopy(valid_plan)
    replacement = context(plan, 0, 1)[0]
    unit = context(plan)[1]
    for key in PANEL_KEYS:
        unit[key] = copy.deepcopy(replacement[key])
    with pytest.raises(ValueError):
        planner.validate(plan, synthetic_meta)


def test_missing_available_test_item_rejected_even_when_all_person_classes_remain(valid_plan, synthetic_meta):
    plan = copy.deepcopy(valid_plan)
    removed = context(plan)[0]['test'][0]
    for unit in context(plan):
        unit['test'] = [p for p in unit['test'] if p != removed]
    # This intentionally preserves the other query text for the same class and
    # person. Merely checking positive class support cannot detect the omission.
    with pytest.raises(ValueError):
        planner.validate(plan, synthetic_meta)


def test_legal_but_unprescribed_role_allocation_rejected(valid_plan, synthetic_meta):
    plan = copy.deepcopy(valid_plan)
    for unit in context(plan):
        old_fit, old_unseen = unit['fit_speakers'], unit['unseen_speakers']
        unit['fit_speakers'], unit['unseen_speakers'] = old_unseen, old_fit
        unit['fit'] = cell_paths(synthetic_meta, old_unseen, unit['fit_prompts'])
        unit['val_seen'] = cell_paths(synthetic_meta, old_unseen, unit['query_prompts'])
        unit['val_unseen'] = cell_paths(synthetic_meta, old_fit, unit['query_prompts'])
    with pytest.raises(ValueError):
        planner.validate(plan, synthetic_meta)


def test_eligible_population_cannot_include_outer_test_person(valid_plan, synthetic_meta):
    plan = copy.deepcopy(valid_plan)
    for unit in context(plan):
        unit['eligible_speakers'] = sorted(unit['eligible_speakers'] + [unit['test_speakers'][0]])
    with pytest.raises(ValueError):
        planner.validate(plan, synthetic_meta)


def test_insufficient_metadata_capacity_stops_generation(synthetic_meta):
    meta = copy.deepcopy(synthetic_meta)
    # The fixture now has fewer than the needed 24 complete female candidates
    # even before excluding test speakers. Never replace these with fake audio.
    keep = {s for s in meta['speakers'] if meta['sex'][s] == 'Male'}
    keep.update(s for s in meta['speakers'][:23])
    meta['cells'] = {k: v for k, v in meta['cells'].items() if k[0] in keep}
    with pytest.raises(ValueError):
        planner.generate(meta)


def frozen_fixture(tmp_path, monkeypatch, valid_plan, synthetic_meta):
    source = tmp_path / 'SYNTHETIC_SOURCE.py'
    source.write_text('# only a synthetic source-freeze fixture\n', encoding='utf-8')
    monkeypatch.setattr(planner, '__file__', str(tmp_path / 'v3' / 'inner_validation' / 'plan.py'))
    monkeypatch.setattr(planner, 'source_files', lambda repo: {source.name: planner.file_sha(source)})
    monkeypatch.setattr(planner, 'read_metadata', lambda repo: synthetic_meta)
    plan = copy.deepcopy(valid_plan)
    plan['sources'] = planner.source_files(tmp_path)
    return source, plan


def write_plan(tmp_path, plan):
    plan['plan_sha256'] = planner.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    path = tmp_path / 'SYNTHETIC_PLAN.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    return path


def test_source_bytes_change_rejected_on_frozen_load(tmp_path, monkeypatch, valid_plan, synthetic_meta):
    source, plan = frozen_fixture(tmp_path, monkeypatch, valid_plan, synthetic_meta)
    path = write_plan(tmp_path, plan)
    planner.load_plan(path, tmp_path)
    source.write_text('# changed synthetic source\n', encoding='utf-8')
    with pytest.raises(ValueError):
        planner.load_plan(path, tmp_path)


def test_imported_planner_must_match_selected_checkout(tmp_path, monkeypatch, valid_plan, synthetic_meta):
    _, plan = frozen_fixture(tmp_path, monkeypatch, valid_plan, synthetic_meta)
    path = write_plan(tmp_path, plan)
    monkeypatch.setattr(planner, '__file__', str(tmp_path / 'OTHER_CHECKOUT' / 'v3' / 'inner_validation' / 'plan.py'))
    with pytest.raises(ValueError):
        planner.load_plan(path, tmp_path)


@pytest.mark.parametrize('field,value', [
    ('fixed_config_index', 0), ('bootstrap_seed', 99), ('bootstrap_repetitions', 2),
    ('primary', 'post_hoc_other_endpoint'),
])
def test_analysis_contract_change_rejected_even_with_recomputed_hash(
        tmp_path, monkeypatch, valid_plan, synthetic_meta, field, value):
    _, plan = frozen_fixture(tmp_path, monkeypatch, valid_plan, synthetic_meta)
    plan['analysis'][field] = value
    path = write_plan(tmp_path, plan)
    with pytest.raises(ValueError):
        planner.load_plan(path, tmp_path)


@pytest.mark.parametrize('field,value', [('schema', 'wrong-schema'), ('draft', True)])
def test_wrong_schema_or_draft_cannot_run(tmp_path, monkeypatch, valid_plan, synthetic_meta, field, value):
    _, plan = frozen_fixture(tmp_path, monkeypatch, valid_plan, synthetic_meta)
    plan[field] = value
    path = write_plan(tmp_path, plan)
    with pytest.raises(ValueError):
        planner.load_plan(path, tmp_path)
