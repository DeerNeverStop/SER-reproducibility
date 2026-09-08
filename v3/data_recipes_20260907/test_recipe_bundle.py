"""Independent acceptance against the real, pinned Study II metadata.

No toy plan, planner build, features, NPZ arrays, checkpoints or new-384 data.
All mutations occur in fresh test copies. Use a fresh D: pytest --basetemp for
the actual acceptance run; the caller limits CPU threads and process priority.
"""
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil

import pytest

from v3.data_recipes_20260907 import recipe_bundle as recipe


EXPECTED_PLAN_SHA = 'f258cabe582d97b3a386a666b566730d00241a5ded0e14ad61d590cb3c861503'
EXPECTED_PLAN_BYTES_SHA = '27d42104e9df3b031c11293caec096f714eb4f13caedde8c9419fc517156ada9'
ROLES = ('fit', 'val', 'test')
REPO = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def byte_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n',
                    encoding='utf-8', newline='\n')


def reseal_weak_manifest(bundle):
    """Change only the export manifest; immutable original pins stay untouched."""
    manifest = read(bundle/'manifest.json')
    manifest['files'] = {p.relative_to(bundle).as_posix(): dict(bytes=p.stat().st_size, sha256=byte_sha(p))
                         for p in bundle.rglob('*') if p.is_file() and p != bundle/'manifest.json'}
    write(bundle/'manifest.json', manifest)


@pytest.fixture(scope='module')
def original_plan():
    raw = gzip.decompress((REPO/'v3/data_design/evidence/core_plan.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_PLAN_BYTES_SHA
    plan = json.loads(raw)
    assert plan['plan_sha256'] == EXPECTED_PLAN_SHA
    assert digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) == EXPECTED_PLAN_SHA
    return plan


@pytest.fixture(scope='module')
def complete_bundle(tmp_path_factory, original_plan):
    out = tmp_path_factory.mktemp('real_study2_acceptance')/'bundle'
    receipt = recipe.build(REPO, out)
    assert receipt['pass'] and receipt['plan_sha256'] == EXPECTED_PLAN_SHA
    return out


@pytest.fixture
def damaged_bundle(complete_bundle, tmp_path):
    out = tmp_path/'damaged_bundle'
    shutil.copytree(complete_bundle, out)
    return out


def test_real_build_verify_and_all_1440_ordered_units_losslessly_reconstructed(complete_bundle, original_plan):
    bundle = complete_bundle
    receipt = recipe.verify(bundle)
    assert receipt['pass'] and receipt['counts'] == dict(units=1440, representative_rows=6524,
        fit_components=720, val_components=90, test_components=90)
    assert receipt['scope']['new_plan_generated'] is False
    assert receipt['scope']['contains_audio_features_weights_or_predictions'] is False
    index = read(bundle/'units.json')
    assert len(index['units']) == 1440
    assert [u['unit_id'] for u in index['units']] == [u['unit_id'] for u in original_plan['units']]
    components = {p.relative_to(bundle).as_posix(): read(p) for p in (bundle/'components').rglob('*.json')}
    assert len(components) == 900
    for relative, paths in components.items():
        assert PurePosixPath(relative).stem == digest(paths)
    unique = {role: set() for role in ROLES}
    fit_counts = Counter()
    for original, item in zip(original_plan['units'], index['units']):
        rebuilt = {k: v for k, v in item.items() if k != 'components'}
        assert set(item['components']) == set(ROLES)
        for role in ROLES:
            rebuilt[role] = components[item['components'][role]]
            unique[role].add(tuple(rebuilt[role]))
        assert rebuilt == original
        assert digest({k: v for k, v in rebuilt.items() if k != 'unit_id'}) == original['unit_id']
        fit_counts[len(rebuilt['fit'])] += 1
    assert {role: len(paths) for role, paths in unique.items()} == dict(fit=720, val=90, test=90)
    assert fit_counts == {288: 720, 576: 720}
    # All original metadata/source files are literal copies, not regenerated.
    manifest = read(bundle/'manifest.json')
    for relative in manifest['original_files']:
        assert (bundle/'originals'/relative).read_bytes() == (REPO/relative).read_bytes()
    assert len(original_plan['source_sha256']) == 16
    assert set(original_plan['source_sha256']) <= set(manifest['original_files'])
    assert not any(p.suffix.lower() in ('.npz', '.pt', '.pth', '.wav', '.flac')
                   for p in bundle.rglob('*') if p.is_file())


def test_real_export_csv_preserves_original_order_and_verified_metadata(complete_bundle, original_plan, tmp_path):
    unit = next(u for u in reversed(original_plan['units']) if u['B'] == 576 and u['S'] == 48)
    out = tmp_path/'one_actual_unit'
    receipt = recipe.export_unit(complete_bundle, unit['unit_id'], out)
    assert receipt['pass'] and receipt['unit_id'] == unit['unit_id']
    assert read(out/'unit.json') == unit
    reps = list(csv.DictReader(io.StringIO(gzip.decompress(
        (REPO/'v3/data_design/evidence/representative_manifest.csv.gz').read_bytes()).decode('utf-8-sig'))))
    rep = {row['relative_path']: row for row in reps}
    with (REPO/'v3/data_design/metadata/VideoDemographics.csv').open(encoding='utf-8-sig', newline='') as stream:
        sex = {row['ActorID']: row['Sex'] for row in csv.DictReader(stream)}
    for role in ROLES:
        with (out/(role+'.csv')).open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream); rows = list(reader)
        assert reader.fieldnames == ['row_index', *reps[0], 'sex']
        assert [int(row['row_index']) for row in rows] == list(range(len(unit[role])))
        assert [row['relative_path'] for row in rows] == unit[role]
        assert receipt['rows'][role] == len(rows)
        for row in rows:
            assert {k: row[k] for k in reps[0]} == rep[row['relative_path']]
            assert row['sex'] == sex[row['speaker']]
            path = PurePosixPath(row['relative_path'])
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in str(path) and ':' not in str(path)
    before = {p.name: byte_sha(p) for p in out.iterdir()}
    with pytest.raises(ValueError, match='empty directory'):
        recipe.export_unit(complete_bundle, unit['unit_id'], out)
    assert {p.name: byte_sha(p) for p in out.iterdir()} == before


@pytest.mark.parametrize('source_kind', ['plan', 'frozen_code'])
def test_build_rejects_corrupt_original_or_frozen_source(complete_bundle, original_plan, tmp_path, source_kind):
    source = tmp_path/'copied_source'; shutil.copytree(complete_bundle/'originals', source)
    relative = 'v3/data_design/evidence/core_plan.json.gz' if source_kind == 'plan' else next(iter(original_plan['source_sha256']))
    target = source/relative
    target.write_bytes(target.read_bytes()+b'\nCORRUPTED_TEST_COPY\n')
    out = tmp_path/'must_not_exist'
    with pytest.raises(ValueError, match='original changed|source changed'):
        recipe.build(source, out)
    assert not out.exists()


def test_reordered_component_cannot_be_authorized_by_rehashing_index_and_manifest(damaged_bundle):
    bundle = damaged_bundle
    index = read(bundle/'units.json')
    old_relative = index['units'][0]['components']['fit']
    values = read(bundle/old_relative)
    assert values[0] != values[1]
    values[0], values[1] = values[1], values[0]
    new_relative = 'components/fit/'+digest(values)+'.json'
    write(bundle/new_relative, values)
    # Delete only this exact fixture member, never any original or directory.
    (bundle/old_relative).unlink()
    for item in index['units']:
        if item['components']['fit'] == old_relative:
            item['components']['fit'] = new_relative
    write(bundle/'units.json', index)
    reseal_weak_manifest(bundle)
    with pytest.raises(ValueError, match='original plan|original|contents'):
        recipe.verify(bundle)


def test_missing_uid_cannot_be_hidden_by_resigning_weak_manifest(damaged_bundle):
    index = read(damaged_bundle/'units.json')
    index['units'].pop()
    write(damaged_bundle/'units.json', index)
    reseal_weak_manifest(damaged_bundle)
    with pytest.raises(ValueError, match='unit index differs'):
        recipe.verify(damaged_bundle)


def test_extra_member_rejected_both_unlisted_and_self_manifested(damaged_bundle):
    (damaged_bundle/'extra.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError, match='unlisted/missing'):
        recipe.verify(damaged_bundle)
    reseal_weak_manifest(damaged_bundle)
    with pytest.raises(ValueError, match='contents/counts'):
        recipe.verify(damaged_bundle)


def test_bad_component_payload_and_unsafe_reference_fail_closed(damaged_bundle):
    index = read(damaged_bundle/'units.json')
    component = damaged_bundle/index['units'][0]['components']['val']
    original = component.read_bytes()
    component.write_bytes(b'not valid JSON\n')
    with pytest.raises(ValueError, match='payload changed'):
        recipe.verify(damaged_bundle)
    component.write_bytes(original)
    index['units'][0]['components']['val'] = '../../outside.wav'
    write(damaged_bundle/'units.json', index)
    reseal_weak_manifest(damaged_bundle)
    with pytest.raises(ValueError, match='unit index differs'):
        recipe.verify(damaged_bundle)


def test_output_input_overlap_and_existing_build_never_overwrite(complete_bundle, original_plan, tmp_path):
    manifest_sha = byte_sha(complete_bundle/'manifest.json')
    with pytest.raises(ValueError, match='empty directory'):
        recipe.build(REPO, complete_bundle)
    assert byte_sha(complete_bundle/'manifest.json') == manifest_sha
    for protected_out in (complete_bundle, complete_bundle/'inside', complete_bundle.parent):
        with pytest.raises(ValueError, match='overlaps input'):
            recipe._new_output(protected_out, (complete_bundle,))
    with pytest.raises(ValueError, match='overlaps input'):
        recipe.export_unit(complete_bundle, original_plan['units'][0]['unit_id'], complete_bundle/'inside')
    assert not (complete_bundle/'inside').exists()
    for bad_uid in ('../outside', '0'*64):
        with pytest.raises(ValueError, match='unit_id'):
            recipe.export_unit(complete_bundle, bad_uid, tmp_path/'bad_uid')
    assert not (tmp_path/'bad_uid').exists()


@pytest.mark.parametrize('name', ['../escape.wav', '/absolute.wav', 'C:/absolute.wav',
                                  'folder\\escape.wav', 'folder/../../escape.wav', 'folder//escape.wav',
                                  'folder/./escape.wav', 'file\x00.wav'])
def test_audio_and_member_path_safety(name):
    with pytest.raises(ValueError, match='unsafe relative member'):
        recipe.safe_member(name)
