"""Synthetic end-to-end export/move/replay only; no real formal results.

Only the source/plan loader is stubbed for the synthetic plan. The real frozen
reservation validator, all ZIP/file/hash checks, and independent numeric audit
run. Dummy checkpoint bytes must never be read or included.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from . import independent_numeric_audit as numeric
from . import ledger_contract
from . import portable_bundle as portable
from .test_independent_numeric_audit import template, write_json, seal_result


@pytest.fixture(scope='module')
def complete_fixture(template, tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_portable')
    source = root/'original'; shutil.copytree(template, source)
    phase, scores = source/'formal', source/'scores'
    plan, lock = numeric.json_read(phase/'plan_snapshot.json'), numeric.json_read(phase/'SOURCE_LOCK.json')
    gate = numeric.json_read(phase/'COMPLETE_GATE.json')
    events = [{'event': 'process_start', 'lock_sha256': lock['lock_sha256']}]
    for unit in numeric.formal_units(plan):
        folder = phase/'units'/unit['unit_id']
        write_json(folder/'reservations/0001.json', dict(unit_id=unit['unit_id'],
            unit_sha256=numeric.semantic_sha(unit), lock_sha256=lock['lock_sha256']))
        events.extend([dict(event='unit_start', unit_id=unit['unit_id'], attempt='attempts/0001'),
            dict(event='unit_done', unit_id=unit['unit_id'], attempt='attempts/0001', done_sha256=numeric.sha(folder/'DONE'))])
    events.append(dict(event='process_complete', completed_here=384))
    (phase/'ledger.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events), encoding='utf-8', newline='\n')
    journal = ledger_contract.audit_events([u for u in plan['units'] if u['phase'] == 'formal'], phase, lock, events)
    gate.update(ledger_sha256=numeric.sha(phase/'ledger.jsonl'), ledger_audit=journal,
                reservation_sha256=journal['reservation_sha256'])
    write_json(phase/'COMPLETE_GATE.json', gate)
    shutil.copyfile(phase/'COMPLETE_GATE.json', scores/'complete_gate.json')
    result = numeric.json_read(scores/'results.json')
    result['inputs']['run_dir'] = str(phase.resolve())
    result['inputs']['byte_sha256']['ledger.jsonl'] = numeric.sha(phase/'ledger.jsonl')
    result['inputs']['complete_gate_sha256'] = result['inputs']['byte_sha256']['COMPLETE_GATE.json'] = numeric.sha(scores/'complete_gate.json')
    result['complete_gate_snapshot']['sha256'] = numeric.sha(scores/'complete_gate.json')
    seal_result(scores/'results.json', result)

    def checked(repo, requested):
        actual_plan = numeric.json_read(requested/'plan_snapshot.json')
        actual_lock = numeric.json_read(requested/'SOURCE_LOCK.json')
        assert actual_plan == plan and actual_lock == lock, 'synthetic source/plan identity changed'
        return actual_plan, actual_lock
    def rows(requested):
        raw = (requested/'ledger.jsonl').read_bytes()
        assert raw.endswith(b'\n')
        return [json.loads(line) for line in raw.splitlines()]
    runner = SimpleNamespace(checked_plan=checked, ledger_rows=rows, audit_events=ledger_contract.audit_events)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(numeric, 'checked_runner', lambda repo: runner)
        archive = root/'original_export.zip'
        exported = portable.export_bundle(Path.cwd(), phase, scores, archive)
        yield SimpleNamespace(root=root, source=source, phase=phase, scores=scores, archive=archive,
                              exported=exported, plan=plan, lock=lock, repo=Path.cwd())


def rewrite_zip(original, target, change):
    with zipfile.ZipFile(original) as archive:
        data = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(data[portable.MANIFEST])
    change(data, manifest)
    manifest['files'] = {name: dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
                         for name, blob in data.items() if name != portable.MANIFEST}
    manifest.pop('manifest_sha256', None)
    manifest['manifest_sha256'] = numeric.semantic_sha(manifest)
    data[portable.MANIFEST] = (json.dumps(manifest, indent=2)+'\n').encode()
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, blob in data.items():
            archive.writestr(name, blob)


def test_export_migrate_and_actual_numerical_replay(complete_fixture, tmp_path):
    case = complete_fixture
    with zipfile.ZipFile(case.archive) as archive:
        members = archive.namelist()
    assert sum(name.endswith('/predictions.npz') for name in members) == 384
    assert sum('/reservations/' in name for name in members) == 384
    assert not any(name.endswith(('.pt', '.pth', '.wav')) for name in members)
    out = tmp_path/'moved'/'accepted_bundle'
    report = tmp_path/'accepted_numeric.json'
    accepted = portable.extract_bundle(case.repo, case.archive, out, report)
    audit = numeric.json_read(report)
    assert accepted['accepted'] and accepted['formal_units'] == 384
    assert audit['pass'] and audit['mismatches'] == 0 and audit['max_abs_error'] == 0
    assert audit['numeric_values_checked'] > 50000
    assert audit['relocation']['applied'] and audit['relocation']['explicitly_allowed']
    assert audit['relocation']['actual_run_dir'] == str((out/'formal').resolve())
    assert audit['relocation']['recorded_run_dir'] == str(case.phase.resolve())
    assert (out/'scores/results.json').read_bytes() == (case.scores/'results.json').read_bytes()
    assert all(not item['exists_at_audit'] and not item['rehash_performed']
               for item in audit['checkpoint_gate_pins_not_rehashed'].values())
    manifest, _, _, ledger = portable.inspect_bundle(case.repo, out)
    assert manifest['checkpoint_bytes_included'] is False and ledger['attempt_counts']['done'] == 384


def test_missing_control_B_rejected_even_with_resigned_manifest(complete_fixture, tmp_path):
    case = complete_fixture
    control = next(unit for unit in case.plan['units'] if unit['phase'] == 'formal' and unit['arm'] == 'B')
    prefix = 'formal/units/'+control['unit_id']+'/'
    def remove(data, unused):
        for name in list(data):
            if name.startswith(prefix): del data[name]
    bad = tmp_path/'missing_B.zip'; rewrite_zip(case.archive, bad, remove)
    with pytest.raises(ValueError, match='unit directory set'):
        portable.extract_bundle(case.repo, bad, tmp_path/'bad_B', tmp_path/'bad_B.json')
    assert not (tmp_path/'bad_B').exists() and not (tmp_path/'bad_B.json').exists()


def test_changed_logits_cannot_be_authorized_by_weak_manifest(complete_fixture, tmp_path):
    case = complete_fixture
    def corrupt(data, unused):
        name = next(name for name in data if name.endswith('/predictions.npz'))
        with np.load(io.BytesIO(data[name]), allow_pickle=False) as old:
            arrays = {key: old[key].copy() for key in old.files}
        arrays['outer__all_epoch_logits'][0, 0, 0] += 1
        buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
        data[name] = buffer.getvalue()
    bad = tmp_path/'changed_logits.zip'; rewrite_zip(case.archive, bad, corrupt)
    with pytest.raises(ValueError, match='committed small artifact differs'):
        portable.extract_bundle(case.repo, bad, tmp_path/'bad_prediction', tmp_path/'bad_prediction.json')
    assert not (tmp_path/'bad_prediction.json').exists()


@pytest.mark.parametrize('extra', ['../escaped.txt', 'C:/escaped.txt', 'formal/../escaped.txt',
                                  'formal/units/x/attempts/0001/checkpoint.pt', 'weights/base.npz'])
def test_unsafe_or_weight_members_refused_before_extraction(complete_fixture, tmp_path, extra):
    def add(data, unused): data[extra] = b'FORBIDDEN'
    bad = tmp_path/'bad.zip'; rewrite_zip(complete_fixture.archive, bad, add)
    with pytest.raises(ValueError, match='unsafe|forbidden'):
        portable.extract_bundle(complete_fixture.repo, bad, tmp_path/'rejected', tmp_path/'reject.json')
    assert not (tmp_path/'rejected').exists() and not (tmp_path/'escaped.txt').exists()


def test_existing_outputs_are_never_overwritten(complete_fixture, tmp_path):
    case = complete_fixture
    before = case.archive.read_bytes()
    with pytest.raises(ValueError, match='new'):
        portable.export_bundle(case.repo, case.phase, case.scores, case.archive)
    assert case.archive.read_bytes() == before
    target = tmp_path/'exists'; target.mkdir(); (target/'keep.txt').write_bytes(b'KEEP')
    with pytest.raises(ValueError, match='new'):
        portable.extract_bundle(case.repo, case.archive, target, tmp_path/'unused.json')
    assert (target/'keep.txt').read_bytes() == b'KEEP'
    report = tmp_path/'exists.json'; report.write_bytes(b'KEEP_JSON')
    with pytest.raises(ValueError, match='new'):
        portable.extract_bundle(case.repo, case.archive, tmp_path/'unused_dir', report)
    assert report.read_bytes() == b'KEEP_JSON' and not (tmp_path/'unused_dir').exists()


def test_relocation_opt_in_waives_only_absolute_run_directory(complete_fixture, tmp_path):
    case = complete_fixture
    bundle = tmp_path/'unpacked'; bundle.mkdir(); portable.safe_extract(case.archive, bundle)
    args = SimpleNamespace(repo=case.repo, run_dir=bundle/'formal', results=bundle/'scores', out_json=tmp_path/'no.json')
    with pytest.raises(ValueError, match='allow-relocated-inputs'):
        numeric.execute(args)
    assert not args.out_json.exists()
    doc = numeric.json_read(bundle/'scores/results.json')
    doc['inputs']['byte_sha256']['ledger.jsonl'] = 'f'*64
    seal_result(bundle/'scores/results.json', doc)
    args.allow_relocated_inputs = True
    with pytest.raises(ValueError, match='input closure differs'):
        numeric.execute(args)
    assert not args.out_json.exists()


def test_partial_formal_export_refused_before_prediction_read(complete_fixture, tmp_path, monkeypatch):
    case = complete_fixture
    copy = tmp_path/'partial'; shutil.copytree(case.source, copy)
    gate = numeric.json_read(copy/'formal/COMPLETE_GATE.json'); gate['verified_units'] = 383
    write_json(copy/'formal/COMPLETE_GATE.json', gate)
    def forbidden(*args, **kwargs): raise AssertionError('must not read predictions before full gate')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='complete formal 384'):
        portable.export_bundle(case.repo, copy/'formal', copy/'scores', tmp_path/'forbidden.zip')
    assert not (tmp_path/'forbidden.zip').exists()


def test_reservation_tampering_rejected_by_frozen_ledger_contract(complete_fixture, tmp_path):
    def corrupt(data, unused):
        name = next(name for name in data if '/reservations/' in name)
        item = json.loads(data[name]); item['lock_sha256'] = '0'*64
        data[name] = json.dumps(item).encode()
    bad = tmp_path/'reservation.zip'; rewrite_zip(complete_fixture.archive, bad, corrupt)
    with pytest.raises(ValueError, match='reservation identity differs'):
        portable.extract_bundle(complete_fixture.repo, bad, tmp_path/'bad', tmp_path/'bad.json')
