"""Full-count/ledger glue tests; mock payload validation, never read study scores.

The real metadata planner is run once. No weights, NPZ or waveform payload is
created/read; verify_done is mocked. Passing these tests cannot certify weights.
Expected-rejection cases deliberately remain failing until the formal gate fix.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from v3.final_program_20260907 import plan as planner
from v3.final_program_20260907 import run


REPO = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')


@pytest.fixture(scope='module')
def canonical_plan():
    value = planner.generate(REPO)
    assert planner.validate(value, REPO)
    return value


@pytest.fixture
def archive(tmp_path, monkeypatch, canonical_plan):
    plan = deepcopy(canonical_plan)
    units = [u for u in plan['units'] if u['phase'] == 'formal']
    lock = dict(phase='formal', lock_sha256='a'*64, plan_sha256=plan['plan_sha256'],
                phase_units=[u['unit_id'] for u in units])
    fixture = SimpleNamespace(out=tmp_path, plan=plan, lock=lock, units=units,
                              done={}, receipt={}, ledger=[], verified=[], checked=0)
    for unit in units:
        uid = unit['unit_id']
        folder = tmp_path/'units'/uid
        done = dict(schema='SYNTHETIC_GLUE_ONLY', unit_id=uid, unit_sha256=run.digest(unit),
                    lock_sha256=lock['lock_sha256'], attempt='attempts/0001', artifacts={})
        dump(folder/'DONE', done)
        dump(folder/'reservations'/'0001.json', dict(unit_id=uid, unit_sha256=run.digest(unit),
             lock_sha256=lock['lock_sha256'], time='2026-09-07T00:00:00+00:00'))
        fixture.done[uid] = done
        fixture.receipt[uid] = {'info': {'initial_state_sha256': run.digest(unit['pair_id'])}}
        fixture.ledger.extend([
            dict(time='2026-09-07T00:00:00+00:00', event='unit_start', unit_id=uid, attempt='attempts/0001'),
            dict(time='2026-09-07T00:00:01+00:00', event='unit_done', unit_id=uid,
                 done_sha256=run.file_sha(folder/'DONE'))])
    write_ledger(fixture)

    def checked(repo, out):
        assert repo == REPO and out == fixture.out
        fixture.checked += 1
        # Normal fixture was really validated above. On tampering, invoke the
        # actual replay (rather than a fake role-specific rejection).
        if fixture.plan != canonical_plan:
            planner.validate(fixture.plan, REPO)
        return fixture.plan, fixture.lock

    def verified(unit, out, actual_lock, rows, weights=True, **kwargs):
        assert weights is True and actual_lock == fixture.lock
        assert rows == fixture.plan['rows'][unit['corpus']]
        fixture.verified.append(unit['unit_id'])
        return fixture.done[unit['unit_id']], fixture.receipt[unit['unit_id']]

    monkeypatch.setattr(run, 'checked_plan', checked)
    monkeypatch.setattr(run, 'verify_done', verified)
    return fixture


def write_ledger(f):
    (f.out/'ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in f.ledger), encoding='utf8')


def add(f, kind, uid=None, attempt=None):
    row = dict(time='2026-09-07T00:00:02+00:00', event=kind)
    if uid is not None:
        row['unit_id'] = uid
    if attempt is not None:
        row['attempt'] = attempt
    f.ledger.append(row)
    write_ledger(f)


def audit(f):
    return run.audit_phase(REPO, f.out)


def test_full_384_including_24_B_requires_every_unit(archive):
    f = archive
    result = audit(f)
    assert result['pass'] is True and result['verified_units'] == result['expected_units'] == 384
    assert set(result['done_sha256']) == {u['unit_id'] for u in f.units}
    assert len(f.verified) == 384 and f.checked == 2
    assert sum(u['analysis_role'] == 'checkpoint_main' for u in f.units) == 360
    assert sum(u['analysis_role'] == 'role_swap_control' for u in f.units) == 24
    assert result['ledger_sha256'] == run.file_sha(f.out/'ledger.jsonl')


@pytest.mark.parametrize('remove', ['A', 'B'])
def test_missing_A_or_control_DONE_never_seals(archive, remove):
    f = archive
    uid = next(u['unit_id'] for u in f.units if u['arm'] == remove)
    (f.out/'units'/uid/'DONE').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        audit(f)
    assert not (f.out/'COMPLETE_GATE.json').exists()


def test_extra_directory_is_not_ignored(archive):
    f = archive
    (f.out/'units'/'unexpected-extra-unit').mkdir()
    with pytest.raises(ValueError, match='unexpected'):
        audit(f)
    assert not (f.out/'COMPLETE_GATE.json').exists()


@pytest.mark.parametrize('kind', ['unit_start', 'unit_failed', 'unit_done'])
def test_unknown_ledger_unit_prevents_full_closure(archive, kind):
    f = archive
    add(f, kind, 'NOT_IN_FROZEN_PLAN', 'attempts/0001')
    with pytest.raises(ValueError):
        audit(f)
    assert not (f.out/'COMPLETE_GATE.json').exists()


@pytest.mark.parametrize('kind', ['unit_start', 'unit_failed'])
def test_extra_attempt_cannot_be_ignored_by_current_DONE(archive, kind):
    f = archive
    add(f, kind, f.units[0]['unit_id'], 'attempts/0002')
    with pytest.raises(ValueError):
        audit(f)
    assert not (f.out/'COMPLETE_GATE.json').exists()


def test_duplicate_success_is_rejected(archive):
    f = archive
    f.ledger.append(deepcopy(f.ledger[1]))
    write_ledger(f)
    with pytest.raises(ValueError, match='duplicate'):
        audit(f)


def test_done_before_start_does_not_prove_a_completed_attempt(archive):
    f = archive
    f.ledger[0], f.ledger[1] = f.ledger[1], f.ledger[0]
    write_ledger(f)
    with pytest.raises(ValueError):
        audit(f)


def test_wrong_pair_initialization_rejects_full_phase(archive):
    f = archive
    uid = next(u['unit_id'] for u in f.units if u['arm'] == 'B')
    f.receipt[uid]['info']['initial_state_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='paired initial'):
        audit(f)


def test_B_cannot_be_relabelled_as_main_via_self_consistent_plan_hash(archive):
    f = archive
    u = next(u for u in f.plan['units'] if u['phase'] == 'formal' and u['arm'] == 'B')
    u['analysis_role'] = 'checkpoint_main'
    f.plan['plan_sha256'] = run.plan_digest(f.plan)
    with pytest.raises(ValueError, match='semantic replay'):
        audit(f)
    assert f.verified == []


def test_half_ledger_line_is_not_repaired_silently(archive):
    f = archive
    with (f.out/'ledger.jsonl').open('ab') as stream:
        stream.write(b'{"event":')
    original = (f.out/'ledger.jsonl').read_bytes()
    with pytest.raises(ValueError, match='incomplete ledger'):
        audit(f)
    assert (f.out/'ledger.jsonl').read_bytes() == original


def test_unexplained_reservation_cannot_be_ignored(archive):
    f = archive
    u = f.units[0]
    dump(f.out/'units'/u['unit_id']/'reservations'/'0002.json', dict(
        unit_id=u['unit_id'], unit_sha256=run.digest(u), lock_sha256=f.lock['lock_sha256']))
    with pytest.raises(ValueError):
        audit(f)


@pytest.mark.parametrize('first_attempt_closed', [True, False])
def test_retry_success_must_account_for_the_prior_attempt(archive, first_attempt_closed):
    f = archive
    u = f.units[0]
    uid = u['unit_id']
    folder = f.out/'units'/uid
    f.done[uid]['attempt'] = 'attempts/0002'
    dump(folder/'DONE', f.done[uid])
    dump(folder/'reservations'/'0002.json', dict(unit_id=uid, unit_sha256=run.digest(u),
         lock_sha256=f.lock['lock_sha256'], time='2026-09-07T00:00:02+00:00'))
    sequence = [f.ledger[0]]
    if first_attempt_closed:
        sequence.append(dict(time='2026-09-07T00:00:01+00:00', event='unit_failed',
                             unit_id=uid, attempt='attempts/0001'))
    sequence.extend([
        dict(time='2026-09-07T00:00:02+00:00', event='unit_start', unit_id=uid, attempt='attempts/0002'),
        dict(time='2026-09-07T00:00:03+00:00', event='unit_done', unit_id=uid,
             done_sha256=run.file_sha(folder/'DONE'))])
    f.ledger[:2] = sequence
    write_ledger(f)
    if first_attempt_closed:
        # A real, closed failed attempt followed by success is not a reason to
        # discard a complete run; it must remain visible as a recorded failure.
        assert audit(f)['pass'] is True
    else:
        with pytest.raises(ValueError):
            audit(f)
