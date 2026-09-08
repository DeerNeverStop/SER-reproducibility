"""Small CPU-only attempt state machines, independent of run/model payloads."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from v3.final_program_20260907 import ledger_contract as contract
from v3.final_program_20260907.test_full_gate import canonical_plan, archive


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def phase(tmp_path):
    units = [dict(unit_id='SYNTHETIC_U0', phase='formal'), dict(unit_id='SYNTHETIC_U1', phase='formal')]
    lock = dict(phase='formal', lock_sha256='a'*64)
    f = SimpleNamespace(out=tmp_path, units=units, lock=lock, events=[], done={})
    for unit in units:
        uid = unit['unit_id']
        folder = tmp_path/'units'/uid
        reserve(f, uid, 1)
        done = dict(unit_id=uid, unit_sha256=contract.digest(unit), lock_sha256=lock['lock_sha256'],
                    attempt='attempts/0001', artifacts={})
        dump(folder/'DONE', done)
        f.done[uid] = done
        f.events += [dict(event='unit_start', unit_id=uid, attempt='attempts/0001'),
                     dict(event='unit_done', unit_id=uid, done_sha256=contract.file_sha(folder/'DONE'))]
    return f


def reserve(f, uid, number):
    unit = next(u for u in f.units if u['unit_id'] == uid)
    dump(f.out/'units'/uid/'reservations'/f'{number:04d}.json', dict(unit_id=uid,
         unit_sha256=contract.digest(unit), lock_sha256=f.lock['lock_sha256']))


def audit(f):
    return contract.audit_events(f.units, f.out, f.lock, f.events)


def retry(f, *, terminal='unit_failed', had_start=True):
    uid = f.units[0]['unit_id']
    reserve(f, uid, 2)
    f.done[uid]['attempt'] = 'attempts/0002'
    dump(f.out/'units'/uid/'DONE', f.done[uid])
    first = [dict(event='unit_start', unit_id=uid, attempt='attempts/0001')] if had_start else []
    if terminal:
        event = dict(event=terminal, unit_id=uid, attempt='attempts/0001')
        if terminal == 'unit_attempt_abandoned':
            event.update(had_start=had_start, reason=contract.ABANDON_REASON)
        first.append(event)
    first += [dict(event='unit_start', unit_id=uid, attempt='attempts/0002'),
              dict(event='unit_done', unit_id=uid,
                   done_sha256=contract.file_sha(f.out/'units'/uid/'DONE'))]
    f.events[:2] = first
    return uid


def test_complete_phase_has_exact_pins_and_no_mutations(phase):
    f = phase
    before = deepcopy(f.events)
    result = audit(f)
    assert result['passed'] and result['units'] == 2
    assert result['attempt_counts'] == dict(reserved=2, started=2, done=2, failed=0,
                                           abandoned=0, abandoned_without_start=0)
    assert len(result['reservation_sha256']) == len(result['done_sha256']) == 2
    assert f.events == before


def test_closed_failure_then_retry_is_counted_not_erased(phase):
    retry(phase)
    counts = audit(phase)['attempt_counts']
    assert counts == dict(reserved=3, started=3, done=2, failed=1, abandoned=0, abandoned_without_start=0)


@pytest.mark.parametrize('had_start', [False, True])
def test_explicit_lock_reacquisition_abandons_orphan(phase, had_start):
    retry(phase, terminal='unit_attempt_abandoned', had_start=had_start)
    counts = audit(phase)['attempt_counts']
    assert counts['done'] == 2 and counts['abandoned'] == 1 and counts['failed'] == 0
    assert counts['abandoned_without_start'] == int(not had_start)


@pytest.mark.parametrize('field,value', [('had_start', False), ('had_start', 1),
                                       ('reason', 'just discard old evidence')])
def test_abandonment_cannot_lie_about_start_or_reason(phase, field, value):
    retry(phase, terminal='unit_attempt_abandoned', had_start=True)
    phase.events[1][field] = value
    with pytest.raises(ValueError):
        audit(phase)


def test_successful_retry_cannot_hide_unclosed_first_attempt(phase):
    retry(phase, terminal=None)
    with pytest.raises(ValueError, match='prior attempt'):
        audit(phase)


@pytest.mark.parametrize('kind', ['unit_start', 'unit_done', 'unit_failed', 'unit_attempt_abandoned'])
def test_unknown_unit_cannot_be_ignored(phase, kind):
    phase.events.append(dict(event=kind, unit_id='FOREIGN', attempt='attempts/0001'))
    with pytest.raises(ValueError, match='outside frozen phase'):
        audit(phase)


def test_unknown_event_is_rejected(phase):
    phase.events.append(dict(event='silently_forget_attempt'))
    with pytest.raises(ValueError, match='unknown ledger event'):
        audit(phase)


@pytest.mark.parametrize('fault', ['done_before_start', 'duplicate_start', 'duplicate_done', 'failed_after_done'])
def test_attempt_order_and_single_terminal(phase, fault):
    f = phase
    if fault == 'done_before_start':
        f.events[0], f.events[1] = f.events[1], f.events[0]
    elif fault == 'duplicate_start':
        f.events.insert(1, deepcopy(f.events[0]))
    elif fault == 'duplicate_done':
        f.events.insert(2, deepcopy(f.events[1]))
    else:
        f.events.insert(2, dict(event='unit_failed', unit_id=f.units[0]['unit_id'], attempt='attempts/0001'))
    with pytest.raises(ValueError):
        audit(f)


def test_commit_annotation_requires_eventual_hash_bound_completion(phase):
    f = phase
    uid = f.units[0]['unit_id']
    f.events.insert(1, dict(event='commit_requires_recovery', unit_id=uid, attempt='attempts/0001'))
    assert audit(f)['commit_recovery_annotations'] == 1
    del f.events[2]
    with pytest.raises(ValueError, match='unclosed'):
        audit(f)


@pytest.mark.parametrize('fault', ['wrong_done_sha', 'wrong_done_attempt', 'wrong_reservation',
                                  'extra_reservation', 'extra_attempt', 'partial_reservation'])
def test_metadata_and_all_attempt_inventory_are_closed(phase, fault):
    f = phase
    uid = f.units[0]['unit_id']
    folder = f.out/'units'/uid
    if fault == 'wrong_done_sha':
        f.events[1]['done_sha256'] = 'b'*64
    elif fault == 'wrong_done_attempt':
        f.events[1]['attempt'] = 'attempts/0002'
    elif fault == 'wrong_reservation':
        dump(folder/'reservations'/'0001.json', {'unit_id': uid, 'unit_sha256': 'bad',
             'lock_sha256': f.lock['lock_sha256']})
    elif fault == 'extra_reservation':
        reserve(f, uid, 2)
    elif fault == 'extra_attempt':
        (folder/'attempts'/'0002').mkdir(parents=True)
    else:
        (folder/'reservations'/'0002.json.partial').write_bytes(b'{')
    with pytest.raises(ValueError):
        audit(f)


def test_explicit_correct_done_attempt_is_accepted(phase):
    phase.events[1]['attempt'] = 'attempts/0001'
    assert audit(phase)['passed']


def test_known_process_events_are_bounded_and_lock_bound(phase):
    f = phase
    f.events.insert(0, dict(event='process_start', lock_sha256=f.lock['lock_sha256']))
    f.events.append(dict(event='process_complete', completed_here=2))
    assert audit(f)['passed']
    f.events[0]['lock_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='different source lock'):
        audit(f)


def test_input_metadata_mutation_during_audit_rejected(phase, monkeypatch):
    f = phase
    original = contract.file_sha
    target = f.out/'units'/f.units[0]['unit_id']/'reservations'/'0001.json'
    def changed(path):
        if path == target:
            return 'f'*64
        return original(path)
    monkeypatch.setattr(contract, 'file_sha', changed)
    with pytest.raises(ValueError, match='changed during audit'):
        audit(f)


def test_full_384_metadata_fixture_has_exact_attempt_closure(archive):
    # Reuses the real-plan, toy-payload fixture. This does not validate weights.
    f = archive
    result = contract.audit_events(f.units, f.out, f.lock, f.ledger)
    assert result['units'] == result['attempt_counts']['done'] == 384
    assert result['attempt_counts']['reserved'] == result['attempt_counts']['started'] == 384
    assert len(result['reservation_sha256']) == len(result['done_sha256']) == 384
