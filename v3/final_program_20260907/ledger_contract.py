"""Complete phase attempt closure; no models, predictions or run-module import.

The caller must hold the phase OS lock and separately verify every committed
payload. Explicit abandonment records describe interrupted attempts; they do
not silently turn an interruption into a successful or never-started run.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


UNIT_EVENTS = {'unit_start', 'unit_done', 'unit_failed', 'unit_attempt_abandoned',
               'commit_requires_recovery'}
PROCESS_EVENTS = {'process_start', 'process_complete', 'operational_yield'}
ABANDON_REASON = 'phase_lock_reacquired_after_interruption'
ATTEMPT_RE = re.compile(r'attempts/(000[12])\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def attempt_number(value):
    require(isinstance(value, str), 'attempt must be a relative string')
    match = ATTEMPT_RE.fullmatch(value)
    require(match is not None, 'attempt must be attempts/0001 or attempts/0002')
    return int(match[1])


def _pinned_json(path, out, pins):
    path = Path(path)
    require(path.resolve().is_relative_to(out) and path.is_file(),
            'missing or unsafe attempt metadata: '+str(path))
    raw = path.read_bytes()
    pins[path.relative_to(out).as_posix()] = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    require(isinstance(value, dict), 'attempt metadata must be a JSON object')
    return value


def audit_events(units, out, lock, events):
    """Validate all reservations/events and return count and small-file pins.

    ``events`` must be the caller's complete, already parsed phase ledger in its
    original line order. This function does not repair or append any record.
    A valid completed phase may retain failed/abandoned attempt 0001 followed
    by successful attempt 0002; all such attempts are explicitly counted.
    """
    out = Path(out).resolve()
    require(isinstance(units, list) and units, 'declared phase units required')
    require(isinstance(events, list), 'complete ordered ledger list required')
    registry = {}
    for unit in units:
        uid = unit.get('unit_id')
        require(isinstance(uid, str) and re.fullmatch(r'[A-Za-z0-9_-]+', uid), 'unsafe unit id')
        require(uid not in registry and unit.get('phase') == lock['phase'], 'unit phase/uniqueness differs')
        registry[uid] = unit
    actual_units = {p.name for p in (out/'units').iterdir() if p.is_dir()}
    require(actual_units == set(registry), 'unit directory set differs from frozen phase')

    pins, states, done_by_unit, done_hashes = {}, {}, {}, {}
    for uid, unit in registry.items():
        folder = out/'units'/uid
        done_path = folder/'DONE'
        done = _pinned_json(done_path, out, pins)
        require(done.get('unit_id') == uid and done.get('unit_sha256') == digest(unit)
                and done.get('lock_sha256') == lock['lock_sha256'], 'DONE identity differs')
        final_index = attempt_number(done.get('attempt'))
        done_by_unit[uid] = done
        done_hashes[uid] = pins[done_path.relative_to(out).as_posix()]
        reservation_folder = folder/'reservations'
        require(reservation_folder.is_dir(), 'attempt reservations missing')
        reservations = list(reservation_folder.iterdir())
        require(reservations and all(p.is_file() and re.fullmatch(r'000[12]\.json', p.name)
                                    for p in reservations), 'invalid or partial reservation inventory')
        indexes = sorted(int(p.stem) for p in reservations)
        require(indexes == list(range(1, len(indexes)+1)) and len(indexes) <= 2,
                'reservation numbers must be contiguous, with at most two attempts')
        require(final_index == indexes[-1], 'reservation exists after the committed successful attempt')
        for path in reservations:
            reservation = _pinned_json(path, out, pins)
            require(reservation.get('unit_id') == uid and reservation.get('unit_sha256') == digest(unit)
                    and reservation.get('lock_sha256') == lock['lock_sha256'], 'reservation identity differs')
            states[uid, int(path.stem)] = dict(start_index=None, terminal_index=None, status='reserved')
        attempts_folder = folder/'attempts'
        if attempts_folder.exists():
            require(attempts_folder.is_dir(), 'attempt inventory is not a directory')
            for path in attempts_folder.iterdir():
                require(path.is_dir() and re.fullmatch(r'000[12]', path.name)
                        and (uid, int(path.name)) in states and path.resolve().is_relative_to(out),
                        'unreserved or unsafe attempt directory')

    recovery_notes = 0
    for index, event in enumerate(events):
        require(isinstance(event, dict), 'ledger record must be an object')
        kind = event.get('event')
        require(kind in UNIT_EVENTS | PROCESS_EVENTS, 'unknown ledger event')
        if kind in PROCESS_EVENTS:
            require('unit_id' not in event and 'attempt' not in event, 'unit fields in process event')
            if kind == 'process_start':
                require(event.get('lock_sha256') == lock['lock_sha256'], 'process uses a different source lock')
            else:
                require(type(event.get('completed_here')) is int
                        and 0 <= event['completed_here'] <= len(units), 'invalid process completion count')
            continue
        uid = event.get('unit_id')
        require(uid in registry, 'ledger references unit outside frozen phase')
        done = done_by_unit[uid]
        attempt = event.get('attempt', done['attempt'] if kind == 'unit_done' else None)
        number = attempt_number(attempt)
        require((uid, number) in states, 'ledger references an unreserved attempt')
        state = states[uid, number]
        if kind == 'unit_start':
            require(state['status'] == 'reserved' and state['start_index'] is None,
                    'attempt has duplicate or post-terminal start')
            require(all(states[uid, old]['status'] in ('failed', 'abandoned') for old in range(1, number)),
                    'retry started before the prior attempt was explicitly closed')
            require(not any(s['status'] == 'done' for (person, _), s in states.items() if person == uid),
                    'new attempt after unit success')
            state.update(status='started', start_index=index)
        elif kind == 'unit_failed':
            require(state['status'] == 'started' and state['start_index'] < index,
                    'failure has no unique preceding open start')
            state.update(status='failed', terminal_index=index)
        elif kind == 'unit_done':
            require(attempt == done['attempt'] and event.get('done_sha256') == done_hashes[uid],
                    'ledger completion does not bind current DONE')
            require(state['status'] == 'started' and state['start_index'] < index,
                    'completion has no unique preceding open start')
            require(not any(s['status'] == 'done' for (person, _), s in states.items() if person == uid),
                    'duplicate unit completion')
            state.update(status='done', terminal_index=index)
        elif kind == 'unit_attempt_abandoned':
            require(event.get('reason') == ABANDON_REASON and type(event.get('had_start')) is bool,
                    'abandonment must explicitly explain phase-lock interruption recovery')
            had_start = state['start_index'] is not None
            require(event['had_start'] is had_start and state['status'] in ('reserved', 'started'),
                    'abandonment contradicts recorded attempt state')
            require(not any(s['status'] == 'done' for (person, _), s in states.items() if person == uid),
                    'abandonment after unit success')
            state.update(status='abandoned', terminal_index=index)
        else:  # A durable DONE can exist before its missing ledger completion is repaired.
            require(attempt == done['attempt'] and state['status'] in ('started', 'done')
                    and state['start_index'] is not None and state['start_index'] < index,
                    'commit recovery annotation lacks a valid committed attempt')
            if 'done_sha256' in event:
                require(event['done_sha256'] == done_hashes[uid], 'commit recovery DONE hash differs')
            recovery_notes += 1

    for (uid, number), state in states.items():
        require(state['status'] in ('done', 'failed', 'abandoned'),
                'unclosed reservation or started attempt: '+uid+f'/attempts/{number:04d}')
    for uid, done in done_by_unit.items():
        successes = [n for (person, n), s in states.items() if person == uid and s['status'] == 'done']
        require(successes == [attempt_number(done['attempt'])], 'unit lacks its unique committed completion')
    # Bind the exact small files used, and fail if anything changed during audit.
    for rel, expected in pins.items():
        require(file_sha(out/rel) == expected, 'attempt metadata changed during audit')
    counts = Counter(state['status'] for state in states.values())
    return dict(schema='ser-final-program-ledger-contract-1', passed=True,
                phase=lock['phase'], lock_sha256=lock['lock_sha256'], units=len(units),
                events=len(events), events_semantic_sha256=digest(events),
                attempt_counts=dict(reserved=len(states), started=sum(s['start_index'] is not None for s in states.values()),
                                    done=counts['done'], failed=counts['failed'], abandoned=counts['abandoned'],
                                    abandoned_without_start=sum(s['status']=='abandoned' and s['start_index'] is None
                                                                for s in states.values())),
                commit_recovery_annotations=recovery_notes,
                reservation_sha256={p: h for p, h in pins.items() if '/reservations/' in p},
                done_sha256=done_hashes,
                attempt_states={uid+f'/attempts/{n:04d}': s['status'] for (uid, n), s in states.items()},
                scope='Complete ordered ledger and reservation closure; payload verification is a separate caller gate')
