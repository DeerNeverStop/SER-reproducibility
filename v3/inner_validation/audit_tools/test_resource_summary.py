"""Synthetic metadata only; invalid model/NPZ bytes deliberately remain unread."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from v3.inner_validation.audit_tools import resource_summary as audit


def unit(index=0):
    return {'unit_id': f'dual_d{index//20:02d}_f{index//4%5}_c{index%4}',
            'draw': index//20, 'fold': index//4%5, 'config_index': index%4,
            'fit': ['synthetic']*576, 'config': {'epochs': 15, 'batch_size': 16}}


def data(u, phase='formal'):
    receipt = {'schema': 'ser-dual-validation-receipt-1', 'unit_id': u['unit_id'], 'phase': phase,
               'plan_sha256': audit.FROZEN_PLAN_SHA, 'unit_sha256': audit.content_hash(u),
               'fit_seconds': 2., 'peak_cuda_bytes': 1000, 'epochs_run': 15,
               'checkpoint_reload_verified': True, 'checkpoint_unique_epochs': [15],
               'checkpoint_unique_epoch_count': 1, 'best_seen_epoch': 15, 'best_unseen_epoch': 15,
               'checkpoint_format': 'wavlm_dual_validation_epoch_deltas',
               'environment': dict(python='synthetic', numpy='synthetic', torch='synthetic', cuda='synthetic',
                                   cudnn=1, gpu='synthetic', threads=1, matmul_tf32=False, cudnn_tf32=True)}
    history = [{'epoch': i, 'optimizer_steps': 36, 'scaler_skipped_steps': int(i == 1)} for i in range(1, 16)]
    return receipt, history


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


@pytest.fixture
def complete(tmp_path):
    units = [unit(i) for i in range(480)]
    plan = {'plan_sha256': audit.FROZEN_PLAN_SHA, 'units': units, 'pilot_units': [u['unit_id'] for u in units[:4]]}
    gates = {}
    for phase, own in (('formal', units), ('pilot', units[:4])):
        mapping, events = {}, []
        for index, u in enumerate(own):
            folder = tmp_path/phase/'units'/u['unit_id']; attempt = folder/'attempts/0001'
            receipt, history = data(u, phase)
            write(attempt/'receipt.json', receipt); write(attempt/'history.json', history)
            (attempt/'checkpoint.pt').write_bytes(b'invalid torch payload')
            (attempt/'predictions.npz').write_bytes(b'invalid numpy payload')
            done = {'schema': 'ser-dual-validation-done-1', 'unit_id': u['unit_id'], 'phase': phase,
                    'plan_sha256': plan['plan_sha256'], 'unit_sha256': audit.content_hash(u),
                    'artifacts': {'attempts/0001/'+n: audit.file_hash(attempt/n) for n in audit.ARTIFACTS}}
            write(folder/'DONE', done); mapping[u['unit_id']] = audit.file_hash(folder/'DONE')
            for event, delta in (('unit_start', 0), ('unit_done', 1)):
                when = datetime(2026, 1, 2 if phase == 'formal' else 1, tzinfo=timezone.utc)+timedelta(seconds=2*index+delta)
                events.append({'event': event, 'unit_id': u['unit_id'], 'time': when.isoformat()})
        ledger = tmp_path/phase/'ledger.jsonl'
        ledger.write_text(''.join(json.dumps(e)+'\n' for e in events), encoding='utf-8', newline='\n')
        gate = {'schema': 'ser-dual-validation-result-gate-1', 'pass': True, 'phase': phase,
                'plan_sha256': plan['plan_sha256'], 'units': len(own), 'done_sha256': mapping,
                'ledger_sha256': audit.file_hash(ledger), 'verified_at': 'synthetic', 'scores_computed': False}
        gate['gate_sha256'] = audit.content_hash(gate); gates[phase] = gate
    return plan, tmp_path, gates


def test_attempted_skipped_and_deduplicated_state_resources():
    u = unit(); receipt, history = data(u)
    history[1]['scaler_skipped_steps'] = 2
    record = audit.unit_resources(u, 'formal', receipt, history, {n: 1 for n in (*audit.ARTIFACTS, 'DONE')})
    result = audit.aggregate([record, record])
    assert record['optimizer_updates_attempted'] == 540
    assert record['amp_updates_skipped_by_scale_decrease'] == 3
    assert record['optimizer_updates_not_skipped'] == 537
    assert result['checkpoint_unique_states_total'] == 2 and result['epochs_total'] == 30
    assert result['fit_seconds']['sum'] == 4 and 'sum' not in result['peak_cuda_allocated_bytes']
    assert next(iter(result['environment_sets'].values()))['units'] == 2


@pytest.mark.parametrize('bad', ['skips', 'attempts', 'epochs'])
def test_invalid_epoch_update_evidence_is_rejected(bad):
    u = unit(); receipt, history = data(u)
    if bad == 'skips': history[0]['scaler_skipped_steps'] = 37
    elif bad == 'attempts': history[0]['optimizer_steps'] = 35
    else: history.pop()
    with pytest.raises(ValueError):
        audit.unit_resources(u, 'formal', receipt, history, {n: 1 for n in (*audit.ARTIFACTS, 'DONE')})


def test_incomplete_gate_rejected_before_results_are_opened(tmp_path):
    units = [unit(i) for i in range(480)]
    plan = {'plan_sha256': audit.FROZEN_PLAN_SHA, 'units': units, 'pilot_units': [u['unit_id'] for u in units[:4]]}
    with pytest.raises(ValueError, match='both formal and pilot'):
        audit.collect(plan, tmp_path, {})
    gate = {'schema': 'ser-dual-validation-result-gate-1', 'pass': True, 'phase': 'formal',
            'plan_sha256': plan['plan_sha256'], 'units': 8, 'done_sha256': {}}
    with pytest.raises(ValueError, match='population mismatch'):
        audit.collect(plan, tmp_path, {'formal': gate, 'pilot': {}})


def test_full_484_collection_does_not_read_large_payloads(complete, monkeypatch):
    plan, outroot, gates = complete
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.suffix not in ('.pt', '.npz'), 'large payload must remain unread by resource audit'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    report = audit.collect(plan, outroot, gates)
    assert report['phases']['formal']['units'] == 480 and report['phases']['pilot']['units'] == 4
    assert report['combined_484_executed_fits']['fit_seconds']['sum'] == 968
    assert report['combined_484_executed_fits']['optimizer_updates_attempted'] == 484*540
    assert report['combined_484_executed_fits']['amp_updates_skipped_by_scale_decrease'] == 484
    assert report['phases']['formal']['recorded_unit_failures'] == 0
    assert report['gpu_rental_elapsed_seconds'] is None and report['actual_bill'] is None
    assert not report['scientific_scores_computed'] and not report['integrity_scope']['large_payloads_rehashed_now']


def test_receipt_mutation_after_passed_gate_fails(complete):
    plan, outroot, gates = complete
    receipt = outroot/'formal/units/dual_d00_f0_c0/attempts/0001/receipt.json'
    value = json.loads(receipt.read_text()); value['fit_seconds'] = 1.
    write(receipt, value)
    with pytest.raises(ValueError, match='bound metadata changed'):
        audit.collect(plan, outroot, gates)
