"""Resources for the complete 480 formal + four pilot fits, after both full gates.

No logits, models, accuracy, or scientific scores are loaded. The prior frozen
result gates are required integrity evidence; large payloads are only stat'ed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

FROZEN_PLAN_SHA = '570cae10577e13bc17a5838d872251beeb52d8b216886d19a8dcb6418a7f5bbd'
ARTIFACTS = ('checkpoint.pt', 'predictions.npz', 'receipt.json', 'history.json')
EXPECTED = {'formal': 480, 'pilot': 4}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def content_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid nonnegative integer resource value')
    return value


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid finite resource value')
    return float(value)


def stats(values):
    values = list(values)
    require(values, 'empty resource population')
    return {'sum': math.fsum(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def read_bound(path, expected=None, pins=None):
    path = Path(path).resolve()
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if expected is not None:
        require(sha == expected, 'bound metadata changed: ' + str(path))
    if pins is not None:
        pins[str(path)] = sha
    return json.loads(payload)


def unit_resources(unit, phase, receipt, history, payload_sizes):
    require(receipt['unit_id'] == unit['unit_id'] and receipt['phase'] == phase, 'receipt unit/phase mismatch')
    require(receipt['schema'] == 'ser-dual-validation-receipt-1' and
            receipt['unit_sha256'] == content_hash(unit), 'receipt configuration identity mismatch')
    require(receipt['epochs_run'] == unit['config']['epochs'] == 15 and
            receipt['checkpoint_reload_verified'] is True, 'full epochs/reload evidence missing')
    require(isinstance(history, list) and [r['epoch'] for r in history] == list(range(1, 16)), 'incomplete epoch history')
    selected = {'best_seen': integer(receipt['best_seen_epoch'], 1),
                'best_unseen': integer(receipt['best_unseen_epoch'], 1), 'last': 15}
    require(max(selected.values()) <= 15, 'checkpoint epoch outside training trajectory')
    epochs = sorted(set(selected.values()))
    require(receipt['checkpoint_unique_epochs'] == epochs and
            receipt['checkpoint_unique_epoch_count'] == len(epochs), 'checkpoint state count mismatch')
    per_epoch = math.ceil(len(unit['fit']) / integer(unit['config']['batch_size'], 1))
    attempted = skipped = 0
    for row in history:
        steps, omitted = integer(row['optimizer_steps']), integer(row['scaler_skipped_steps'])
        require(steps == per_epoch and omitted <= steps, 'attempted/skipped update history mismatch')
        attempted += steps
        skipped += omitted
    environment = receipt['environment']
    require(isinstance(environment, dict) and all(k in environment for k in
            ('python', 'numpy', 'torch', 'cuda', 'cudnn', 'gpu', 'threads', 'matmul_tf32', 'cudnn_tf32')),
            'incomplete recorded software environment')
    require(set(payload_sizes) == set(ARTIFACTS) | {'DONE'}, 'payload size population mismatch')
    return {'phase': phase, 'unit_id': unit['unit_id'], 'draw': unit['draw'], 'fold': unit['fold'],
            'config_index': unit['config_index'], 'fit_seconds': number(receipt['fit_seconds']),
            'peak_cuda_allocated_bytes': integer(receipt['peak_cuda_bytes']),
            'epochs_run': 15, 'optimizer_updates_attempted': attempted,
            'amp_updates_skipped_by_scale_decrease': skipped,
            'optimizer_updates_not_skipped': attempted - skipped,
            'selected_epochs': selected, 'checkpoint_unique_states': len(epochs),
            'checkpoint_format': receipt['checkpoint_format'],
            'payload_bytes': {k: integer(v) for k, v in payload_sizes.items()},
            'environment_id': content_hash(environment), 'environment': environment}


def aggregate(records):
    attempted = sum(r['optimizer_updates_attempted'] for r in records)
    skipped = sum(r['amp_updates_skipped_by_scale_decrease'] for r in records)
    environments = {}
    for r in records:
        entry = environments.setdefault(r['environment_id'], {'environment': r['environment'], 'units': 0})
        entry['units'] += 1
    byte_totals = {name: sum(r['payload_bytes'][name] for r in records) for name in (*ARTIFACTS, 'DONE')}
    return {'units': len(records), 'fit_seconds': stats(r['fit_seconds'] for r in records),
            'peak_cuda_allocated_bytes': {k: v for k, v in stats(r['peak_cuda_allocated_bytes'] for r in records).items() if k != 'sum'},
            'epochs_total': sum(r['epochs_run'] for r in records),
            'optimizer_updates_attempted': attempted,
            'amp_updates_skipped_by_scale_decrease': skipped,
            'optimizer_updates_not_skipped': attempted - skipped,
            'amp_skipped_fraction_of_attempted': skipped / attempted if attempted else None,
            'checkpoint_unique_states_total': sum(r['checkpoint_unique_states'] for r in records),
            'checkpoint_unique_states_histogram': dict(sorted(Counter(str(r['checkpoint_unique_states']) for r in records).items())),
            'checkpoint_formats': dict(sorted(Counter(r['checkpoint_format'] for r in records).items())),
            'payload_bytes_by_file': byte_totals, 'payload_bytes_total': sum(byte_totals.values()),
            'environment_sets': environments}


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'ledger time lacks timezone')
    return result


def collect(plan, outroot, gates):
    """Strict complete-grid collection; no partial-report mode or score access."""
    require(plan['plan_sha256'] == FROZEN_PLAN_SHA, 'not the fixed dual-validation plan')
    require(set(gates) == set(EXPECTED), 'both formal and pilot full gates are required')
    all_units = plan['units']
    formal_ids = [u['unit_id'] for u in all_units]
    require(len(formal_ids) == len(set(formal_ids)) == 480 and len(set(plan['pilot_units'])) == 4,
            'plan does not describe 480 formal and four pilot units')
    outroot = Path(outroot).resolve()
    pins, records, phases, all_times = {}, [], {}, []
    for phase in ('formal', 'pilot'):
        units = all_units if phase == 'formal' else [u for u in all_units if u['unit_id'] in plan['pilot_units']]
        ids = [u['unit_id'] for u in units]
        require(len(ids) == EXPECTED[phase], 'wrong phase population')
        gate = gates[phase]
        require(gate['schema'] == 'ser-dual-validation-result-gate-1' and gate['pass'] is True and
                gate['phase'] == phase and gate['plan_sha256'] == plan['plan_sha256'] and
                gate['units'] == EXPECTED[phase] and set(gate['done_sha256']) == set(ids),
                'full gate identity/population mismatch: ' + phase)
        require(content_hash({k: v for k, v in gate.items() if k != 'gate_sha256'}) == gate['gate_sha256'],
                'full gate content hash mismatch')
        root = outroot / phase
        ledger = root / 'ledger.jsonl'
        payload = ledger.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == gate['ledger_sha256'] and payload.endswith(b'\n'),
                'ledger changed after full gate')
        pins[str(ledger)] = gate['ledger_sha256']
        events = [json.loads(line) for line in payload.decode().splitlines()]
        starts = [e for e in events if e['event'] == 'unit_start']
        stops = [e for e in events if e['event'] == 'unit_done']
        failures = [e for e in events if e['event'] == 'unit_failed']
        require(Counter(e['unit_id'] for e in starts) == Counter(e['unit_id'] for e in stops) == Counter(ids) and
                not failures, 'accepted ledger lacks unique successful closures')
        phase_records = []
        for unit in units:
            uid = unit['unit_id']; folder = root / 'units' / uid
            done = read_bound(folder / 'DONE', gate['done_sha256'][uid], pins)
            require(done['schema'] == 'ser-dual-validation-done-1' and done['unit_id'] == uid and
                    done['phase'] == phase and done['plan_sha256'] == plan['plan_sha256'] and
                    done['unit_sha256'] == content_hash(unit), 'DONE identity mismatch')
            expected = {'attempts/0001/' + name for name in ARTIFACTS}
            require(set(done['artifacts']) == expected, 'unexpected artifact list')
            require({p.name for p in (folder / 'attempts').iterdir()} == {'0001'}, 'unaccounted attempt directory')
            attempt = folder / 'attempts/0001'
            receipt = read_bound(attempt / 'receipt.json', done['artifacts']['attempts/0001/receipt.json'], pins)
            history = read_bound(attempt / 'history.json', done['artifacts']['attempts/0001/history.json'], pins)
            require(receipt['plan_sha256'] == plan['plan_sha256'], 'receipt plan differs')
            sizes = {'DONE': (folder / 'DONE').stat().st_size}
            for name in ARTIFACTS:
                path = attempt / name
                require(path.is_file() and not path.is_symlink(), 'missing/linked payload: ' + str(path))
                sizes[name] = path.stat().st_size
            phase_records.append(unit_resources(unit, phase, receipt, history, sizes))
        first = min(timestamp(e['time']) for e in starts)
        last = max(timestamp(e['time']) for e in stops)
        require(last >= first, 'negative observed ledger interval')
        all_times.extend((first, last))
        summary = aggregate(phase_records)
        summary.update(gate_sha256=gate['gate_sha256'], recorded_unit_starts=len(starts),
                       recorded_unit_completions=len(stops), recorded_unit_failures=len(failures),
                       observed_attempt_directories=len(units),
                       first_unit_start=first.isoformat(), last_completion_event=last.isoformat(),
                       observed_ledger_interval_seconds=(last-first).total_seconds(),
                       by_configuration={str(i): aggregate([r for r in phase_records if r['config_index'] == i]) for i in range(4)})
        phases[phase] = summary
        records.extend(phase_records)
    return {'schema': 'ser-dual-validation-resource-summary-1', 'pass': True,
            'plan_sha256': plan['plan_sha256'], 'scientific_scores_computed': False,
            'phases': phases, 'combined_484_executed_fits': aggregate(records),
            'observed_interval_first_start_to_last_completion_seconds': (max(all_times)-min(all_times)).total_seconds(),
            'gpu_rental_elapsed_seconds': None, 'actual_bill': None, 'unrecorded_failed_work_seconds': None,
            'definitions': {
                'fit_seconds': 'Host perf_counter interval covering CUDA cache/reset plus fit_dual: model/base loading and hashes, uncached waveform preprocessing, all 15 training epochs and both validations, checkpoint creation and fsync, selected/last inference, fresh-model disk reload and all replay predictions. Excludes subsequent predictions/history/receipt serialization, DONE sealing and runner final verification. Not GPU busy time, rental time or a bill.',
                'peak_cuda_allocated_bytes': 'torch.cuda.max_memory_allocated after per-fit reset; not reserved memory, whole-process nvidia-smi usage or total board memory.',
                'optimizer_updates_attempted': 'Sum of history.optimizer_steps, including AMP-skipped minibatches. Skips are inferred from a GradScaler scale decrease; not-skipped = attempted minus these recorded skips.',
                'checkpoint_states': 'Distinct saved epoch deltas per fit, deduplicated when best_seen/best_unseen/last select the same epoch. Counts do not equal total trained epochs.',
                'payload_bytes': 'Current sizes of each unit checkpoint, predictions, history, receipt and DONE; excludes base model, ledger, transfer archives, audit files, raw audio and features.',
                'observed_ledger_interval': 'Elapsed first unit_start to last unit_done event, including gaps between batches and any delayed recovery event; not a sum of GPU work and not the complete rental interval.',
                'failures': 'Zero failures/unclosed units in these accepted full-gate ledgers does not establish zero failures or zero unrecorded work throughout the rental.'},
            'integrity_scope': {
                'requires_prior_full_result_gates': True, 'formal_units': 480, 'pilot_units': 4,
                'currently_rehashed': 'Gate/plan/source in CLI; all DONE, receipt, history and both ledger files.',
                'large_payloads_rehashed_now': False,
                'large_payload_limit': 'Checkpoint/prediction files are only checked for existence/type and stat size here. Their byte validity depends on the supplied prior full gates; this report does not replace a fresh full artifact gate.',
                'logits_loaded': False},
            'unit_resources': records, 'metadata_file_sha256': pins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'plan', 'outroot', 'formal-gate', 'pilot-gate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repo, outroot, output = args.repo.resolve(), args.outroot.resolve(), args.out.resolve()
    require(not output.exists() and not output.is_relative_to(repo) and not output.is_relative_to(outroot),
            'output must be a new file outside repository and result tree')
    operator_sha = file_hash(__file__)
    require(Path(__file__).resolve() == repo / 'v3/inner_validation/audit_tools/resource_summary.py', 'operator is outside requested repo')
    sys.path.insert(0, str(repo))
    planner = importlib.import_module('v3.inner_validation.plan')
    require(Path(planner.__file__).resolve() == repo / 'v3/inner_validation/plan.py', 'loaded planner is outside requested repo')
    plan_sha = file_hash(args.plan)
    plan = planner.load_plan(args.plan, repo)
    gate_paths = {'formal': args.formal_gate.resolve(), 'pilot': args.pilot_gate.resolve()}
    gate_pins = {str(p): file_hash(p) for p in gate_paths.values()}
    gates = {phase: read_bound(p, gate_pins[str(p)]) for phase, p in gate_paths.items()}
    report = collect(plan, outroot, gates)
    require(file_hash(args.plan) == plan_sha and file_hash(__file__) == operator_sha and
            planner.source_files(repo) == plan['sources'], 'operator/plan/source changed during resource audit')
    for path, sha in {**report['metadata_file_sha256'], **gate_pins}.items():
        require(file_hash(path) == sha, 'bound metadata changed during resource audit: ' + path)
    report.update(plan_file_sha256=plan_sha, operator_source_sha256=operator_sha,
                  gate_file_sha256=gate_pins, source_sha256=plan['sources'],
                  generated_at=datetime.now(timezone.utc).isoformat())
    report['report_sha256'] = content_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({'pass': True, 'formal_units': 480, 'pilot_units': 4,
                      'scientific_scores_computed': False, 'out': str(output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
