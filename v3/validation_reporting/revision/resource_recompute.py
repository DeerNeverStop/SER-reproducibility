"""Read historical small resource metadata only; never load audio, logits or weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def quantile(values, q):
    a = sorted(values)
    h = (len(a) - 1) * q
    i = math.floor(h)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (h - i)


def stats(values):
    a = list(values)
    assert a and all(math.isfinite(x) for x in a)
    return dict(n=len(a), min=min(a), median=statistics.median(a), p90=quantile(a, .9),
                max=max(a), mean=statistics.mean(a), sum=sum(a))


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--dual-root', type=Path, required=True)
    ap.add_argument('--coverage-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), 'Output must be new; never overwrite prior audit'
    pins = {}

    def read(path):
        path = path.resolve()
        assert path.suffix in {'.json', '.jsonl', '.py'} or path.name == 'DONE', path
        b = path.read_bytes()
        pins[str(path)] = hashlib.sha256(b).hexdigest()
        return b

    def js(path):
        return json.loads(read(path))

    dual = args.dual_root
    source = js(dual / 'analysis/resource_summary.json')
    plan = js(dual / 'inputs/plan.json')
    assert source['pass'] and source['plan_sha256'] == plan['plan_sha256']
    formal_gate = js(dual / 'analysis/formal_gate.json')
    pilot_gate = js(dual / 'analysis/pilot_gate.json')
    assert formal_gate['pass'] and pilot_gate['pass']
    # These are previously accepted gates, not a new proof of large-blob integrity.
    for phase, gate in [('formal', formal_gate), ('pilot', pilot_gate)]:
        assert gate['plan_sha256'] == plan['plan_sha256']
        declared = {Path(k).name: v for k, v in source['gate_file_sha256'].items()}
        assert declared[f'{phase}_gate.json'] == pins[str((dual / f'analysis/{phase}_gate.json').resolve())]
    by_id = {u['unit_id']: u for u in plan['units']}
    assert len(by_id) == 480
    all_rows, phase_results, history_keys = [], {}, set()
    for phase, expected in [('formal', 480), ('pilot', 4)]:
        ledger_path = dual / 'runs' / phase / 'ledger.jsonl'
        events = [json.loads(line) for line in read(ledger_path).decode().splitlines()]
        starts, dones = {}, {}
        for event in events:
            if event['event'] == 'unit_start':
                assert event['unit_id'] not in starts
                starts[event['unit_id']] = event
            if event['event'] == 'unit_done':
                assert event['unit_id'] not in dones
                dones[event['unit_id']] = event
        assert not any(e['event'] == 'unit_failed' for e in events)
        units = [u for u in source['unit_resources'] if u['phase'] == phase]
        assert len(units) == len(starts) == len(dones) == expected
        if phase == 'formal':
            assert {u['unit_id'] for u in units} == set(by_id)
        rows = []
        for unit in units:
            uid = unit['unit_id']
            directory = dual / 'runs' / phase / 'units' / uid
            done = js(directory / 'DONE')
            assert done['unit_id'] == uid and done['plan_sha256'] == plan['plan_sha256']
            receipt_path = directory / 'attempts/0001/receipt.json'
            history_path = directory / 'attempts/0001/history.json'
            receipt, history = js(receipt_path), js(history_path)
            for path in (receipt_path, history_path):
                assert done['artifacts'][path.relative_to(directory).as_posix()] == pins[str(path.resolve())]
            assert receipt['unit_id'] == uid and receipt['phase'] == phase
            assert receipt['fit_seconds'] == unit['fit_seconds'] == dones[uid]['fit_seconds']
            assert receipt['epochs_run'] == len(history) == 15
            assert [h['epoch'] for h in history] == list(range(1, 16))
            assert all(h['optimizer_steps'] == 36 for h in history)
            for h in history:
                history_keys.update(h)
            # Loss values are deliberately never used or copied.
            attempted = sum(h['optimizer_steps'] for h in history)
            skipped = sum(h['scaler_skipped_steps'] for h in history)
            assert attempted == unit['optimizer_updates_attempted']
            assert skipped == unit['amp_updates_skipped_by_scale_decrease']
            start, finish = timestamp(starts[uid]['time']), timestamp(dones[uid]['time'])
            wall = finish - start
            assert wall > receipt['fit_seconds'] > 0
            row = {k: unit[k] for k in ('unit_id', 'phase', 'config_index', 'draw', 'fold',
                                        'fit_seconds', 'checkpoint_unique_states', 'payload_bytes',
                                        'peak_cuda_allocated_bytes', 'environment_id')}
            row.update(started_at=starts[uid]['time'], done_at=dones[uid]['time'],
                       unit_ledger_wall_seconds=wall,
                       ledger_wall_minus_fit_seconds=wall - receipt['fit_seconds'],
                       optimizer_attempts=attempted, amp_skipped=skipped,
                       epoch_optimizer_attempts=[h['optimizer_steps'] for h in history],
                       epoch_amp_skips=[h['scaler_skipped_steps'] for h in history])
            assert receipt['peak_cuda_bytes'] == row['peak_cuda_allocated_bytes']
            for filename, size in unit['payload_bytes'].items():
                artifact = directory / ('DONE' if filename == 'DONE' else 'attempts/0001/' + filename)
                assert artifact.stat().st_size == size
            rows.append(row)
        all_rows += rows
        phase_results[phase] = aggregate(rows)
        phase_results[phase].update(
            first_start=min(x['started_at'] for x in rows), last_done=max(x['done_at'] for x in rows),
            event_envelope_seconds=max(timestamp(x['done_at']) for x in rows) - min(timestamp(x['started_at']) for x in rows),
            ledger_events=dict(Counter(e['event'] for e in events)))
    formal = [u for u in all_rows if u['phase'] == 'formal']
    configs = {str(i): aggregate([u for u in formal if u['config_index'] == i]) for i in range(4)}
    cfg3 = configs['3']
    config3 = next(u['config'] for u in plan['units'] if u['config_index'] == 3)

    batches = []
    for i in range(60):
        marker = js(dual / f'ops/batch_{i:02}_backed_up.json')
        durable = js(Path(marker['backup_receipt']))
        assert marker == durable and marker['pass_local'] and marker['cloud_payload_absent']
        assert marker['plan_sha256'] == plan['plan_sha256']
        assert marker['unit_ids'] == [u['unit_id'] for u in plan['units'][8*i:8*i+8]]
        batches.append(dict(batch=i, **{k: marker[k] for k in ('transfer_seconds', 'transfer_bytes',
                       'network_MiBps', 'transfer_stream_count', 'verified_bytes', 'verified_file_count')}))
    transfer_seconds = sum(x['transfer_seconds'] for x in batches)
    transfer_bytes = sum(x['transfer_bytes'] for x in batches)
    restore = js(dual / 'ops/restore_first_summary.json')
    assert restore['pass'] and restore['units_checked'] == 4
    start, stop = js(dual / 'ops/start.json'), js(dual / 'ops/stop.json')
    assert start['pod_id'] == stop['pod_id'] and stop['status'] == 'EXITED'
    rental = dict(started_at=start['started_at'], stopped_at=stop['confirmed_at'],
                  seconds=timestamp(stop['confirmed_at'])-timestamp(start['started_at']))
    active = js(dual / 'ops/latest_progress.json')
    # An isolated nvidia-smi snapshot is not a maximum and cannot be used as one.
    coverage = js(args.repo / 'v3/speaker_coverage/reports/data/resource_audit.json')
    assert coverage['pass']
    coverage_rental = js(args.coverage_root / 'ops/rental_final.json')
    coverage_restore = js(args.coverage_root / 'ops/restore9_summary.json')
    coverage_ft = [u for u in coverage['phases']['formal']['per_successful_unit'] if u['model'] == 'wavlm_ft']
    assert len(coverage_ft) == 180
    one_epoch = [u['payload_bytes']['checkpoint.pt'] for u in formal if u['checkpoint_unique_states'] == 1]
    source_lines = {}
    for name in ('run.py', 'engine.py'):
        path = args.repo / 'v3/inner_validation' / name
        read(path)
        source_lines[name] = str(path.resolve())
    read(Path(__file__))
    report = dict(
        schema='ser-historical-resource-review-1', pass_metadata_checks=True,
        plan_sha256=plan['plan_sha256'], scientific_scores_read_or_computed=False,
        large_tensor_or_audio_bytes_read=False, large_artifacts_stat_only=True,
        input_sha256=pins, quantile_definition='linear: sorted[(n-1)*q], interpolate adjacent entries',
        dual_phases=phase_results, dual_by_config=configs, config3=config3,
        history_field_names=sorted(history_keys),
        missing_measurements=dict(epoch_seconds=None, train_only_seconds=None, validation_seconds=None,
                                  per_fit_checkpoint_save_reload_seconds=None, cpu_model=None,
                                  allocated_cpu_cores=None, peak_process_ram_bytes=None,
                                  total_host_ram_bytes=None, gpu_busy_seconds=None),
        software_environment_sets=source['phases']['formal']['environment_sets'],
        definitions=source['definitions'],
        ledger_wall_definition='unit_done ISO timestamp minus unit_start; includes per-unit wrapping/sealing but not initial shared gate or gaps between units; wall clock, not monotonic measurement',
        history_timing_definition='No epoch/train/evaluation/save timing fields were recorded. Epoch loss/step records cannot recover those durations.',
        transfer=dict(batches=batches, seconds=transfer_seconds, bytes=transfer_bytes,
                      weighted_MiBps=transfer_bytes/transfer_seconds/2**20,
                      batch_seconds=stats(x['transfer_seconds'] for x in batches),
                      streams=sorted({x['transfer_stream_count'] for x in batches}),
                      scope='four parallel SSH tar streams per completed 8-fit batch; transfer totals overlap training, not additive rental hours; excludes initial audio/model/environment uploads'),
        independent_restore={k: restore[k] for k in ('units_checked','checkpoint_role_comparisons',
            'total_wall_seconds','forward_seconds_total','waveform_prepare_seconds')},
        rental=rental,
        isolated_gpu_snapshot=dict(observed_at_unix=active['observed_at'], raw=active['gpu_util_memMiB_totalMiB'],
            columns='utilization_percent, memory_used_MiB, memory_total_MiB', is_peak=False),
        coverage_crosscheck=dict(
            scope='Different prior speaker-coverage experiment; descriptive support only, not an ablation of dual validation',
            formal_ft_count=180, fit_seconds=stats(u['fit_seconds'] for u in coverage_ft),
            unit_wall_seconds=stats(u['unit_wall_seconds'] for u in coverage_ft),
            peak_cuda_allocated_bytes=stats(u['peak_cuda_allocated_bytes'] for u in coverage_ft),
            rental={k: coverage_rental[k] for k in ('started_at_utc','stop_confirmed_at_utc','elapsed_seconds_to_stop_confirmation')},
            restore={k: coverage_restore[k] for k in ('units_checked','total_wall_seconds','inference_seconds_total','waveform_prepare_seconds')}),
        tentative_240_fit_scenario=dict(
            frozen_or_executed=False, fits=240, pairs=120,
            assumptions='same RTX5090 observed environment, 576 fit clips, batch16, 15 epochs, top4+head; future fixed cfg3/last-only implementation not benchmarked',
            old_full_pipeline_fit_hours_by_mean=cfg3['fit_seconds']['mean']*240/3600,
            old_full_pipeline_fit_hours_by_p90=cfg3['fit_seconds']['p90']*240/3600,
            old_full_pipeline_fit_hours_by_observed_max=cfg3['fit_seconds']['max']*240/3600,
            last_only_measured_seconds=None,
            one_epoch_checkpoint_bytes_observed=stats(one_epoch),
            one_epoch_240_checkpoint_bytes_proxy=statistics.median(one_epoch)*240,
            old_cfg3_240_checkpoint_bytes_proxy=cfg3['checkpoint_bytes']/120*240,
            caveat='Storage proxy assumes unchanged FP32 epoch-delta architecture/serialization; new metadata/class head/optimizer saving can alter it. Runtime projections are historical scaling scenarios, not promised rental duration or a price quote.'),
        per_unit_resource_rows=all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'out':str(args.out), 'formal':len(formal), 'cfg3':cfg3['fit_seconds'],
                      'transfer_seconds':transfer_seconds,'rental_seconds':rental['seconds']}, ensure_ascii=False))


def aggregate(rows):
    return dict(units=len(rows), fit_seconds=stats(u['fit_seconds'] for u in rows),
                unit_ledger_wall_seconds=stats(u['unit_ledger_wall_seconds'] for u in rows),
                ledger_wall_minus_fit_seconds=stats(u['ledger_wall_minus_fit_seconds'] for u in rows),
                peak_cuda_allocated_bytes=stats(u['peak_cuda_allocated_bytes'] for u in rows),
                checkpoint_bytes=sum(u['payload_bytes']['checkpoint.pt'] for u in rows),
                payload_bytes=sum(sum(u['payload_bytes'].values()) for u in rows),
                checkpoint_unique_states_histogram=dict(Counter(u['checkpoint_unique_states'] for u in rows)),
                optimizer_attempts=sum(u['optimizer_attempts'] for u in rows),
                amp_skips=sum(u['amp_skipped'] for u in rows))


if __name__ == '__main__':
    main()
