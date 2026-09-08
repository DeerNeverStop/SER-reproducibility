"""Separate-process disk checkpoint replay; no fitting or outer science scores.

Run this CLI in a fresh Python process after the requested units have DONE.
--run-dir is a single phase directory containing SOURCE_LOCK.json and
plan_snapshot.json. Repeat --unit-id to restore multiple native-label units.
--roots is the existing JSON mapping corpus names to local audio directories.

The frozen plan/commitment checks and the model/preprocessing/prediction code
are shared with the original implementation. Actual disk restoration and logits
comparison are performed here; receipt assertions do not substitute for replay.
This proves replay for requested saved states, not independent model training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time


SCHEMA = 'ser-final-program-independent-checkpoint-replay-1'
GROUPS = ('A', 'B', 'outer')
ATOL = 1e-5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def output_path(path, run_dir, audio_roots):
    path = Path(path).resolve()
    require(path.suffix.lower() == '.json', 'audit output must be a JSON file')
    require(not path.exists(), 'audit output already exists; use a fresh name')
    require(not path.is_relative_to((run_dir / 'units').resolve()),
            'audit output must be outside every phase unit directory')
    require(not any(path.is_relative_to(root) for root in audio_roots.values()),
            'audit output must not be written inside an audio root')
    return path


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation cannot replace an earlier successful report. A crashed
    # write is not reusable: the next run needs a different, fresh output path.
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def selected_units(plan, lock, requested):
    require(requested and len(requested) == len(set(requested)), 'unit IDs empty or duplicated')
    require(all(isinstance(uid, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', uid)
                and uid not in ('.', '..') for uid in requested), 'unsafe unit ID')
    units = [u for u in plan['units'] if u['phase'] == lock['phase']]
    lookup = {u['unit_id']: u for u in units}
    require(len(lookup) == len(units), 'duplicate phase unit in plan')
    require(set(requested) <= set(lock['phase_units']) & set(lookup), 'unit not in frozen phase')
    return [lookup[uid] for uid in requested]


def imported_modules(repo):
    sys.path.insert(0, str(repo))
    modules = {}
    for short in ('run', 'engine', 'plan'):
        name = 'v3.final_program_20260907.' + short
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == (repo / 'v3/final_program_20260907' / (short + '.py')).resolve(),
                'loaded module is outside requested repository: ' + short)
        modules[short] = module
    modules['run'].check_import_locations(repo)
    # Legacy deployment imports ser_v2 through repo/v2 rather than v2.ser_v2.
    # The frozen runner's v2/v3 namespace check does not cover that alias.
    for name, module in list(sys.modules.items()):
        if name == 'ser_v2' or name.startswith('ser_v2.'):
            expected = (repo / 'v2' / Path(*name.split('.'))).resolve()
            location = getattr(module, '__file__', None)
            if location is None:
                locations = list(getattr(module, '__path__', []))
                require(locations and all(Path(path).resolve() == expected for path in locations),
                        'legacy namespace outside requested repository: ' + name)
            else:
                require(Path(location).resolve() in (expected.with_suffix('.py'), expected / '__init__.py'),
                        'legacy module outside requested repository: ' + name)
    return modules


def audio_inventory(plan, units, roots):
    pinned, inventory = {}, {}
    for corpus in sorted({unit['corpus'] for unit in units}):
        require(corpus in roots, 'audio root missing for corpus: ' + corpus)
        root = roots[corpus]
        require(root.is_dir(), 'audio root is not a directory: ' + corpus)
        paths = sorted({p for unit in units if unit['corpus'] == corpus
                        for group in GROUPS for p in unit['report'][group]})
        listed = {}
        for rel in paths:
            path = (root / rel).resolve()
            row = plan['rows'][corpus][rel]
            require(path.is_relative_to(root) and path.is_file(), 'report audio path escapes or is missing')
            size, sha = path.stat().st_size, file_sha(path)
            require(size == int(row['bytes']) and sha == row['sha256'],
                    'report audio byte identity differs: ' + corpus + '/' + rel)
            pinned[str(path)] = sha
            listed[rel] = {'sha256': sha, 'bytes': size}
        inventory[corpus] = {'root': str(root), 'files': listed,
                             'count': len(listed), 'inventory_sha256': digest(listed)}
    return pinned, inventory


def compare_logits(expected, actual, classes):
    import numpy as np
    actual = np.asarray(actual)
    require(expected.shape == actual.shape and expected.ndim == 2 and expected.shape[1] == classes
            and np.isfinite(expected).all() and np.isfinite(actual).all(), 'invalid replay logits')
    difference = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)), initial=0.))
    return {'pass': difference <= ATOL, 'max_abs_diff': difference, 'atol': ATOL, 'rtol': 0.,
            'rows': len(expected), 'classes': classes, 'logit_values': int(expected.size)}


def environment(torch, np, device):
    return {'python': platform.python_version(), 'executable': sys.executable,
            'numpy': np.__version__, 'torch': str(torch.__version__),
            'torchaudio': importlib.metadata.version('torchaudio'), 'cuda': torch.version.cuda,
            'device_type': device.type,
            'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


def environment_comparison(original, actual, device_type):
    keys = ('torch', 'torchaudio', 'cuda', 'device', 'threads')
    result = {key: {'original': original.get(key), 'replay': actual.get(key),
                    'equal': original.get(key) == actual.get(key)} for key in keys}
    if device_type == 'cuda':
        require(all(value['equal'] for value in result.values()),
                'CUDA replay differs from the recorded execution environment')
    return result


def execute(args):
    started_at, began = now(), time.perf_counter()
    repo, run_dir = Path(args.repo).resolve(), Path(args.run_dir).resolve()
    model_path, roots_path = Path(args.model_path).resolve(), Path(args.roots).resolve()
    require(repo.is_dir() and run_dir.is_dir(), 'repository or run directory missing')
    roots_doc = read_json(roots_path)
    require(isinstance(roots_doc, dict) and all(isinstance(c, str) and isinstance(p, str)
                                               for c, p in roots_doc.items()), 'invalid roots JSON')
    roots = {corpus: Path(root).resolve() for corpus, root in roots_doc.items()}
    out = output_path(args.out_json, run_dir, roots)
    source_sha = file_sha(__file__)
    # The model builder uses only a local state_dict. These settings also prevent
    # incidental library fallback downloads during this standalone invocation.
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    modules = imported_modules(repo)
    runner, engine = modules['run'], modules['engine']
    plan, lock = runner.checked_plan(repo, run_dir)
    units = selected_units(plan, lock, args.unit_id)
    require(file_sha(model_path) == lock['model_sha256'], 'base model bytes differ from frozen lock')
    pinned = {str(run_dir / 'SOURCE_LOCK.json'): file_sha(run_dir / 'SOURCE_LOCK.json'),
              str(run_dir / 'plan_snapshot.json'): file_sha(run_dir / 'plan_snapshot.json'),
              str(model_path): lock['model_sha256'], str(roots_path): file_sha(roots_path)}
    audio_pins, audio = audio_inventory(plan, units, roots)
    pinned.update(audio_pins)
    import numpy as np
    import torch
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda'), 'replay device must be cpu or cuda')
    require(device.type != 'cuda' or torch.cuda.is_available(), 'requested CUDA is unavailable')
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    def forbidden(*unused, **kwargs):
        raise RuntimeError('training or model downloading is prohibited in checkpoint replay')
    # These are process-local guards, not edits to frozen files.
    engine.fit_unit = forbidden
    torch.hub.download_url_to_file = forbidden
    prepared = []
    for unit in units:
        rows = plan['rows'][unit['corpus']]
        folder = (run_dir / 'units' / unit['unit_id']).resolve()
        require(folder.is_relative_to((run_dir / 'units').resolve()), 'unit path escapes run directory')
        done_bytes = (folder / 'DONE').read_bytes()
        done_hash = hashlib.sha256(done_bytes).hexdigest()
        done, receipt = runner.verify_done(unit, run_dir, lock, rows, weights=True,
                                            done_doc=json.loads(done_bytes))
        require(file_sha(folder / 'DONE') == done_hash, 'DONE changed during commitment verification')
        pins = {str(folder / 'DONE'): done_hash}
        pins.update({str((folder / rel).resolve()): sha for rel, sha in done['artifacts'].items()})
        pinned.update(pins)
        prepared.append((unit, folder / done['attempt'], receipt, pins))
    reports, caches = [], {}
    for unit, attempt, receipt, artifact_pins in prepared:
        unit_began = time.perf_counter()
        classes, corpus = unit['n_classes'], unit['corpus']
        saved = torch.load(attempt / 'checkpoint.pt', map_location='cpu', weights_only=True)
        model, initial_sha = engine.build_model(model_path, unit['seeds']['initialization'], classes)
        frozen_sha = engine.frozen_hash(model)
        require(initial_sha == saved['initial_state_sha256'] and frozen_sha == saved['frozen_parameter_sha256'],
                'reconstructed initial/frozen state differs from checkpoint')
        names, buffers = engine.delta_names(model)
        require(names == saved['trainable_parameter_names'] and buffers == saved['buffer_names'],
                'checkpoint trainable/buffer names differ from actual model')
        actual_environment = environment(torch, np, device)
        env_comparison = environment_comparison(receipt['environment'], actual_environment, device.type)
        model.to(device).eval()
        if corpus not in caches:
            caches[corpus] = engine.WaveCache(roots[corpus], 16000)
        cache = caches[corpus]
        comparisons = {}
        with np.load(attempt / 'predictions.npz', allow_pickle=False) as original:
            stored_epochs = original['epochs'].tolist()
            require(stored_epochs == unit['prediction_epochs'], 'NPZ epoch index changed')
            for epoch in sorted({int(value) for value in saved['selected_epochs'].values()}):
                engine.load_delta(model, saved['epoch_states'][str(epoch)], frozen_sha)
                index = stored_epochs.index(epoch)
                names_for_epoch = sorted(name for name, value in saved['selected_epochs'].items() if value == epoch)
                before = engine.rng_state(device)
                for group in GROUPS:
                    expected = original[group + '__all_epoch_logits'][index]
                    actual = engine.predict_group(model, unit['report_batches'][group], cache,
                                                  int(unit['config']['eval_cap_seconds'] * 16000), classes, device)
                    comparison = compare_logits(expected, actual, classes)
                    comparison['checkpoint_names'] = names_for_epoch
                    comparisons[f'{epoch}/{group}'] = comparison
                engine.check_rng(before, device)
        expected_coverage = {f'{epoch}/{group}' for epoch in saved['selected_epochs'].values() for group in GROUPS}
        require(set(comparisons) == expected_coverage, 'missing saved-epoch/group replay')
        reports.append({'unit_id': unit['unit_id'], 'corpus': corpus, 'arm': unit['arm'],
                        'phase': unit['phase'], 'n_classes': classes,
                        'pass': all(value['pass'] for value in comparisons.values()),
                        'selected_epochs': saved['selected_epochs'], 'comparisons': comparisons,
                        'max_abs_diff': max(value['max_abs_diff'] for value in comparisons.values()),
                        'initial_state_sha256': initial_sha, 'frozen_parameter_sha256': frozen_sha,
                        'base_state_sha256': saved['base_state_sha256'],
                        'committed_file_sha256': artifact_pins, 'environment': actual_environment,
                        'recorded_environment_comparison': env_comparison,
                        'wall_seconds': time.perf_counter() - unit_began,
                        'scientific_scores_computed': False})
        print(json.dumps({'unit_id': unit['unit_id'], 'restored': reports[-1]['pass'],
                          'saved_epoch_group_comparisons': len(comparisons),
                          'max_abs_diff': reports[-1]['max_abs_diff']}), flush=True)
        del model, saved, expected, actual
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for path, sha in pinned.items():
        require(file_sha(path) == sha, 'input or committed artifact changed during replay: ' + path)
    final_plan, final_lock = runner.checked_plan(repo, run_dir)
    require(final_plan['plan_sha256'] == plan['plan_sha256'] and final_lock['lock_sha256'] == lock['lock_sha256'],
            'phase identity changed during replay')
    require(file_sha(__file__) == source_sha, 'replay source changed during process')
    result = {'schema': SCHEMA, 'pass': all(report['pass'] for report in reports),
              'started_at': started_at, 'completed_at': now(), 'pid': os.getpid(),
              'invocation': sys.argv, 'repo': str(repo), 'run_dir': str(run_dir),
              'phase': lock['phase'], 'plan_sha256': plan['plan_sha256'],
              'lock_sha256': lock['lock_sha256'], 'source_commit': lock['source_commit'],
              'frozen_source_sha256': lock['sources'], 'replay_source_sha256': source_sha,
              'input_and_artifact_sha256': pinned, 'report_audio_inventory': audio,
              'requested_unit_ids': args.unit_id, 'units_checked': len(reports), 'reports': reports,
              'saved_epoch_group_comparisons': sum(len(r['comparisons']) for r in reports),
              'checkpoint_name_group_comparisons': sum(3 * len(r['selected_epochs']) for r in reports),
              'max_abs_diff': max(r['max_abs_diff'] for r in reports),
              'atol': ATOL, 'rtol': 0., 'wall_seconds': time.perf_counter() - began,
              'scientific_scores_computed': False, 'new_fits': 0,
              'scope': 'Actual disk reconstruction of every saved winner/last for requested DONE units, '
                       'using shared frozen architecture, preprocessing, batching and prediction implementation. '
                       'Report audio bytes and committed artifacts rehashed; fit audio unnecessary for replay. '
                       'Validation values may be recomputed by the shared commitment checker to verify winners; '
                       'no outer CE/UAR/accuracy or scientific hypothesis test is computed. '
                       'This selected-unit audit is not a full-phase gate or independent retraining.'}
    result['audit_sha256'] = digest(result)
    write_once(out, result)
    print(json.dumps({'pass': result['pass'], 'units_checked': len(reports),
                      'max_abs_diff': result['max_abs_diff'], 'out_json': str(out)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'run-dir', 'roots', 'model-path', 'out-json'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--unit-id', action='append', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        return 0 if execute(args)['pass'] else 1
    except Exception as error:
        print(json.dumps({'pass': False, 'error_type': type(error).__name__, 'error': str(error),
                          'scientific_scores_computed': False}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
