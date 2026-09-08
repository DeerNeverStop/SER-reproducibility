"""Export or accept a relocated, checkpoint-free 384-unit numerical replay ZIP.

The original formal full-weight gate and main scores must already exist. The
exporter runs the independent numerical audit and the frozen ledger contract,
then copies an exact allowlist. The receiver preserves every sealed byte and
re-runs numerical replay under the explicitly recorded relocation exception.
This is not training reproduction, checkpoint verification, public registration,
or a digital signature. Keep the original full archive and an external ZIP SHA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from types import SimpleNamespace
import zipfile

from . import independent_numeric_audit as numeric


SCHEMA = 'ser-final-program-portable-bundle-1'
TOOLS = ('independent_numeric_audit.py', 'portable_bundle.py')
TOP = ('SOURCE_LOCK.json', 'plan_snapshot.json', 'COMPLETE_GATE.json', 'ledger.jsonl')
SCORES = tuple(name+'.csv' for name in numeric.TABLE_COUNTS) + ('results.json', 'complete_gate.json')
MANIFEST = 'bundle_manifest.json'
AUDIT = 'audits/export_numeric_audit.json'
MAX_MEMBER = 1024**3
MAX_TOTAL = 8*1024**3
require, sha, read = numeric.require, numeric.sha, numeric.json_read


def semantic_document(value, key):
    require(isinstance(value, dict) and value.get(key) == numeric.semantic_sha(
        {k: v for k, v in value.items() if k != key}), 'semantic document SHA differs: '+key)


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and not name.startswith('/') and '\x00' not in name, 'unsafe archive path')
    parts = name.split('/')
    require(all(part not in ('', '.', '..') and re.fullmatch(r'[A-Za-z0-9_.-]+', part)
                and not part.endswith(('.', ' ')) and not re.fullmatch(
                    r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', part)
                for part in parts), 'unsafe archive path component')
    return PurePosixPath(name)


def path_at(root, relative):
    parts = safe_name(relative).parts
    result = root.joinpath(*parts)
    require(result.resolve().is_relative_to(root.resolve()), 'path escapes bundle root')
    require(not any(root.joinpath(*parts[:n]).is_symlink() for n in range(1, len(parts)+1)),
            'symlink is not a portable regular artifact')
    return result


def tool_pins(repo):
    folder = repo/'v3/final_program_20260907'
    require(Path(__file__).resolve() == (folder/'portable_bundle.py').resolve()
            and Path(numeric.__file__).resolve() == (folder/TOOLS[0]).resolve(), 'wrong audit-tool checkout')
    return {'v3/final_program_20260907/'+name: sha(folder/name) for name in TOOLS}


def ledger_and_files(repo, phase, results):
    """Reuse the frozen source and reservation validator, never its weight gate."""
    runner = numeric.checked_runner(repo)
    plan, lock = runner.checked_plan(repo, phase)
    units = [u for u in plan['units'] if u['phase'] == 'formal']
    numeric.formal_units(plan)
    gate, score_gate = read(phase/'COMPLETE_GATE.json'), read(results/'complete_gate.json')
    for item in (gate, score_gate):
        numeric.gate_identity(item, plan, lock)
    require(sha(phase/'ledger.jsonl') == gate['ledger_sha256'] == score_gate['ledger_sha256'],
            'complete ledger differs from the score gate')
    ledger = runner.audit_events(units, phase, lock, runner.ledger_rows(phase))
    for item in (gate, score_gate):
        require(ledger == item['ledger_audit'] and ledger['reservation_sha256'] == item['reservation_sha256']
                and ledger['done_sha256'] == item['done_sha256'], 'complete reservation/ledger closure differs')
    files = {'formal/'+name: phase/name for name in TOP}
    files.update({'scores/'+name: results/name for name in SCORES})
    for relative, expected_sha in ledger['reservation_sha256'].items():
        path = path_at(phase, relative)
        require(sha(path) == expected_sha, 'reservation differs from full gate')
        files['formal/'+relative] = path
    for unit in units:
        prefix = 'units/'+unit['unit_id']
        done_path = path_at(phase, prefix+'/DONE')
        done = read(done_path)
        require(sha(done_path) == gate['done_sha256'][unit['unit_id']], 'DONE differs from full gate')
        require(done['artifacts'] == gate['artifacts_sha256'][unit['unit_id']]
                == score_gate['artifacts_sha256'][unit['unit_id']], 'artifact gate identity differs')
        require(re.fullmatch(r'attempts/000[12]', done['attempt']), 'invalid committed attempt')
        expected = {done['attempt']+'/'+name for name in ('predictions.npz', 'history.json', 'receipt.json', 'checkpoint.pt')}
        require(set(done['artifacts']) == expected, 'committed artifact inventory differs')
        files['formal/'+prefix+'/DONE'] = done_path
        for name in ('predictions.npz', 'history.json', 'receipt.json'):
            relative = prefix+'/'+done['attempt']+'/'+name
            path = path_at(phase, relative)
            require(sha(path) == done['artifacts'][done['attempt']+'/'+name], 'committed small artifact differs')
            files['formal/'+relative] = path
    for name, path in files.items():
        safe_name(name)
        require(path.is_file() and not path.is_symlink(), 'missing regular payload file')
    return plan, lock, ledger, files


def audit_proof(path, results, plan, lock, pins):
    proof = read(path)
    semantic_document(proof, 'audit_sha256')
    require(proof.get('pass') is True and proof.get('formal_units') == 384
            and proof.get('mismatches') == 0 and proof.get('numeric_values_checked', 0) > 0
            and proof.get('max_abs_error', float('inf')) <= numeric.TOLERANCE,
            'successful complete independent numerical audit required')
    require(proof['plan_sha256'] == plan['plan_sha256'] and proof['lock_sha256'] == lock['lock_sha256']
            and proof['frozen_sources'] == lock['sources'] and proof['source_commit'] == lock['source_commit']
            and proof['audit_source_sha256'] == pins['v3/final_program_20260907/independent_numeric_audit.py']
            and proof['main_result_sha256'] == sha(results/'results.json')
            and proof['score_gate_sha256'] == sha(results/'complete_gate.json'), 'export audit identity differs')
    return proof


def readme():
    return '''# Checkpoint-free numerical replay bundle

This contains all 384 formal committed predictions, histories and receipts,
all DONE files, the complete ledger and attempt reservations, source lock and
plan, five score tables and score JSON/gate, and the export numerical audit.
Failed/abandoned attempts remain visible in the full ledger/reservations; their
uncommitted payloads are excluded. Four pilot trajectories are not scientific
inputs and are not included. Absolute paths in sealed metadata are preserved.

No raw audio, pretrained base or checkpoint bytes are included. Existing full
gates bind historical checkpoint hashes; this bundle does not recheck current
weights or reproduce training. Retain the original full archive separately.
The ZIP SHA should be conveyed through a trusted separate channel. A self-hash
manifest is a corruption check, not authentication or public preregistration.

Use the same source checkout and compatible Python/NumPy/SciPy/PyTorch runtime.
The frozen source checker imports the runner, but no models are loaded or run.
From the checkout, accept to a fresh directory (the audit JSON must be outside):

```text
python -B -m v3.final_program_20260907.portable_bundle extract --repo . --archive bundle.zip --out NEW_DIRECTORY --out-json NEW_AUDIT.json
```

Or, after safe extraction and manifest validation, independently replay:

```text
python -B -m v3.final_program_20260907.independent_numeric_audit --repo . --run-dir NEW_DIRECTORY/formal --results NEW_DIRECTORY/scores --out-json NEW_AUDIT.json --allow-relocated-inputs
```

Only the original absolute results.inputs.run_dir string is treated as location
metadata. All original sealed files, relative inventories, predictions, labels,
SHA commitments and numeric values remain mandatory and unmodified. No weak
manifest may override DONE, full-gate or score-input commitments.
'''


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def regular_inventory(folder):
    files = {}
    for path in folder.rglob('*'):
        require(not path.is_symlink(), 'symlink in bundle')
        if path.is_file():
            name = path.relative_to(folder).as_posix()
            safe_name(name)
            files[name] = path
        else:
            require(path.is_dir(), 'nonregular bundle entry')
    require(len({name.casefold() for name in files}) == len(files), 'case-colliding payload paths')
    return files


def inspect_bundle(repo, bundle):
    repo, bundle = Path(repo).resolve(), Path(bundle).resolve()
    manifest = read(bundle/MANIFEST)
    semantic_document(manifest, 'manifest_sha256')
    require(manifest['schema'] == SCHEMA and manifest['tool_sha256'] == tool_pins(repo), 'bundle or tool identity differs')
    files = regular_inventory(bundle)
    require(set(files) == set(manifest['files']) | {MANIFEST}, 'exact bundle file inventory differs')
    for name, record in manifest['files'].items():
        path = path_at(bundle, name)
        require(record == dict(bytes=path.stat().st_size, sha256=sha(path)), 'bundle file size/SHA differs: '+name)
    plan, lock, ledger, payload = ledger_and_files(repo, bundle/'formal', bundle/'scores')
    require(set(manifest['files']) == set(payload) | {'README.md', AUDIT},
            'bundle allowlist differs: extra weights/data or missing evidence')
    require(manifest['plan_sha256'] == plan['plan_sha256'] and manifest['lock_sha256'] == lock['lock_sha256']
            and manifest['frozen_sources'] == lock['sources'] and manifest['source_commit'] == lock['source_commit'],
            'manifest scientific source identity differs')
    audit_proof(bundle/AUDIT, bundle/'scores', plan, lock, manifest['tool_sha256'])
    return manifest, plan, lock, ledger


def export_bundle(repo, run_dir, results, out_zip):
    repo, phase, results, out_zip = (Path(p).resolve() for p in (repo, run_dir, results, out_zip))
    require(not out_zip.exists() and out_zip.suffix.lower() == '.zip'
            and not out_zip.is_relative_to(phase) and not out_zip.is_relative_to(results),
            'ZIP output must be new and outside all input directories')
    pins = tool_pins(repo)
    plan, lock, ledger, files = ledger_and_files(repo, phase, results)
    before = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in files.items()}
    with tempfile.TemporaryDirectory(prefix='ser-numeric-export-') as temporary:
        temporary = Path(temporary)
        staging = temporary/'bundle'; staging.mkdir()
        (staging/'audits').mkdir()
        audit = numeric.execute(SimpleNamespace(repo=repo, run_dir=phase, results=results,
            out_json=staging/AUDIT, allow_relocated_inputs=False))
        require(audit['pass'], 'independent numerical audit failed; no bundle exported')
        for name, source in files.items():
            target = path_at(staging, name); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(before[name] == dict(bytes=target.stat().st_size, sha256=sha(target)), 'copy differs from input snapshot')
        (staging/'README.md').write_text(readme(), encoding='utf-8', newline='\n')
        manifest = dict(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(),
            plan_sha256=plan['plan_sha256'], lock_sha256=lock['lock_sha256'], source_commit=lock['source_commit'],
            frozen_sources=lock['sources'], tool_sha256=pins,
            files={name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(regular_inventory(staging).items())},
            formal_units=384, checkpoint_bytes_included=False, raw_audio_included=False, pretrained_base_included=False,
            scope='Numerical replay only; historical full-weight gate retained; original archives still required for weight/inference replay')
        manifest['manifest_sha256'] = numeric.semantic_sha(manifest)
        write_json(staging/MANIFEST, manifest)
        inspect_bundle(repo, staging)
        archive_path = temporary/'ready.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(regular_inventory(staging).items()):
                archive.write(path, name)
        # Pin every small input both before and after audit/copy/compression.
        for name, source in files.items():
            require(before[name] == dict(bytes=source.stat().st_size, sha256=sha(source)), 'export input changed during work')
        again = ledger_and_files(repo, phase, results)
        require(again[1] == lock and again[2] == ledger and set(again[3]) == set(files)
                and tool_pins(repo) == pins, 'source or closure changed during export')
        digest = sha(archive_path)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open('rb') as source, out_zip.open('xb') as target:
            shutil.copyfileobj(source, target, 1024*1024); target.flush(); os.fsync(target.fileno())
        require(sha(out_zip) == digest, 'published ZIP differs')
    return dict(schema=SCHEMA, exported=True, archive=str(out_zip), sha256=digest,
                bytes=out_zip.stat().st_size, manifest_sha256=manifest['manifest_sha256'], files=len(manifest['files']),
                numerical_audit_pass=True, checkpoints_included=False)


def safe_extract(archive_path, target):
    """Stream regular allowlisted files; never trust ZipFile.extract paths."""
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        require(0 < len(infos) <= 5000 and len({i.filename.casefold() for i in infos}) == len(infos),
                'duplicate, colliding or excessive ZIP inventory')
        require(sum(i.file_size for i in infos) <= MAX_TOTAL, 'ZIP payload too large')
        for info in infos:
            safe_name(info.filename)
            mode = info.external_attr >> 16
            require(not info.is_dir() and not info.flag_bits & 1 and (not stat.S_IFMT(mode) or stat.S_ISREG(mode))
                    and 0 <= info.file_size <= MAX_MEMBER, 'nonregular or excessive ZIP member')
        manifest_info = next((i for i in infos if i.filename == MANIFEST), None)
        require(manifest_info is not None and manifest_info.file_size <= 16*1024**2, 'missing or excessive manifest')
        manifest = json.loads(archive.read(MANIFEST))
        semantic_document(manifest, 'manifest_sha256')
        require(manifest['schema'] == SCHEMA and set(manifest['files']) | {MANIFEST} == {i.filename for i in infos},
                'ZIP and manifest inventory differ')
        # Coarse allowlist precedes extraction, then the exact plan/gate-derived
        # inventory is enforced by inspect_bundle. No arbitrary NPZ is admitted.
        def allowed(name):
            return name in {MANIFEST, 'README.md', AUDIT} | {'formal/'+n for n in TOP} | {'scores/'+n for n in SCORES} or bool(
                re.fullmatch(r'formal/units/[A-Za-z0-9_-]+/(DONE|reservations/000[12]\.json|attempts/000[12]/(predictions\.npz|history\.json|receipt\.json))', name))
        require(all(allowed(i.filename) for i in infos), 'forbidden archive member: weights/data or unknown file')
        for info in infos:
            path = path_at(target, info.filename); path.parent.mkdir(parents=True, exist_ok=True)
            expected = manifest['files'].get(info.filename)
            if expected is not None:
                require(set(expected) == {'bytes', 'sha256'} and expected['bytes'] == info.file_size,
                        'manifest/ZIP size differs')
            hasher, count = hashlib.sha256(), 0
            with archive.open(info) as source, path.open('xb') as output:
                for block in iter(lambda: source.read(1024*1024), b''):
                    count += len(block); require(count <= info.file_size, 'ZIP expanded beyond declared size')
                    hasher.update(block); output.write(block)
            require(count == info.file_size and (expected is None or hasher.hexdigest() == expected['sha256']),
                    'extracted payload size/SHA differs')


def extract_bundle(repo, archive, out_dir, out_json):
    repo, archive, out_dir, out_json = (Path(p).resolve() for p in (repo, archive, out_dir, out_json))
    require(archive.is_file() and not out_dir.exists() and not out_json.exists() and out_json.suffix.lower() == '.json'
            and not out_json.is_relative_to(out_dir) and not archive.is_relative_to(out_dir),
            'extraction directory/audit output must be new and separate from inputs')
    archive_sha, pins = sha(archive), tool_pins(repo)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stay on the destination filesystem so the final directory publication is
    # a rename. Failed verification leaves no apparently accepted output tree.
    with tempfile.TemporaryDirectory(prefix='ser-numeric-accept-', dir=out_dir.parent) as temporary:
        staging = Path(temporary)/'bundle'; staging.mkdir()
        safe_extract(archive, staging)
        manifest, plan, lock, ledger = inspect_bundle(repo, staging)
        require(sha(archive) == archive_sha, 'ZIP changed during extraction')
        # The stable user-facing path must be the actual path used by numerical
        # replay, so publish only the structurally validated tree, then audit it.
        # If replay fails the directory is retained with no acceptance report.
        require(not out_dir.exists(), 'output appeared during extraction')
        os.rename(staging, out_dir)
        temporary_audit = Path(temporary)/'relocated_numeric_audit.json'
        audit = numeric.execute(SimpleNamespace(repo=repo, run_dir=out_dir/'formal', results=out_dir/'scores',
            out_json=temporary_audit, allow_relocated_inputs=True))
        require(audit['pass'], 'relocated numerical audit failed; extracted directory is not accepted')
        again = inspect_bundle(repo, out_dir)
        require(again[0] == manifest and again[2] == lock and again[3] == ledger
                and sha(archive) == archive_sha and tool_pins(repo) == pins, 'bundle/source changed during acceptance')
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with temporary_audit.open('rb') as source, out_json.open('xb') as target:
            shutil.copyfileobj(source, target); target.flush(); os.fsync(target.fileno())
        require(sha(out_json) == sha(temporary_audit), 'published numerical audit differs')
    return dict(accepted=True, archive_sha256=archive_sha, manifest_sha256=manifest['manifest_sha256'],
                directory=str(out_dir), audit=str(out_json), audit_sha256=audit['audit_sha256'],
                formal_units=384, numeric_values_checked=audit['numeric_values_checked'], max_abs_error=audit['max_abs_error'],
                scope='Relocated numerical verification; no checkpoint bytes or model inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    export = commands.add_parser('export')
    for name in ('repo', 'run-dir', 'results', 'out'):
        export.add_argument('--'+name, type=Path, required=True)
    extract = commands.add_parser('extract')
    for name in ('repo', 'archive', 'out', 'out-json'):
        extract.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_bundle(args.repo, args.run_dir, args.results, args.out) if args.action == 'export' else extract_bundle(
            args.repo, args.archive, args.out, args.out_json)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, error=str(error)), ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
