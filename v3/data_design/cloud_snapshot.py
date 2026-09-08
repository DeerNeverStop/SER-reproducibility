"""Back up completed Study II artifacts without interpreting prediction arrays."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import tarfile
from datetime import datetime, timezone

PLAN_SHA = 'f258cabe582d97b3a386a666b566730d00241a5ded0e14ad61d590cb3c861503'
UID = re.compile(r'[0-9a-f]{64}\Z')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def capture(run, ops):
    run, ops = Path(run).resolve(), Path(ops).resolve()
    ledger = (run / 'ledger.jsonl').read_bytes()
    if not ledger.endswith(b'\n'):
        raise ValueError('Incomplete concurrent ledger line; retry snapshot')
    events = [json.loads(line) for line in ledger.splitlines() if line]
    completed = {e['unit_id']: e for e in events if e.get('event') == 'unit_done'}
    files = {'core/ledger.jsonl': ledger}
    for name in ('run_identity.json', 'environment_cnn.json'):
        files['core/' + name] = (run / name).read_bytes()
    identity = json.loads(files['core/run_identity.json'])
    if identity.get('plan_sha256') != PLAN_SHA:
        raise ValueError('Run plan identity mismatch')
    environment = json.loads(files['core/environment_cnn.json'])
    lock = run / '.core-run.lock'
    if lock.exists():
        files['core/.core-run.lock'] = lock.read_bytes()
    for uid, event in completed.items():
        if not UID.fullmatch(uid):
            raise ValueError('Invalid unit ID')
        folder = run / 'units' / uid
        payloads = {name: (folder / name).read_bytes() for name in ('DONE', 'unit.json', 'predictions.npz')}
        done, receipt = json.loads(payloads['DONE']), json.loads(payloads['unit.json'])
        if (done.get('unit_id') != uid or receipt.get('unit_id') != uid
                or done.get('plan_sha256') != PLAN_SHA or receipt.get('plan_sha256') != PLAN_SHA
                or receipt.get('model') != 'cnn' or receipt.get('environment') != environment
                or receipt.get('attempt') != event.get('attempt')):
            raise ValueError('Unit identity/environment/attempt mismatch')
        if (digest(payloads['unit.json']) != done.get('unit_json_sha256')
                or digest(payloads['predictions.npz']) != done.get('predictions_sha256')
                or receipt.get('predictions_sha256') != done.get('predictions_sha256')):
            raise ValueError('Unit artifact hash mismatch')
        for name, value in payloads.items():
            files[f'core/units/{uid}/{name}'] = value
    for name in ('first8.log', 'first8.status', 'full.log', 'full.status', 'independent_host.json'):
        path = ops / name
        if path.is_file():
            files['ops/' + name] = path.read_bytes()
    closed = (len(completed) == 720 and 'core/.core-run.lock' not in files
              and files.get('ops/full.status', b'').strip() == b'0')
    if closed and (run / 'ledger.jsonl').read_bytes() != ledger:
        raise ValueError('Final ledger changed while snapshotting')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = {'schema': 'ser-study2-cloud-snapshot-1', 'created_at': timestamp,
              'host': socket.gethostname(), 'plan_sha256': PLAN_SHA,
              'cnn_done': len(completed), 'cnn_block_stopped_complete': closed,
              'scientific_metrics_read': False,
              'files_sha256': {name: digest(value) for name, value in files.items()}}
    files['snapshot.json'] = (json.dumps(report, sort_keys=True, indent=2) + '\n').encode()
    destination = ops / 'backups'
    destination.mkdir(exist_ok=True)
    archive = destination / f'cnn_{timestamp}.tar.gz'
    with archive.open('xb') as handle:
        with tarfile.open(fileobj=handle, mode='w:gz') as tar:
            for name, value in files.items():
                item = tarfile.TarInfo(name)
                item.size = len(value)
                item.mode = 0o600
                tar.addfile(item, io.BytesIO(value))
        handle.flush()
        os.fsync(handle.fileno())
    return {'archive': str(archive), 'sha256': digest(archive.read_bytes()),
            'bytes': archive.stat().st_size, 'cnn_done': len(completed),
            'cnn_block_stopped_complete': closed, 'scientific_metrics_read': False}


def restore(archive, expected_sha, destination):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    payload = archive.read_bytes()
    if digest(payload) != expected_sha:
        raise ValueError('Transfer SHA mismatch')
    if destination.exists():
        raise ValueError('Snapshot destination already exists; never overwrite')
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or 'snapshot.json' not in names:
            raise ValueError('Duplicate members or missing snapshot manifest')
        content = {}
        for item in members:
            path = PurePosixPath(item.name)
            if (not item.isfile() or path.is_absolute() or '..' in path.parts
                    or '\\' in item.name or ':' in item.name
                    or (item.name != 'snapshot.json' and path.parts[0] not in ('core', 'ops'))):
                raise ValueError('Unsafe archive member')
            content[item.name] = tar.extractfile(item).read()
        manifest = json.loads(content.pop('snapshot.json'))
        if (manifest.get('schema') != 'ser-study2-cloud-snapshot-1'
                or manifest.get('plan_sha256') != PLAN_SHA
                or manifest.get('files_sha256') != {n: digest(v) for n, v in content.items()}):
            raise ValueError('Snapshot manifest/file SHA mismatch')
        content['snapshot.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        destination.mkdir(parents=True, exist_ok=False)
        for name, value in content.items():
            target = destination.joinpath(*PurePosixPath(name).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Unsafe extraction target')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            if digest(target.read_bytes()) != digest(value):
                raise ValueError('Restored file SHA mismatch')
    return {'pass': True, 'destination': str(destination), 'archive_sha256': expected_sha,
            'files_verified': len(content)-1, 'cnn_done': manifest['cnn_done'],
            'cnn_block_stopped_complete': manifest['cnn_block_stopped_complete'], 'scientific_metrics_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    remote = sub.add_parser('capture')
    remote.add_argument('--run', required=True)
    remote.add_argument('--ops', required=True)
    local = sub.add_parser('restore')
    local.add_argument('--archive', required=True)
    local.add_argument('--sha256', required=True)
    local.add_argument('--destination', required=True)
    args = parser.parse_args()
    report = capture(args.run, args.ops) if args.command == 'capture' else restore(args.archive, args.sha256, args.destination)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
