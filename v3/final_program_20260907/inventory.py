"""Read-only full-corpus byte/decode audit before planning new training.

No scores, features, model inference, or GPU use. Retain exact prior hygiene.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from v2.ser_v2.corpora import apply_hygiene, parse_filename

ROOTS = {
    'cremad': 'E:/科研/SER/AudioWAV',
    'ravdess': 'E:/科研/SER/data',
    'subesco': 'E:/claudework_data/ICASSP2027-corpora/subesco/extracted',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_one(args):
    corpus, root, row, decode = args
    path = (root / row['relative_path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('path leaves corpus root')
    if path.stat().st_size != int(row['bytes']) or sha(path) != row['sha256']:
        raise ValueError(f'byte identity changed: {corpus}/{row["relative_path"]}')
    parsed = parse_filename(corpus, row['relative_path'])
    if parsed is None or any(str(parsed[k]) != str(row[k]) for k in
                             ('speaker', 'label', 'sentence', 'take', 'intensity')):
        raise ValueError(f'filename/manifest mismatch: {path.name}')
    result = {'corpus': corpus, 'path': row['relative_path'], 'bytes': int(row['bytes']),
              'sha256': row['sha256'], 'decoded': decode}
    if decode:
        waveform, sr = sf.read(path, dtype='float32', always_2d=True)
        if len(waveform) == 0 or sr <= 0 or not np.isfinite(waveform).all():
            raise ValueError(f'invalid decoded audio: {path}')
        result.update(sample_rate=int(sr), channels=int(waveform.shape[1]),
                      frames=len(waveform), seconds=len(waveform) / sr,
                      peak_abs=float(np.max(np.abs(waveform))))
    return result


def run(repo, out, workers):
    if out.exists():
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True)
    began = time.perf_counter()
    result = dict(schema='ser-final-program-audio-inventory-1',
                  created_at=datetime.now(timezone.utc).isoformat(),
                  scope='Full manifest bytes and all retained audio decoded; no new model outcomes',
                  script_sha256=sha(Path(__file__)),
                  hygiene_source_sha256=sha(repo / 'v2/ser_v2/corpora.py'),
                  no_gpu=True, corpora={})
    for corpus, root_string in ROOTS.items():
        root = Path(root_string)
        manifest = repo / f'v2/manifests/{corpus}_manifest.csv'
        original_sha = sha(manifest)
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len({r['relative_path'] for r in rows}) != len(rows):
            raise ValueError('duplicated manifest path')
        clean, hygiene = apply_hygiene(corpus, rows)
        kept = {r['relative_path'] for r in clean}
        inputs = [(corpus, root, r, r['relative_path'] in kept) for r in rows]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            checks = list(pool.map(verify_one, inputs))
        if sha(manifest) != original_sha:
            raise ValueError('manifest changed during audit')
        decoded = [r for r in checks if r['decoded']]
        fields = ['corpus', 'path', 'bytes', 'sha256', 'decoded', 'sample_rate',
                  'channels', 'frames', 'seconds', 'peak_abs']
        with (out / f'{corpus}_audio_check.csv').open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(checks)
        result['corpora'][corpus] = dict(
            root=str(root), manifest_sha256=original_sha, raw_count=len(rows),
            retained_count=len(clean), byte_verified=len(checks), decoded=len(decoded),
            total_bytes=sum(r['bytes'] for r in checks),
            total_retained_seconds=sum(r['seconds'] for r in decoded),
            sample_rate_counts=dict(Counter(r['sample_rate'] for r in decoded)),
            channel_counts=dict(Counter(r['channels'] for r in decoded)),
            duration_quantiles=np.quantile([r['seconds'] for r in decoded], [0, .5, .9, 1]).tolist(),
            people=len({r['speaker'] for r in clean}),
            texts=len({r['sentence'] for r in clean}),
            labels=sorted({r['label'] for r in clean}), hygiene=hygiene,
            table_sha256=sha(out / f'{corpus}_audio_check.csv'))
        print(json.dumps({'corpus': corpus, 'bytes_passed': len(checks),
                          'decoded_passed': len(decoded)}, ensure_ascii=False), flush=True)
    result['wall_seconds'] = time.perf_counter() - began
    result['pass'] = True
    (out / 'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    print(json.dumps({'pass': True, 'wall_seconds': result['wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    run(a.repo.resolve(), a.out.resolve(), a.workers)
