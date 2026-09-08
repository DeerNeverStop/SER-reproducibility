#!/usr/bin/env python3
"""Verify this public snapshot's file sizes and SHA-256; no external dependencies."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import sys


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'ser-public-snapshot-1':
        raise ValueError('unrecognized publication manifest')
    seen, count = set(), 0
    for item in manifest['files']:
        name = item['path']
        rel = PurePosixPath(name)
        if (not name or rel.is_absolute() or '..' in rel.parts or '\\' in name
                or ':' in name or name in seen or rel.parts[0] == '.git'):
            raise ValueError('unsafe or duplicate manifest member')
        seen.add(name)
        target = root.joinpath(*rel.parts)
        if not target.resolve().is_relative_to(root) or target.is_symlink() or not target.is_file():
            raise ValueError('missing or unsafe member: ' + name)
        payload = target.read_bytes()
        if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('size/SHA mismatch: ' + name)
        count += 1
    if count != manifest['file_count']:
        raise ValueError('manifest count mismatch')
    print(f'PASS: {count} published files match sizes and SHA-256. '
          'The manifest is not a digital signature or a full training audit; '
          'unlisted local files are outside its coverage.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        verify(args.repo)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('FAIL: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
