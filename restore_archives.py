"""Restore this Git data snapshot into a new directory, verifying every byte."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parent
index = json.loads((BASE / 'archive-index.json').read_text())
if len(sys.argv) != 2:
    raise SystemExit('Usage: python3 restore_archives.py NEW_OUTPUT_DIRECTORY')
target = Path(sys.argv[1]).resolve()
target.mkdir(parents=True, exist_ok=False)

for record in index['archives']:
    name = record['file']
    if Path(name).name != name or not name:
        raise ValueError('Invalid archive filename')
    complete_hash = hashlib.sha256()
    size = 0
    with (target / name).open('xb') as dest:
        for part in record['parts']:
            source = (BASE / part['path']).resolve()
            if not source.is_relative_to(BASE) or not source.is_file():
                raise ValueError('Invalid archive part')
            part_hash = hashlib.sha256()
            part_size = 0
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    part_hash.update(block)
                    complete_hash.update(block)
                    part_size += len(block)
                    size += len(block)
                    dest.write(block)
            if part_size != part['bytes'] or part_hash.hexdigest() != part['sha256']:
                raise ValueError('Archive part integrity failure: ' + part['path'])
    if size != record['bytes'] or complete_hash.hexdigest() != record['sha256']:
        raise ValueError('Whole archive integrity failure: ' + name)
    print(name, size, 'SHA-256 verified', flush=True)

for record in index['metadata']:
    name = record['file']
    if Path(name).name != name:
        raise ValueError('Invalid metadata filename')
    source = BASE / name
    data = source.read_bytes()
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Metadata integrity failure: ' + name)
    with (target / name).open('xb') as dest:
        dest.write(data)
print('All archives and metadata restored into', target)
