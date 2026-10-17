"""Download all release volumes, verify SHA-256, and safely extract together."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import zipfile

BASE = 'https://github.com/kang-jun-mo12/k4-whole-flow-gt/releases/download/k4-whole-flow-gt-20260922/'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--destination', type=Path, default=Path('downloaded'))
    p.add_argument('--no-extract', action='store_true')
    a = p.parse_args()
    dest = a.destination.resolve()
    cache = dest/'release_assets'
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE+'ASSET_MANIFEST.json') as r:
        manifest = json.load(r)
    for asset in manifest['assets']:
        name = asset['file']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('invalid asset name')
        target = cache/name
        if not (target.exists() and target.stat().st_size == asset['bytes'] and sha(target) == asset['sha256']):
            print('Downloading', name, flush=True)
            tmp = target.with_suffix('.download')
            with urllib.request.urlopen(BASE+name) as r, tmp.open('wb') as f:
                while data := r.read(4 << 20):
                    f.write(data)
            if tmp.stat().st_size != asset['bytes'] or sha(tmp) != asset['sha256']:
                raise ValueError('asset integrity failure: '+name)
            tmp.replace(target)
        if not a.no_extract:
            print('Extracting', name, flush=True)
            with zipfile.ZipFile(target) as z:
                for member in z.infolist():
                    resolved = (dest/member.filename).resolve()
                    if not resolved.is_relative_to(dest):
                        raise ValueError('unsafe archive path')
                z.extractall(dest)
    print('Complete:', dest/manifest['package'])


if __name__ == '__main__':
    main()
