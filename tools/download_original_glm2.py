"""Download a pinned public gLM2 checkpoint with verified HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import time
import requests

REPOSITORY = 'tattabio/gLM2_650M_UR50'
REVISION = 'a5937fbb770bd8bae65e3034aacd823b041dcba3'
DIRECTORY = Path('artifacts/glm2_650m_ur50')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    r = requests.get(f'https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}',
                     params={'blobs': 'true'}, timeout=30)
    r.raise_for_status()
    metadata = r.json()
    info = next(x for x in metadata['siblings'] if x['rfilename'] == 'model.safetensors')
    expected, size = info['lfs']['sha256'], info['size']
    path = DIRECTORY/'model.safetensors'
    if path.exists():
        raise FileExistsError(path)
    temporary = DIRECTORY/'model.safetensors.ranges-part'
    descriptor = os.open(temporary, os.O_CREAT | os.O_RDWR | (0 if args.resume else os.O_TRUNC), 0o644)
    if args.resume and os.fstat(descriptor).st_size != size:
        raise ValueError('Resume size differs from pinned checkpoint')
    os.ftruncate(descriptor, size)
    block = 2 << 20
    starts=list(range(0,size,block))
    if args.resume:
        holes=[];position=0
        while position<size:
            try:
                data=os.lseek(descriptor,position,os.SEEK_DATA)
            except OSError as error:
                if error.errno != errno.ENXIO: raise
                holes.append((position,size));break
            if data>position: holes.append((position,data))
            position=os.lseek(descriptor,data,os.SEEK_HOLE)
        starts=[x for x in starts if any(x<b and min(x+block,size)>a for a,b in holes)]
    started = time.monotonic()

    def fetch(start):
        end = min(start+block, size)-1
        for attempt in range(4):
            try:
                response = requests.get(
                    f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.safetensors',
                    params={'chunk': str(start),'attempt':str(time.time_ns())},
                    headers={'Range': f'bytes={start}-{end}'}, timeout=(20,60))
                response.raise_for_status()
                if (response.status_code != 206 or
                    response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}' or
                    len(response.content) != end-start+1):
                    raise ValueError('Server returned the wrong byte range')
                count = os.pwrite(descriptor, response.content, start)
                if count != len(response.content):
                    raise IOError('Incomplete disk write')
                return count
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    total = size-sum(min(block,size-x) for x in starts)
    print('Missing ranges',len(starts),'bytes',size-total,flush=True)
    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            for future in as_completed([pool.submit(fetch, i) for i in starts]):
                total += future.result()
                if total % (128<<20) == 0 or total == size:
                    print(total, '/', size, 'bytes;', round(time.monotonic()-started,1), 'seconds', flush=True)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256()
    with temporary.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8<<20), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('Downloaded weight SHA256 does not match Hugging Face LFS')
    temporary.rename(path)
    files = {}
    for item in DIRECTORY.iterdir():
        if item.is_file() and item.suffix in ('.json','.py','.safetensors'):
            h = hashlib.sha256()
            with item.open('rb') as handle:
                for chunk in iter(lambda:handle.read(8<<20), b''):
                    h.update(chunk)
            files[item.name] = h.hexdigest()
    (DIRECTORY/'manifest.json').write_text(json.dumps(dict(repository=REPOSITORY,revision=REVISION,
        weights_sha256=expected,size=size,files=files),indent=2)+'\n')
    print('VERIFIED', expected, flush=True)


if __name__ == '__main__':
    main()
