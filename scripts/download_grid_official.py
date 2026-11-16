"""Acquire verified links from the official GRID index; video + word alignments only.

Existing archives/files are preserved. Interrupted downloads remain .part files.
No audio, synthetic samples, models, training, or backend changes are involved.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import threading
import time
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
INDEX = 'https://spandh.dcs.shef.ac.uk/gridcorpus/'
LOCK = threading.RLock()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def discover_links(html, base=INDEX):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.urls = []
        def handle_starttag(self, tag, attributes):
            if tag == 'a' and dict(attributes).get('href'):
                self.urls.append(urljoin(base, dict(attributes)['href']))
    parser = Links(); parser.feed(html)
    found = {}
    for url in parser.urls:
        parts = urlparse(url)
        if parts.scheme not in {'http', 'https'} or parts.hostname != urlparse(base).hostname:
            continue
        match = re.search(r'/s(\d+)/(video|align)/([^/]+)$', parts.path)
        if not match:
            continue
        number, category, name = match.groups()
        kind = 'video_normal' if category == 'video' and name.endswith('.mpg_vcd.zip') else 'alignment' if category == 'align' and name.endswith('.tar') else None
        if kind:
            key = (int(number), kind)
            if key in found and found[key] != url:
                raise ValueError('Ambiguous official links: ' + str(key))
            found[key] = url
    return found


def verify_archive(path, kind):
    if kind == 'video_normal':
        with zipfile.ZipFile(path) as archive:
            names = [item for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith('.mpg')]
            if not names or archive.testzip() is not None:
                raise ValueError('Empty or corrupt video ZIP')
            return len(names), sum(item.file_size for item in names)
    with tarfile.open(path, 'r:*') as archive:
        members = [item for item in archive if item.isfile() and item.name.lower().endswith('.align')]
        if not members:
            raise ValueError('No word annotations in TAR')
        for item in members:
            contents = archive.extractfile(item).read()
            if len(contents) != item.size:
                raise ValueError('Truncated annotation member')
            if not re.search(rb'\d+\s+\d+\s+\w+', contents):
                raise ValueError('Invalid alignment text')
        return len(members), sum(item.size for item in members)


def download(url, destination, *, attempts, deadline, reserve_bytes):
    partial = destination.with_name(destination.name + '.part')
    sidecar = partial.with_name(partial.name + '.json')
    for attempt in range(attempts):
        if time.monotonic() >= deadline:
            raise TimeoutError('Run time budget reached; partial file retained')
        offset = partial.stat().st_size if partial.exists() else 0
        previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if offset and offset == previous.get('expected_bytes'):
            return partial, previous  # Complete bytes can be verified after an interrupted EOF/rename.
        headers = {'User-Agent': 'LipNet-GRID-research-acquisition/1.0', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
            if previous.get('etag') or previous.get('last_modified'):
                headers['If-Range'] = previous.get('etag') or previous['last_modified']
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                if response.status not in {200, 206}:
                    raise ValueError('Unexpected HTTP status: ' + str(response.status))
                if 'html' in response.headers.get('Content-Type', '').lower():
                    raise ValueError('Server returned HTML instead of an archive')
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset:
                        raise ValueError('Invalid resume Content-Range')
                    expected = int(match[3])
                    mode = 'ab'
                else:
                    offset = 0
                    expected = int(response.headers.get('Content-Length', 0))
                    mode = 'wb'  # A server ignoring Range must restart, never append.
                if not expected:
                    raise ValueError('Archive has no verifiable Content-Length')
                if shutil.disk_usage(destination.parent).free < reserve_bytes + expected * 2:
                    raise OSError('Insufficient free disk with reserve and extraction allowance')
                metadata = {'url': url, 'final_url': response.url, 'expected_bytes': expected,
                    'etag': response.headers.get('ETag'), 'last_modified': response.headers.get('Last-Modified'),
                    'http_status': response.status, 'range_supported': response.status == 206 or response.headers.get('Accept-Ranges') == 'bytes'}
                atomic_json(sidecar, metadata)
                print(f'{destination.parent.parent.name} {destination.name}: HTTP {response.status}, {offset:,}/{expected:,} bytes', flush=True)
                with partial.open(mode) as target:
                    transferred, last_log = offset, time.monotonic()
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Run time budget reached; partial file retained')
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        if transferred == 0 and block.lstrip().lower().startswith((b'<!doctype html', b'<html')):
                            raise ValueError('HTML body masquerading as archive')
                        target.write(block); transferred += len(block)
                        if time.monotonic() - last_log >= 15:
                            print(f'{destination.name}: {transferred/1e6:.1f}/{expected/1e6:.1f} MB', flush=True)
                            last_log = time.monotonic()
                if partial.stat().st_size != expected:
                    raise ValueError(f'Truncated response: {partial.stat().st_size} != {expected}')
                return partial, metadata
        except Exception as error:
            print(f'{destination.name}: attempt {attempt+1}/{attempts}: {error}', flush=True)
            if attempt + 1 == attempts or time.monotonic() >= deadline:
                raise
            time.sleep(min(8, 2 ** (attempt + 1)))
    raise RuntimeError('Download retry loop exhausted')


def extract_preserving(archive_path, corpus, speaker, kind):
    number = int(speaker[1:])
    target = corpus / f'T{number:02d}' / ('video' if kind == 'video_normal' else 'alignment') / (speaker if kind == 'video_normal' else 'align')
    if not target.resolve().is_relative_to(corpus.resolve()):
        raise ValueError('Extraction escapes the corpus directory')
    target.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    seen = set()
    archive = zipfile.ZipFile(archive_path) if kind == 'video_normal' else tarfile.open(archive_path, 'r:*')
    with archive:
        members = archive.infolist() if kind == 'video_normal' else archive.getmembers()
        for member in members:
            name = member.filename if kind == 'video_normal' else member.name
            components = name.replace('\\', '/').split('/')
            if name.startswith(('/', '\\')) or '..' in components or any(':' in value for value in components):
                raise ValueError('Unsafe archive member path')
            if not name.lower().endswith('.mpg' if kind == 'video_normal' else '.align'):
                continue
            if kind != 'video_normal' and not member.isfile():
                raise ValueError('Nonregular annotation member')
            basename = Path(name.replace('\\', '/')).name
            if basename in seen:
                raise ValueError('Duplicate archive basename: ' + basename)
            seen.add(basename)
            contents = archive.read(member) if kind == 'video_normal' else archive.extractfile(member).read()
            destination = target / basename
            if destination.exists():
                if destination.read_bytes() != contents:
                    raise FileExistsError('Existing differing file preserved: ' + str(destination))
                skipped += 1; continue
            temporary = destination.with_name(destination.name + '.extracting')
            temporary.write_bytes(contents)
            os.replace(temporary, destination)
            written += 1
    return {'written': written, 'existing_verified': skipped, 'members': len(seen), 'target': str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/external/grid')
    parser.add_argument('--speakers', nargs='+', default=['all'])
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--max-minutes', type=float, default=45)
    parser.add_argument('--reserve-gib', type=float, default=20)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.parallel < 1 or args.attempts < 1 or args.max_minutes <= 0:
        parser.error('Parallelism, attempts, and time budget must be positive')
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(INDEX, headers={'User-Agent': 'LipNet-GRID-research-acquisition/1.0'}), timeout=30) as response:
        html = response.read().decode('utf-8')
    links = discover_links(html)
    available = sorted(number for number, kind in links if kind == 'video_normal' and (number, 'alignment') in links)
    requested = available if args.speakers == ['all'] else sorted({int(value.lower().removeprefix('s')) for value in args.speakers})
    unknown = set(requested) - set(available)
    if unknown:
        parser.error('Requested speakers have no verified official video/alignment pair: ' + str(sorted(unknown)))
    print(json.dumps({'official_speakers': available, 'selected': requested, 'free_gib': round(shutil.disk_usage(root).free/2**30, 2), 'audio_downloaded': False}), flush=True)
    if args.dry_run:
        print(json.dumps({f's{n}': {kind:links[n,kind] for kind in ('video_normal','alignment')} for n in requested}, indent=2)); return
    (root / 'official-index.html').write_text(html, encoding='utf-8')
    manifest_path = root / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'source': INDEX, 'research_conditions': 'Publisher states freely available for research use. No broader redistribution/commercial grant is inferred.', 'assets': {}}
    manifest['index_sha256'] = sha256(root / 'official-index.html')
    manifest['free_bytes_before'] = shutil.disk_usage(root).free
    deadline = time.monotonic() + args.max_minutes * 60
    def speaker_job(number):
        speaker = f's{number}'
        archive_dir = root / speaker / 'archives'; archive_dir.mkdir(parents=True, exist_ok=True)
        for kind in ('alignment', 'video_normal'):
            key = speaker + '/' + kind
            url = links[number, kind]
            filename = f'{speaker}_align.tar' if kind == 'alignment' else f'{speaker}_video_normal.mpg_vcd.zip'
            destination = archive_dir / filename
            record = {'speaker_id': speaker, 'kind': kind, 'source_url': url, 'path': str(destination.relative_to(root)), 'status': 'pending'}
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Time budget reached before starting asset')
                if destination.exists():
                    members, expanded = verify_archive(destination, kind)
                    record['download'] = 'existing_archive_verified'
                else:
                    partial, metadata = download(url, destination, attempts=args.attempts, deadline=deadline, reserve_bytes=int(args.reserve_gib*2**30))
                    members, expanded = verify_archive(partial, kind)
                    if destination.exists():
                        raise FileExistsError('Another process created the archive; preserved')
                    os.replace(partial, destination)
                    record.update(metadata); record['download'] = 'downloaded'
                record.update(bytes=destination.stat().st_size, sha256=sha256(destination), archive_members=members, expanded_bytes=expanded, archive_integrity='passed')
                record['extraction'] = extract_preserving(destination, root / speaker / 'corpus', speaker, kind)
                record['status'] = 'verified_and_extracted'
                print(f'{key}: verified, {members} members, {record["bytes"]:,} bytes', flush=True)
            except Exception as error:
                record.update(status='failed_or_deferred', error=str(error))
                print(f'{key}: {error}', flush=True)
            finally:
                record['checked_utc'] = datetime.now(timezone.utc).isoformat()
                with LOCK:
                    manifest['assets'][key] = record
                    atomic_json(manifest_path, manifest)
    # Existing s1 is checked first; new speakers start at s2 in ascending order.
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(speaker_job, number) for number in requested]
        for future in as_completed(futures):
            future.result()
    manifest['free_bytes_after'] = shutil.disk_usage(root).free
    atomic_json(manifest_path, manifest)
    failures = [key for key,value in manifest['assets'].items() if value['status'] != 'verified_and_extracted']
    print(json.dumps({'manifest': str(manifest_path), 'failures': failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
