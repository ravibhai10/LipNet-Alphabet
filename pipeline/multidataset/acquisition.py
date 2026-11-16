"""Bounded public-source transfers and preserving ZIP/TAR extraction.

Drive support follows the ordinary public virus-scan confirmation form only.
Login, authorization and HTML error pages are never treated as dataset bytes.
"""
from __future__ import annotations
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
from urllib.parse import urljoin, urlparse, parse_qs
import zipfile
import requests
from pipeline.word_recognition.common import atomic_json, sha256, timestamp

MIRACL_PAGE="https://abenhamadou.github.io/miraclvc1/"


class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=='a' and values.get('href'):self.links.append(values['href'])
        if tag=='form':self.action=values.get('action')
        if tag=='input' and values.get('name'):self.fields[values['name']]=values.get('value','')


def miracl_public_response(session, headers):
    index=session.get(MIRACL_PAGE,timeout=30);index.raise_for_status()
    parser=Page();parser.feed(index.text)
    links=[urljoin(MIRACL_PAGE,x) for x in parser.links if urlparse(urljoin(MIRACL_PAGE,x)).hostname=='drive.google.com' and urlparse(urljoin(MIRACL_PAGE,x)).scheme=='https']
    if len(links)!=1:raise ValueError("Author index no longer identifies exactly one complete public archive.")
    response=session.get(links[0],headers=headers,stream=True,timeout=30)
    if 'html' not in response.headers.get('Content-Type','').lower():return response,links[0]
    response.raise_for_status()
    text=next(response.iter_content(65536),b'').decode('utf-8','replace');response.close()
    form=Page();form.feed(text)
    parsed=urlparse(form.action or '')
    if parsed.scheme!='https' or parsed.hostname!='drive.usercontent.google.com' or parsed.path!='/download' or 'confirm' not in form.fields:
        raise PermissionError("Author link requires authorization or has no ordinary public download confirmation.")
    if form.fields.get('id')!=parse_qs(urlparse(links[0]).query).get('id',[None])[0]:
        raise ValueError('Public confirmation identifies a different archive.')
    # Do not persist the ephemeral UUID/confirmation fields or Google session cookies.
    return session.get(form.action,params=form.fields,headers=headers,stream=True,timeout=30),links[0]


def transfer(opener, target, *, max_bytes=6*2**30, reserve_bytes=20*2**30, minutes=10, expected_sha256=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        actual=sha256(target)
        if expected_sha256 and actual!=expected_sha256:raise ValueError("Existing archive fingerprint mismatch; preserved.")
        return {"status":"existing", "bytes":target.stat().st_size,"sha256":actual}
    partial=target.with_name(target.name+'.part');sidecar=partial.with_name(partial.name+'.json')
    deadline=time.monotonic()+minutes*60
    for attempt in range(3):
        if time.monotonic()>=deadline:raise TimeoutError("Download budget expired; partial retained.")
        offset=partial.stat().st_size if partial.exists() else 0
        previous=json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if offset and not previous:raise ValueError("Unidentified partial file is protected.")
        if offset and offset==previous.get('expected_bytes'):
            actual=sha256(partial)
            if expected_sha256 and actual!=expected_sha256:raise ValueError('Completed partial checksum mismatch; preserved.')
            if not (zipfile.is_zipfile(partial) or tarfile.is_tarfile(partial)):raise ValueError('Completed partial is not a readable archive.')
            partial.rename(target)
            return {'status':'recovered_complete_partial','bytes':offset,'sha256':actual,'source':previous.get('publisher_url'),
                    'publisher_checksum_compared':expected_sha256 is not None}
        headers={'Accept-Encoding':'identity'}
        if offset:
            headers['Range']=f'bytes={offset}-'
            if previous.get('etag'):headers['If-Range']=previous['etag']
        try:
            response,source=opener(headers)
            with response:
                if response.status_code in (429,500,502,503,504):
                    raise requests.HTTPError('Temporary publisher failure; bounded retry.')
                if response.status_code not in (200,206) or 'html' in response.headers.get('Content-Type','').lower():
                    raise PermissionError(f"Dataset unavailable: HTTP {response.status_code} or HTML response.")
                identity=hashlib.sha256(source.encode()).hexdigest()
                if offset and previous.get('source_identity')!=identity:raise ValueError("Partial belongs to another publisher asset.")
                mode='wb'
                if response.status_code==206:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset:raise ValueError("Invalid resume byte range.")
                    expected=int(match[3]);mode='ab'
                    if previous.get('expected_bytes') not in (None,expected) or previous.get('etag') and response.headers.get('ETag') not in (None,previous['etag']):raise ValueError("Resumed entity changed.")
                else:
                    offset=0;expected=int(response.headers.get('Content-Length','0'))
                if not 0<expected<=max_bytes:raise ValueError("Missing or excessive verified archive size.")
                if shutil.disk_usage(target.parent).free<reserve_bytes+expected-offset:raise OSError("Insufficient free disk with reserve.")
                chunks=response.iter_content(2**20);first=next(chunks,b'')
                is_tar=len(first)>262 and first[257:262]==b'ustar'
                if offset==0 and (first.lstrip().lower().startswith((b'<html',b'<!doctype')) or not (first.startswith((b'PK',b'BZh',b'\x1f\x8b')) or is_tar)):raise ValueError("Response is not a supported archive.")
                if offset+len(first)>expected:raise ValueError('Oversized initial archive block.')
                atomic_json(sidecar,{'source_identity':identity,'expected_bytes':expected,'etag':response.headers.get('ETag'),'publisher_url':source})
                count,last_log=offset,time.monotonic()
                with partial.open(mode) as stream:
                    for block in (first,):stream.write(block);count+=len(block)
                    for block in chunks:
                        if time.monotonic()>=deadline:raise TimeoutError("Download budget expired; partial retained.")
                        if count+len(block)>expected:raise ValueError("Oversized archive response.")
                        stream.write(block);count+=len(block)
                        if time.monotonic()-last_log>=15:
                            print(f"Dataset bytes {count:,}/{expected:,}",flush=True);last_log=time.monotonic()
                if count!=expected:raise ValueError("Incomplete download; partial retained.")
                actual=sha256(partial)
                if expected_sha256 and actual!=expected_sha256:raise ValueError("Publisher checksum mismatch.")
                if target.exists():raise FileExistsError("Concurrent final archive is protected.")
                partial.rename(target)
                return {'status':'downloaded','bytes':count,'sha256':actual,'source':source,'publisher_checksum_compared':expected_sha256 is not None,
                        'http_status':response.status_code,'content_type':response.headers.get('Content-Type'),
                        'content_range':response.headers.get('Content-Range'),'content_length':response.headers.get('Content-Length'),
                        'etag':response.headers.get('ETag'),'completed_at':timestamp()}
        except (requests.RequestException,ValueError) as error:
            if attempt==2:raise RuntimeError(f"Acquisition stopped ({type(error).__name__}); partial preserved.") from None
            time.sleep(min(2**attempt,2))
    raise RuntimeError("No transfer completed.")


def safe_target(root,name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:raise ValueError("Unsafe archive path.")
    result=root.joinpath(*path.parts).resolve()
    if not result.is_relative_to(root.resolve()):raise ValueError("Archive traversal or link escape.")
    return result


def extract_preserving(archive, root, *, reserve_bytes=20*2**30):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    zipped=zipfile.is_zipfile(archive)
    context=zipfile.ZipFile(archive) if zipped else tarfile.open(archive,'r:*')
    with context as source:
        if zipped and source.testzip() is not None:raise ValueError("ZIP CRC validation failed.")
        members=source.infolist() if zipped else source.getmembers()
        pending,total,seen=[],0,set()
        for item in members:
            name=item.filename if zipped else item.name
            target=safe_target(root,name);key=str(target).casefold()
            if key in seen:raise ValueError("Duplicate/case-colliding archive members.")
            seen.add(key)
            if zipped:
                if (item.external_attr>>16)&0o170000==0o120000:raise ValueError("Archive symlinks rejected.")
                directory=item.is_dir();size=item.file_size
            else:
                if not item.isdir() and not item.isfile():raise ValueError("Archive links/devices rejected.")
                directory=item.isdir();size=item.size
            if directory:continue
            if not target.exists():total+=size
            pending.append((item,target,size))
        if shutil.disk_usage(root).free<total+reserve_bytes:raise OSError("Insufficient disk for expanded members and reserve.")
        written=reused=0
        for item,target,size in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            with (source.open(item) if zipped else source.extractfile(item)) as stream:
                if target.exists():
                    h=hashlib.sha256()
                    for block in iter(lambda:stream.read(2**20),b''):h.update(block)
                    if target.stat().st_size!=size or sha256(target)!=h.hexdigest():raise ValueError("Existing extracted member differs; preserved.")
                    reused+=1;continue
                part=target.with_name(target.name+'.part')
                if part.exists():
                    if part.stat().st_size>size:raise ValueError("Extracted partial is too large.")
                    with part.open('rb') as prefix:
                        for block in iter(lambda:prefix.read(2**20),b''):
                            if stream.read(len(block))!=block:raise ValueError("Extracted partial prefix differs; preserved.")
                with part.open('ab' if part.exists() else 'xb') as out:shutil.copyfileobj(stream,out,2**20)
                if part.stat().st_size!=size:raise ValueError("Truncated archive member.")
                if target.exists():raise FileExistsError("Concurrent extraction target is protected.")
                part.rename(target);written+=1
    return {'members':len(pending),'written':written,'verified_existing':reused,'additional_bytes':total,'archive_sha256':sha256(archive)}


def acquire_miracl(root,minutes=10):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    archive=root/'archives/MIRACL-VC1_all_in_one.zip'
    receipt=root/'acquisition.json'
    previous=json.loads(receipt.read_text()) if receipt.exists() else None
    with requests.Session() as session:
        result=transfer(lambda headers:miracl_public_response(session,headers),archive,minutes=minutes)
    if previous:
        if previous.get('sha256')!=result['sha256'] or previous.get('bytes')!=result['bytes']:
            raise ValueError('Archive differs from recorded acquisition; files and receipt preserved.')
        if result['status']=='existing':
            result={**previous,'last_verification':{**result,'verified_at':timestamp()}}
    # Save acquisition before extraction so a bounded transfer remains reusable.
    atomic_json(root/'acquisition.json',result)
    result['extraction']=extract_preserving(archive,root/'original')
    atomic_json(root/'acquisition.json',result)
    return result
