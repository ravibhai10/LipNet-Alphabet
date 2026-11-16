"""Acquire the author's non-boundary visual-only LRW research checkpoint.

Model weights are not the restricted LRW corpus. Use is limited by the author's
research/comparison license; no publisher accuracy is a local benchmark result.
"""
from pathlib import Path
import json
import re
import subprocess
import sys
from urllib.parse import urlparse,parse_qs
import requests

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.multidataset.acquisition import Page,transfer
from pipeline.word_recognition.common import atomic_json,sha256,timestamp


def main():
    source=ROOT/'external/lrw_tcn_reference'
    readme=(source/'README.md').read_text()
    row=next(x for x in readme.splitlines() if x.startswith('|resnet18_dctcn_video '))
    link=re.search(r'\[GoogleDrive\]\(([^)]+)\)',row)[1]
    session=requests.Session();page=session.get(link,timeout=30);page.raise_for_status()
    parsed=urlparse(page.url);match=re.search(r'/file/d/([^/]+)',parsed.path)
    if parsed.scheme!='https' or parsed.hostname!='drive.google.com' or not match:raise PermissionError('Author checkpoint link is not a public Drive file.')
    identifier=match[1]
    def opener(headers):
        response=session.get('https://drive.google.com/uc',params={'id':identifier,'export':'download'},headers=headers,stream=True,timeout=30)
        if 'html' in response.headers.get('Content-Type','').lower():
            response.raise_for_status();form=Page();form.feed(next(response.iter_content(65536),b'').decode('utf-8','replace'));response.close()
            action=urlparse(form.action or '')
            if action.scheme!='https' or action.hostname!='drive.usercontent.google.com' or action.path!='/download' or form.fields.get('id')!=identifier or 'confirm' not in form.fields:
                raise PermissionError('Checkpoint is not accessible via normal public confirmation.')
            response=session.get(form.action,params=form.fields,headers=headers,stream=True,timeout=30)
        return response,link
    root=ROOT/'models/external/lrw_tcn';root.mkdir(parents=True,exist_ok=True)
    previous=json.loads((root/'acquisition.json').read_text()) if (root/'acquisition.json').exists() else None
    result=transfer(opener,root/'resnet18_dctcn_video.pth',max_bytes=512*2**20,minutes=5)
    if previous:
        if previous['sha256']!=result['sha256'] or previous['bytes']!=result['bytes']:raise ValueError('Previously verified author checkpoint changed; preserved without updating receipt.')
        if result['status']=='existing':result={**previous,'last_verification':result}
    result.update(checked_at=timestamp(),publisher='mpc001/Lipreading_using_Temporal_Convolutional_Networks',
                  source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                  model='resnet18_dctcn_video',modality='video_only',word_boundary_input=False,
                  config_sha256=sha256(source/'configs/lrw_resnet18_dctcn.json'),
                  labels_sha256=sha256(source/'labels/500WordsSortedList.txt'),
                  license_sha256=sha256(source/'LICENSE'),license='Author license: individual research/private study/comparative prototype; not commercial operational development',
                  publisher_accuracy_not_local_evaluation=True,publisher_checksum_available=False)
    atomic_json(root/'acquisition.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
