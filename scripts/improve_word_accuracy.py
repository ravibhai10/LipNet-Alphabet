"""Windows CPU stage runner; one model process, explicit logs and native exits."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.word_recognition.common import atomic_json,timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'runs/word_accuracy/avhubert_miracl8_v1')
    p.add_argument('--through',choices=('prepare','extract','train','evaluate'),default='train')
    args=p.parse_args();run=args.run.resolve()
    word=ROOT/'.venv-word-recognition/Scripts/python.exe';encoder=ROOT/'.venv-avhubert-cpu/Scripts/python.exe'
    if not word.is_file() or not encoder.is_file():p.error('The existing isolated word and AV-HuBERT CPU environments are required.')
    stages=['init','prepare','extract','train','evaluate'];stages=stages[:stages.index(args.through)+1]
    if 'extract' in stages and not (run/'features.jsonl').exists():
        subprocess.run([str(word),str(ROOT/'scripts/patch_fairseq_meta.py')],cwd=ROOT,check=True)
    files={'init':'protocol.json','prepare':'prepared.jsonl','extract':'features.jsonl','train':'selection.json','evaluate':'evaluation.json'}
    logs=run.parent/'logs';logs.mkdir(parents=True,exist_ok=True);invocation=uuid.uuid4().hex[:10];records=[]
    for stage in stages:
        if (run/files[stage]).exists():
            print(f'Reuse completed {stage}: {run/files[stage]}',flush=True);continue
        python=encoder if stage=='extract' else word
        command=[str(python),'-u','-m','pipeline.word_recognition.pretrained',stage,'--run',str(run),'--minutes','60' if stage=='extract' else '30']
        log=logs/f'{run.name}-{invocation}-{stage}.txt'
        print(f'Running {stage}; log {log}',flush=True)
        entry={'stage':stage,'command':command,'log':str(log),'started_at':timestamp()};records.append(entry)
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'},timeout=75*60)
        entry.update(returncode=result.returncode,finished_at=timestamp())
        atomic_json(logs/f'{run.name}-{invocation}-commands.json',records)
        if result.returncode:
            raise RuntimeError(f'{stage} failed with native exit {result.returncode}; inspect {log}. Completed inputs/caches preserved.')
    print(f'Completed through {args.through}; artifacts {run}',flush=True)


if __name__=='__main__':main()
