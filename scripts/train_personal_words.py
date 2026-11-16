"""Run the real personal-recording workflow in the existing Windows CPU environments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.word_recognition.common import atomic_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles',type=Path,nargs='+',required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--through',choices=['import','extract','train','evaluate'],default='evaluate')
    parser.add_argument('--minutes',type=int,default=30)
    args=parser.parse_args()
    if not 1<=args.minutes<=120:parser.error('Choose a 1–120 minute extraction budget.')
    run=args.run.resolve();environments={'word':ROOT/'.venv-word-recognition/Scripts/python.exe','encoder':ROOT/'.venv-avhubert-cpu/Scripts/python.exe'}
    if not all(p.is_file() for p in environments.values()):parser.error('The existing word-recognition and AV-HuBERT CPU environments are required.')
    if not all(p.is_file() for p in args.bundles):parser.error('Provide the exported JSON recording file(s), not a placeholder.')
    protocol=run/'personal_protocol.json'
    if protocol.exists():
        saved=json.loads(protocol.read_text())
        if sorted(sha256(p) for p in args.bundles)!=sorted(r['sha256'] for r in saved['bundles']):
            parser.error('Resume requires the exact same exported bundles. Choose a new run for changed recordings.')
    stages=['import','extract','train','evaluate'];stages=stages[:stages.index(args.through)+1]
    logs=run.parent/'personal_logs';logs.mkdir(parents=True,exist_ok=True);token=uuid.uuid4().hex[:10];receipts=[]
    for stage in stages:
        completed={'import':'personal_protocol.json','extract':'features.jsonl','train':'selection.json','evaluate':'evaluation.json'}[stage]
        if (run/completed).exists():
            print('Reuse completed '+stage,flush=True);continue
        python=environments['encoder' if stage=='extract' else 'word']
        command=[str(python),'-u','-m','pipeline.word_recognition.personalized',stage,'--run',str(run),'--minutes',str(args.minutes)]
        if stage=='import':command+=['--bundles',*[str(p.resolve()) for p in args.bundles]]
        log=logs/f'{run.name}-{token}-{stage}.txt';started=time.perf_counter()
        print(f'{stage}: {log}',flush=True)
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=(args.minutes+10)*60,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        receipts.append({'stage':stage,'command':command,'exit_code':result.returncode,'seconds':time.perf_counter()-started,'log':str(log)})
        atomic_json(logs/f'{run.name}-{token}-commands.json',receipts)
        if result.returncode:raise SystemExit(f'{stage} failed (exit {result.returncode}); inspect {log}. Completed caches are retained.')
    print(f'Finished through {args.through}: {run}')
    if (run/'evaluation.json').exists():
        report=json.loads((run/'evaluation.json').read_text())
        print(json.dumps({k:report[k] for k in ['correct','test_samples','top1_accuracy','difference_percentage_points','eligible_for_manual_research_activation']},indent=2))
        print('Active models were not changed. Review the comparison before selecting a personal head.')


if __name__=='__main__':main()
