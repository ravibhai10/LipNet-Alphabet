"""Replay frozen predictions, never infer/train or send data to a provider.

This measures preservation and local gate overhead, not contextual LLM accuracy.
"""
import asyncio
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.services.context_interpretation import InterpretationRequest, InterpretationService


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def macro_f1(labels,predictions,vocab):
    values=[]
    for word in vocab:
        tp=sum(a==word and b==word for a,b in zip(labels,predictions))
        fp=sum(a!=word and b==word for a,b in zip(labels,predictions))
        fn=sum(a==word and b!=word for a,b in zip(labels,predictions))
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return statistics.mean(values)


async def benchmark():
    path=ROOT/'.qa/reliability_20261109/final_frozen_evaluation.json'
    data=json.loads(path.read_text()); vocab=data['vocabulary']; calls=[]
    async def forbidden(payload):
        calls.append(payload)
        raise AssertionError('Isolated clips do not provide genuine context')
    service=InterpretationService(forbidden)
    report={'scope':'Frozen prediction replay only, no fresh model inference, no provider calls or contextual accuracy claim.',
        'source':str(path.relative_to(ROOT)).replace('\\','/'),'source_sha256':digest(path),
        'threshold_tuning':False,'test_model_selection':False,'automatic_correction':False,'partitions':{}}
    logging.getLogger('app.services.context_interpretation').disabled=True
    for partition,values in data['partitions'].items():
        predictions=values['predictions']; labels=[]; base=[]; enhanced=[]; latency=[]; abstained=0
        for i,p in enumerate(predictions):
            top=sorted(zip(vocab,p['probabilities']),key=lambda pair:pair[1],reverse=True)[:5]
            original=p['candidate'] if p['accepted_after'] else None
            request=InterpretationRequest(mode='word',event_id=f'{partition}_{i}',visual_candidate=p['candidate'],
                original_word=original,visual_uncertain=original is None,candidates=[{'word':w,'prob':s} for w,s in top],
                context='',history=[],consent_text=False)
            start=time.perf_counter(); result=await service.interpret(request,'offline-replay')
            latency.append((time.perf_counter()-start)*1000)
            assert result['safe_prediction']==original and result['applied'] is False
            labels.append(p['label']); base.append(p['candidate']); enhanced.append(result['visual_candidate'])
            abstained+=result['safe_prediction'] is None
        ordered=sorted(latency); n=len(labels)
        metrics={'accuracy':sum(a==b for a,b in zip(labels,base))/n,'macro_f1':macro_f1(labels,base,vocab),
            'abstention_rate':abstained/n}
        report['partitions'][partition]={'samples':n,'base':metrics,'enhanced':dict(metrics),
            'corrections_gained':0,'errors_introduced':0,'net_gain':0,'safe_prediction_changes':0,
            'gate_overhead_ms':{'p50':statistics.median(latency),'p95':ordered[min(n-1,int(.95*n))]},
            'base_model_latency_ms':None,'provider_latency_ms':None,
            'limitation':'No conversational context. Existing accepted_after consensus decisions are replayed, not freshly evaluated.'}
    report['provider_calls']=len(calls)
    assert not calls
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'.qa/llm_context_20261118')
    output=parser.parse_args().output; output.mkdir(parents=True,exist_ok=True)
    for name in ('benchmark.json','protected_comparison.json'):
        if (output/name).exists(): raise SystemExit(f'Refusing to replace report: {name}')
    report=asyncio.run(benchmark())
    before=json.loads((ROOT/'.qa/webcam_phase1_20261118/before.json').read_text())
    protected=dict(before['checkpoints'])
    for path,expected in before['existing_file_hashes'].items():
        if any(s in path.lower() for s in ('manifest','vocab','evaluation','training_summary','baseline')):
            protected[path]=expected
    compared=[]
    for path,expected in protected.items():
        actual=digest(ROOT/path) if (ROOT/path).is_file() else None
        compared.append({'path':path,'expected_sha256':expected,'sha256':actual,'unchanged':expected==actual})
    comparison={'scope':'All 114 inventoried .pt/.pth assets plus snapshot manifests, vocabularies and evaluation artifacts. Not a new full 44 GB dataset hash scan.',
        'compared':len(compared),'changed':[row for row in compared if not row['unchanged']], 'files':compared}
    report['free_disk_gib']=shutil.disk_usage(ROOT).free/(1024**3)
    report['process']='CPU-only text policy replay; existing environments/dependencies unchanged.'
    for name,value in [('benchmark.json',report),('protected_comparison.json',comparison)]:
        with (output/name).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2)
    print(json.dumps({'benchmark':report,'protected_compared':len(compared),'changed':len(comparison['changed'])},indent=2))
    if comparison['changed']: raise SystemExit(1)


if __name__=='__main__': main()
