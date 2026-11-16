"""Recompute the frozen word baseline and a prespecified consensus rejection rule.

This never fits weights or thresholds. Historical test reuse is explicit and
is not a new untouched benchmark. Validation/test records remain independent
utterances; overlapping consistency windows never increase the denominator.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.word_recognition.common import read_rows,sha256
from pipeline.word_recognition.pretrained import LAYERS,sequence_probabilities
from pipeline.word_recognition.consensus import temporal_consensus
from pipeline.word_recognition.temporal_experiment import metrics


def evaluate(run, split):
    if split not in {'valid','test'}:raise ValueError('Only validation or explicitly reopened historical test.')
    head=torch.load(run/'head.pt',map_location='cpu',weights_only=True)
    policy=json.loads((run/'rejection_policy.json').read_text())
    if policy['head_sha256']!=sha256(run/'head.pt'):raise ValueError('Frozen policy/head mismatch.')
    recorded=json.loads((run/'evaluation.json').read_text())['rejection_policy']
    verified_policy={**policy,**{k:recorded[k] for k in ('threshold','margin')}}
    rows=[r for r in read_rows(run/'features.jsonl') if r['split']==split]
    known=[];phrase=[]
    for row in rows:
        if not row['usable']:
            target=known if row['role']=='known_word' else phrase
            target.append({'id':row['id'],'probabilities':None,'accepted_before':False,'accepted_after':False,
                           'correct':False,'label':row.get('word_label')});continue
        if sha256(row['feature'])!=row['feature_sha256']:raise ValueError('Frozen feature changed.')
        sequence=np.load(row['feature'],allow_pickle=False)[LAYERS.index(head['feature_layer'])]
        p=sequence_probabilities(head,sequence);order=np.argsort(-p)
        accepted=bool(p[order[0]]>=policy['threshold'] and p[order[0]]-p[order[1]]>=policy['margin'])
        consensus=temporal_consensus(head,sequence,verified_policy)
        candidate=head['vocabulary'][int(order[0])]
        verified=bool(p[order[0]]>=verified_policy['threshold'] and p[order[0]]-p[order[1]]>=verified_policy['margin'])
        after=bool(verified and consensus['confirmed'] and consensus['views'][0]['candidate']==candidate)
        result={'id':row['id'],'speaker':row['speaker_id'],'label':row.get('word_label'),'candidate':candidate,
                'probabilities':p.tolist(),'correct':candidate==row.get('word_label'),
                'accepted_before':accepted,'accepted_after':after}
        (known if row['role']=='known_word' else phrase).append(result)
    usable=[r for r in known if r['probabilities'] is not None]
    p=np.array([r['probabilities'] for r in usable]);y=np.array([head['vocabulary'].index(r['label']) for r in usable])
    result=metrics(p,y,head['vocabulary'],len(known))
    confidences=p.max(1);correct=p.argmax(1)==y;ece=0.
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidences>low)&(confidences<=high)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidences[mask].mean())
    result.update(split=split,head_sha256=sha256(run/'head.pt'),confidence_calibrated=False,
                  before_policy=policy,after_policy=verified_policy,
                  expected_calibration_error_10_bins=float(ece),
                  brier_score=float(np.mean(np.sum((p-np.eye(len(head['vocabulary']))[y])**2,axis=1))),
                  predictions=known,phrase_predictions=phrase,
                  errors=[r for r in known if not r['correct']],
                  limitations='Same frozen historical corpus. Phrase false acceptance does not measure arbitrary unseen isolated-word OOV rejection.')
    for suffix in ['before','after']:
        accepted=[r for r in known if r['accepted_'+suffix]]
        result[suffix]={'known_samples':len(known),'accepted':len(accepted),
                        'accepted_correct':sum(r['correct'] for r in accepted),
                        'rejection_rate':1-len(accepted)/len(known),
                        'accepted_accuracy':sum(r['correct'] for r in accepted)/len(accepted) if accepted else None,
                        'phrase_samples':len(phrase),'phrase_false_accepts':sum(r['accepted_'+suffix] for r in phrase),
                        'phrase_false_accept_rate':sum(r['accepted_'+suffix] for r in phrase)/len(phrase)}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--include-historical-test',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Evaluation output protected; choose a new output.')
    run=ROOT/'runs/word_accuracy/avhubert_miracl8_v1'
    result={'vocabulary':torch.load(run/'head.pt',weights_only=True)['vocabulary'],
            'policy':'Before: current policy file. After: recorded evaluation gate plus three fixed 25-frame feature windows; no tuning.',
            'model_selection_test_used':False,
            'historical_test_reopened':args.include_historical_test,
            'partitions':{s:evaluate(run,s) for s in (['valid','test'] if args.include_historical_test else ['valid'])}}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({s:{k:v for k,v in r.items() if k not in {'predictions','phrase_predictions','errors','per_class','confusion_matrix'}}
                      for s,r in result['partitions'].items()},indent=2))
