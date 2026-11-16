"""Score actual video-only decoder outputs; this module is not a trained decoder."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
from .adapters import normalize
from pipeline.word_recognition.common import atomic_json, sha256


def edit_distance(reference, hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,left in enumerate(reference,1):
        current=[i]
        for j,right in enumerate(hypothesis,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]


def ctc_greedy(logits, vocabulary, blank=0):
    """Collapse real [T,C] scores. Token joining must match the trained dictionary.

    Returns tokens, not invented spaces/BPE text. A real checkpoint's dictionary
    and tokenizer are required before these tokens can become a transcript.
    """
    logits=np.asarray(logits)
    if logits.ndim!=2 or logits.shape[0]<1 or logits.shape[1]!=len(vocabulary) or not np.isfinite(logits).all():
        raise ValueError('Finite [time,dictionary] logits required.')
    if len(set(vocabulary))!=len(vocabulary) or not 0<=blank<len(vocabulary):raise ValueError('Invalid CTC dictionary.')
    result=[];previous=None
    for index in logits.argmax(axis=1).tolist():
        if index!=previous and index!=blank:result.append(vocabulary[index])
        previous=index
    return result


def score(manifest, predictions, output):
    """Micro-average edit errors over exactly the held-out manifest IDs.

    Prediction provenance is recorded, not independently proved by this scorer.
    Empty hypotheses are valid errors. Empty reference strings are rejected.
    Character errors include normalized spaces and punctuation, stated explicitly.
    """
    output=Path(output)
    if output.exists():raise FileExistsError('Evaluation output is protected.')
    rows=[json.loads(x) for x in Path(manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    held=[r for r in rows if r.get('split')=='test' and r.get('task') in {'sentence','phrase'}]
    reference={r['id']:r for r in held}
    if not reference or len(reference)!=len(held):raise ValueError('Unique held-out sequence references required.')
    items=[json.loads(x) for x in Path(predictions).read_text(encoding='utf-8').splitlines() if x.strip()]
    if len({p['id'] for p in items})!=len(items) or {p['id'] for p in items}!=set(reference):raise ValueError('Predictions must match every held-out ID exactly once.')
    groups=defaultdict(lambda:dict(samples=0,word_errors=0,reference_words=0,character_errors=0,reference_characters=0,exact=0))
    provenance=set();results=[]
    for item in items:
        if item.get('modality')!='video_only' or not re.fullmatch('[a-f0-9]{64}',item.get('checkpoint_sha256','')):
            raise ValueError('Actual video-only checkpoint provenance required, not a template sentence.')
        row=reference[item['id']];ref=normalize(row['transcript'])
        if not isinstance(item.get('hypothesis'),str):raise ValueError('Decoder hypothesis must be text.')
        hyp=normalize(item['hypothesis']) if item['hypothesis'].strip() else ''
        errors=edit_distance(ref.split(),hyp.split());chars=edit_distance(ref,hyp)
        result={'id':item['id'],'reference':ref,'hypothesis':hyp,'word_errors':errors,'character_errors':chars}
        results.append(result);provenance.add(item['checkpoint_sha256'])
        for key in ('overall',f"dataset:{row['dataset']}",f"speaker:{row['dataset']}:{row.get('speaker_id','unknown')}"):
            g=groups[key];g['samples']+=1;g['word_errors']+=errors;g['reference_words']+=len(ref.split())
            g['character_errors']+=chars;g['reference_characters']+=len(ref);g['exact']+=int(ref==hyp)
    for g in groups.values():
        g.update(wer=g['word_errors']/g['reference_words'],cer=g['character_errors']/g['reference_characters'],exact_sequence_accuracy=g['exact']/g['samples'])
    report={'metrics':dict(groups),'samples':results,'checkpoint_sha256':sorted(provenance),
            'reference_manifest_sha256':sha256(manifest),'predictions_sha256':sha256(predictions),
            'character_policy':'Unicode NFC casefold, whitespace collapsed; includes spaces and punctuation',
            'provenance_limit':'Scoring metadata does not independently verify that predictions came from a model.'}
    atomic_json(output,report)
    return report
