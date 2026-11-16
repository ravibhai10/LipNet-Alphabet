"""Bounded MIRACL RGB word pilot; separate from GRID and the active backend.

python -m pipeline.multidataset.experiment --manifest ... --run ...
Each held-out speaker contributes the same prespecified repetitions per word.
No audio, phrase-as-word labels, or test-based model selection is permitted.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pipeline.word_recognition.common import ROOT, atomic_json, fingerprint, sha256, read_rows, write_rows, timestamp
from pipeline.word_recognition.model import WordVideoNet
from src.video_training import resample_temporal
from tracking.mouth_canonicalization import aligned_mouth_crop
from .adapters import validate, WORD_TASKS

PREPROCESS={'frames':24,'size':64,'gray':True,'range':[0,1],'alignment':'measured eye-axis mouth similarity; margin 2.2; 96px intermediate',
            'temporal':'linear interpolation of ordered tracked frames; native FPS unknown, no timestamps fabricated',
            'tracking_coverage':0.8,'max_interior_gap':5,'encoder':'scratch WordVideoNet; no pretrained feature cache'}


def available_gib():
    if os.name!='nt':return None
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    state=Memory();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError('Memory preflight unavailable.')
    return state.available/2**30


def select(rows,repetitions):
    if not 1<=repetitions<=10:raise ValueError('Choose 1..10 repetitions, before evaluation.')
    if any(r['dataset']!='miracl' for r in rows):raise ValueError('This pilot is MIRACL only; do not merge incompatible vocabularies.')
    chosen=[r for r in rows if r['task']=='isolated_word' and r['repetition']<=repetitions]
    report=validate(chosen);words=sorted({r['word_label'] for r in chosen})
    if len(words)!=8:raise ValueError('Expected eight unambiguous publisher word classes.')
    counts=Counter((r['speaker_id'],r['word_label']) for r in chosen)
    if any(counts[s,w]!=repetitions for s in {r['speaker_id'] for r in chosen} for w in words):raise ValueError('Incomplete prespecified word/speaker quota.')
    return chosen,words,report


def source_digest(row):
    h=hashlib.sha256()
    for name in row['frames']:
        path=Path(name);h.update(path.name.encode());h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def tracked_clip(row,tracker,mp):
    if row['input_kind']!='color_frames' or source_digest(row)!=row['source_sha256']:raise ValueError('RGB source fingerprint mismatch.')
    crops=[];gap=0;longest=0
    for name in row['frames']:
        image=cv2.imread(name)
        if image is None or image.shape!=(480,640,3):raise ValueError('RGB source decode/geometry changed.')
        result=tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
        if result.face_landmarks:
            points=np.asarray([(p.x,p.y,p.z) for p in result.face_landmarks[0]],dtype=np.float32)
            crop=aligned_mouth_crop(image,points)
            crops.append(cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(64,64),interpolation=cv2.INTER_LINEAR));gap=0
        else:
            gap+=1
            if crops:longest=max(longest,gap)
    if len(crops)<2 or len(crops)<len(row['frames'])*.8 or longest>5:raise ValueError('Insufficient measured face tracking; no fallback crop.')
    clip=np.rint(resample_temporal(crops,24)).clip(0,255).astype(np.uint8)
    if clip.shape!=(24,64,64):raise ValueError('Preprocessed geometry mismatch.')
    return clip,{'tracked':len(crops),'source':len(row['frames']),'max_gap':longest}


def prepare(rows,run,contract,cache_root=None):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    directory=Path(cache_root) if cache_root else run/'clips';directory.mkdir(parents=True,exist_ok=True)
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'frontend/public/models/face_landmarker.task')),
                                        running_mode=vision.RunningMode.IMAGE,num_faces=1,min_face_detection_confidence=.5,min_face_presence_confidence=.5)
    result=[];failures=[];start=time.perf_counter()
    with vision.FaceLandmarker.create_from_options(options) as tracker:
        for n,row in enumerate(rows):
            key=fingerprint({'source':row['source_sha256'],'bounds':[row['start_frame'],row['end_frame']],
                             'preprocessing':PREPROCESS,'landmarker':contract['landmarker_sha256'],'encoder':PREPROCESS['encoder']})
            target=directory/(key+'.npy');meta=target.with_suffix('.json')
            try:
                if target.exists() or meta.exists():
                    if not target.is_file() or not meta.is_file():raise ValueError('Incomplete cache artifact is protected.')
                    detail=json.loads(meta.read_text());clip=np.load(target,allow_pickle=False)
                    if detail['cache_key']!=key or detail['sha256']!=sha256(target) or source_digest(row)!=row['source_sha256']:raise ValueError('Cache fingerprint changed.')
                else:
                    clip,tracking=tracked_clip(row,tracker,mp)
                    temporary=target.with_suffix('.npy.part')
                    if temporary.exists():raise FileExistsError('Unidentified clip partial is protected.')
                    with temporary.open('xb') as stream:np.save(stream,clip,allow_pickle=False)
                    temporary.rename(target)
                    detail={'cache_key':key,'sha256':sha256(target),'tracking':tracking}
                    atomic_json(meta,detail)
                if clip.dtype!=np.uint8 or clip.shape!=(24,64,64):raise ValueError('Cache geometry/type mismatch.')
                result.append({**row,'clip':str(target.resolve()),'clip_sha256':detail['sha256'],'tracking':detail['tracking']})
            except (ValueError,FileNotFoundError) as error:
                failures.append({'id':row['id'],'split':row['split'],'reason':str(error)})
            if (n+1)%24==0:print(f'Measured mouth alignment {n+1}/{len(rows)}',flush=True)
    atomic_json(run/'preparation.json',{'seconds':time.perf_counter()-start,'selected':len(rows),'usable':len(result),'failures':failures})
    if failures and (not contract['allow_training_exclusions'] or any(f['split']!='train' for f in failures)):
        raise ValueError('Prespecified pilot has failed clips; inspect preparation.json. No silent replacements or evaluation.')
    if failures:print(f'Explicitly excluded {len(failures)} training clips only; held-out quotas unchanged.',flush=True)
    validate(result)
    write_rows(run/'selected.jsonl',result)
    return result


class Clips(Dataset):
    def __init__(self,rows,words,training=False):
        if training and any(r['split']!='train' for r in rows):raise ValueError('Augmentation is training only.')
        self.rows=rows;self.words=words;self.training=training
        self.clips=[]
        for row in rows:
            if sha256(row['clip'])!=row['clip_sha256']:raise ValueError('Prepared clip changed.')
            self.clips.append(torch.from_numpy(np.load(row['clip'],allow_pickle=False)))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        x=self.clips[index].float().unsqueeze(0)/255.
        if self.training:
            # Coherent spatial/photometric operations across the whole sequence.
            dx,dy=random.randint(0,6),random.randint(0,6)
            x=torch.nn.functional.pad(x,(3,3,3,3),mode='replicate')[:,:,dy:dy+64,dx:dx+64]
            if random.random()<.5:x=x.flip(-1)
            x=(x*random.uniform(.8,1.2)+random.uniform(-.1,.1)).clamp(0,1)
            if random.random()<.3:
                first=random.randint(1,20);length=random.randint(2,3)
                x[:,first:first+length]=x[:,first-1:first]
        return x,self.words.index(self.rows[index]['word_label']),index


def loader(rows,words,training=False):
    return DataLoader(Clips(rows,words,training),batch_size=2,shuffle=training,num_workers=0)


def pass_epoch(model,batches,optimizer=None):
    model.train(optimizer is not None);loss_sum=correct=total=0;predictions=[]
    criterion=nn.CrossEntropyLoss(label_smoothing=.1 if optimizer is not None else 0)
    with torch.set_grad_enabled(optimizer is not None):
        for tensors,labels,indices in batches:
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            logits=model(tensors);loss=criterion(logits,labels)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss.')
            if optimizer is not None:
                loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            probabilities=logits.detach().softmax(-1)
            loss_sum+=loss.item()*len(labels);correct+=int((logits.argmax(-1)==labels).sum());total+=len(labels)
            predictions.extend({'index':int(i),'target':int(y),'probabilities':p.tolist()} for i,y,p in zip(indices,labels,probabilities))
    if not total:raise ValueError('Empty partition.')
    return {'loss':loss_sum/total,'accuracy':correct/total,'samples':total},predictions


def save_checkpoint(path,artifact):
    temp=path.with_suffix('.pt.tmp');torch.save(artifact,temp);os.replace(temp,path)


def uncertainty(probabilities,words,minimum=.7,margin=.15):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(len(words),) or len(words)<2 or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-5):raise ValueError('Invalid class probabilities.')
    order=np.argsort(-p);accepted=bool(p[order[0]]>=minimum and p[order[0]]-p[order[1]]>=margin)
    return {'word':words[order[0]] if accepted else None,'uncertain':not accepted,'confidence':float(p[order[0]]),
            'candidate':words[order[0]],'threshold':minimum,'margin_threshold':margin,
            'out_of_vocabulary_detection':'not validated; rejection is a confidence heuristic, not proof of vocabulary membership'}


def evaluate(model,rows,words,run,checkpoint):
    from sklearn.metrics import classification_report,confusion_matrix
    metrics,outputs=pass_epoch(model,loader(rows,words));actual=[];predicted=[];conf=[];speakers=defaultdict(lambda:[0,0]);top5=0;accepted=0;records=[]
    for output in outputs:
        row=rows[output['index']];p=np.array(output['probabilities']);target=output['target'];prediction=int(p.argmax());policy=uncertainty(p,words)
        actual.append(target);predicted.append(prediction);conf.append(float(p.max()));top5+=int(target in np.argsort(-p)[:5]);accepted+=int(not policy['uncertain'])
        speakers[row['speaker_id']][0]+=int(target==prediction);speakers[row['speaker_id']][1]+=1
        records.append({'id':row['id'],'speaker':row['speaker_id'],'ground_truth':words[target],'prediction':words[prediction],'probabilities':output['probabilities'],'display':policy})
    matrix=confusion_matrix(actual,predicted,labels=range(len(words)));np.savetxt(run/'confusion.csv',matrix,delimiter=',',fmt='%d',header=','.join(words))
    report=classification_report(actual,predicted,labels=range(len(words)),target_names=words,output_dict=True,zero_division=0)
    perword={word:{**report[word],'correct':int(matrix[i,i]),'total':int(matrix[i].sum()),'accuracy':float(matrix[i,i]/matrix[i].sum())} for i,word in enumerate(words)}
    pairs=sorted([{'actual':words[i],'predicted':words[j],'count':int(matrix[i,j])} for i in range(len(words)) for j in range(len(words)) if i!=j and matrix[i,j]],key=lambda x:-x['count'])
    # Exact same held-out rows for the baseline; choose majority using TRAIN labels.
    train_counts=Counter(r['word_label'] for r in read_rows(run/'selected.jsonl') if r['split']=='train');majority=sorted(train_counts,key=lambda w:(-train_counts[w],w))[0]
    baseline=sum(r['word_label']==majority for r in rows)/len(rows)
    summary={'task':'MIRACL isolated RGB words; eight unambiguous classes','vocabulary':words,'test_samples':len(rows),
             'top1_accuracy':metrics['accuracy'],'top5_accuracy':top5/len(rows),'single_word_wer':1-metrics['accuracy'],
             'macro_f1':report['macro avg']['f1-score'],'per_word':perword,'confusion_matrix':matrix.tolist(),'confused_pairs':pairs,
             'held_out_speakers':{s:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for s,v in speakers.items()},
             'confidence':{'mean':float(np.mean(conf)),'minimum':min(conf),'maximum':max(conf),'quantiles':np.quantile(conf,[0,.25,.5,.75,1]).tolist()},
             'uncertainty_policy_accepted':accepted,'rejection_thresholds_prespecified':{'confidence':.7,'margin':.15},
             'baseline':{'method':'training-majority word; alphabetical tie break','word':majority,'top1_accuracy':baseline},
             'production_ready':False,'live_activation':False,'oov_detection_validated':False,
             'checkpoint_sha256':sha256(checkpoint),'selected_manifest_sha256':sha256(run/'selected.jsonl'),
             'limitations':['Small prespecified pilot, two repetitions per speaker by default.','Thresholds uncalibrated; no reliable open-set detection established.','No cross-dataset broad-English or continuous-speech result.']}
    write_rows(run/'test_predictions.jsonl',records);atomic_json(run/'evaluation.json',summary)
    # Standalone, offline interactive matrix with exact word labels and counts.
    html='<!doctype html><meta charset="utf-8"><title>MIRACL held-out word confusion</title><style>body{background:#0a0a0c;color:#edebe4;font:16px system-ui;margin:3rem}td,th{padding:.7rem;text-align:center}td{background:#17362e}button{font:inherit}</style><h1>Eight-word MIRACL pilot</h1><p>Rows: actual word. Columns: predicted word. Small held-out-speaker experiment; no deployment claim.</p><label>Highlight counts at least <input type="range" min="0" max="'+str(int(matrix.max()))+'" value="0" oninput="document.querySelectorAll(\'td\').forEach(x=>x.style.opacity=+x.textContent>=+this.value?1:.2)"></label><table><tr><th>Actual / predicted</th>'+''.join('<th>'+w+'</th>' for w in words)+'</tr>'
    for w,counts in zip(words,matrix):html+='<tr><th>'+w+'</th>'+''.join('<td>'+str(int(n))+'</td>' for n in counts)+'</tr>'
    (run/'confusion.html').write_text(html+'</table>',encoding='utf-8')
    return summary


def run_experiment(manifest,run,repetitions=2,epochs=12,minutes=8,resume=False,allow_training_exclusions=False,cache_root=None):
    if epochs<1 or epochs>100 or minutes<=0 or minutes>60:raise ValueError('Bounded epochs and 0..60 minute invocation required.')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    memory=available_gib()
    if memory is not None and memory<.35:raise MemoryError('Less than .35 GiB available after PyTorch import; pilot deferred.')
    rows,words,split_report=select(read_rows(manifest),repetitions)
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    contract={'schema':1,'dataset':'miracl','task':'isolated_word','manifest_sha256':sha256(manifest),'vocabulary':words,
              'preprocessing':PREPROCESS,'landmarker_sha256':sha256(ROOT/'frontend/public/models/face_landmarker.task'),
              'model_source_sha256':sha256(ROOT/'pipeline/word_recognition/model.py'),'seed':42,'repetitions':repetitions,'batch_size':2,'threads':1,
              'allow_training_exclusions':allow_training_exclusions}
    if (run/'contract.json').exists():
        if not resume or json.loads((run/'contract.json').read_text())!=contract:raise ValueError('Protected run; explicit resume and matching contract required.')
        if (run/'evaluation.json').exists():raise FileExistsError('Completed test evaluation is protected; do not repeatedly tune on test.')
    else:
        if any(run.iterdir()):raise FileExistsError('Choose an empty run directory.')
        atomic_json(run/'contract.json',contract);atomic_json(run/'split_inventory.json',split_report)
    rows=prepare(rows,run,contract,cache_root);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    train_rows=[r for r in rows if r['split']=='train'];valid_rows=[r for r in rows if r['split']=='valid'];test_rows=[r for r in rows if r['split']=='test']
    training=loader(train_rows,words,True);validation=loader(valid_rows,words)
    model=WordVideoNet(len(words));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
    # Genuine two-sample gradient smoke, before any optimizer step.
    tensors,labels,smoke_indices=next(iter(training));logits,shapes=model(tensors,return_shapes=True);nn.CrossEntropyLoss()(logits,labels).backward()
    gradients={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
    if not all(v is not None and math.isfinite(v) for v in gradients.values()) or not any(v>0 for v in gradients.values()):raise ValueError('Gradient smoke failed.')
    atomic_json(run/'smoke.json',{'real_ids':[train_rows[int(i)]['id'] for i in smoke_indices],'shapes':shapes,'gradient_norms':gradients,'parameters':sum(p.numel() for p in model.parameters()),'available_gib_after_torch_import':memory})
    best=math.inf;stale=0;history=[];start_epoch=1;start=time.perf_counter()
    if resume and (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if saved['contract']!=contract:raise ValueError('Checkpoint contract mismatch.')
        model.load_state_dict(saved['model_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best_loss'];stale=saved['stale'];start_epoch=saved['epoch']+1
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
    for epoch in range(start_epoch,epochs+1):
        if time.perf_counter()-start>=minutes*60:break
        tick=time.perf_counter();tr,_=pass_epoch(model,training,optimizer);va,_=pass_epoch(model,validation);scheduler.step(va['loss'])
        improved=va['loss']<best
        if improved:best=va['loss'];stale=0
        else:stale+=1
        history.append({'epoch':epoch,'train':tr,'valid':va,'seconds':time.perf_counter()-tick,'lr':optimizer.param_groups[0]['lr']})
        artifact={'schema':1,'model_state_dict':model.state_dict(),'contract':contract,'vocabulary':words,'epoch':epoch,'optimizer':optimizer.state_dict(),
                  'scheduler':scheduler.state_dict(),'history':history,'best_loss':best,'stale':stale,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state()}
        save_checkpoint(run/'last.pt',artifact)
        if improved:save_checkpoint(run/'best.pt',artifact)
        print(f"Epoch {epoch}: train {tr['accuracy']:.4f}, valid {va['accuracy']:.4f}, validation loss {va['loss']:.6f}",flush=True)
        if stale>=4:break
    if not (run/'best.pt').exists():raise RuntimeError('No epoch completed within budget; resume later.')
    saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict'],strict=True)
    atomic_json(run/'training.json',{'finished_at':timestamp(),'history':history,'best_epoch':saved['epoch'],'training_seconds':time.perf_counter()-start,
                                   'selection':'validation loss only','contract':contract,'prespecified_max_epochs':epochs})
    summary=evaluate(model,test_rows,words,run,run/'best.pt')
    print(json.dumps({k:summary[k] for k in ('test_samples','top1_accuracy','top5_accuracy','macro_f1','production_ready')},indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=2);parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--minutes',type=float,default=8);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-training-exclusions',action='store_true');parser.add_argument('--cache-root',type=Path)
    args=parser.parse_args();run_experiment(args.manifest,args.run,args.repetitions,args.epochs,args.minutes,args.resume,args.allow_training_exclusions,args.cache_root)
