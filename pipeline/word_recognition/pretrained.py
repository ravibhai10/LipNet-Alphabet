"""Real MIRACL words -> measured 96px ROIs -> frozen AV-HuBERT -> supervised head.

Stages are resumable and separate. Historical test repetitions 01/02 are
quarantined. Never select a feature layer or classifier using test predictions.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import cv2
import numpy as np
from .common import ROOT,atomic_json,fingerprint,read_rows,write_rows,sha256,timestamp
from pipeline.multidataset.adapters import validate
from tracking.mouth_canonicalization import aligned_mouth_crop
from src.video_training import resample_temporal

ENCODER=ROOT/'models/external/avhubert/base_vox_iter5.pt'
PREPROCESS={'version':'miracl-avhubert-measured96-v1','roi_size':96,'target_frames':29,
            'gray':True,'alignment':'eye-axis; measured mouth center; margin 2.2',
            'source_fps':None,'temporal':'fixed-length linear word normalization; native FPS unknown',
            'crop':88,'mean':.421,'std':.165,'range_before_normalization':[0,1],
            'tracking_coverage':.8,'max_interior_gap':5,'audio_inference':False}
LAYERS=(6,9,12)


def initialize(manifest,run):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    if any(run.iterdir()):raise FileExistsError('Choose a new experiment directory; existing artifacts are protected.')
    all_rows=read_rows(manifest);validate(all_rows)
    words=[r for r in all_rows if r['task']=='isolated_word']
    vocab=sorted({r['word_label'] for r in words})
    if len(vocab)!=8 or len(words)!=1200:raise ValueError('The complete eight-class authentic MIRACL manifest is required.')
    quarantine=[r for r in words if r['split']=='test' and r['repetition']<=2]
    selected=[r for r in words if r not in quarantine]
    # Separate, small multi-word phrase diagnostic; not trained as word classes.
    ood=[r for r in all_rows if r['task']=='phrase' and ((r['split']=='valid' and r['repetition']<=2) or (r['split']=='test' and r['repetition'] in (3,4)))]
    for row in ood:row['role']='phrase_ood'
    for row in selected:row['role']='known_word'
    if Counter(r['split'] for r in selected)!=dict(train=720,valid=240,test=192):raise ValueError('Unexpected frozen split quotas.')
    contract={'schema':1,'created_at':timestamp(),'source_manifest':str(Path(manifest).resolve()),'source_manifest_sha256':sha256(manifest),
              'vocabulary':vocab,'preprocessing':PREPROCESS,'layers':list(LAYERS),'encoder_sha256':sha256(ENCODER),
              'landmarker_sha256':sha256(ROOT/'frontend/public/models/face_landmarker.task'),'seed':42,
              'selection':'known-word validation only; no test feature/head selection',
              'test_policy':'fresh repetitions 03..10 on the original held-out test speakers; prior test speakers were previously evaluated',
              'quarantined_previous_test_ids':[r['id'] for r in quarantine],
              'word_counts':dict(Counter(r['split'] for r in selected)),
              'phrase_ood_counts':dict(Counter(r['split'] for r in ood))}
    write_rows(run/'sources.jsonl',selected+ood);contract['sources_sha256']=sha256(run/'sources.jsonl')
    atomic_json(run/'protocol.json',contract)
    return contract


def contract_for(run):
    run=Path(run);contract=json.loads((run/'protocol.json').read_text())
    if contract['preprocessing']!=PREPROCESS or contract['sources_sha256']!=sha256(run/'sources.jsonl') or contract['source_manifest_sha256']!=sha256(contract['source_manifest']):raise ValueError('Experiment source/preprocessing contract changed.')
    if contract['landmarker_sha256']!=sha256(ROOT/'frontend/public/models/face_landmarker.task'):raise ValueError('Landmarker changed.')
    return contract,read_rows(run/'sources.jsonl')


def source_digest(row):
    digest=hashlib.sha256()
    for name in row['frames']:
        path=Path(name);digest.update(path.name.encode());digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def roi(row,tracker,mp):
    if source_digest(row)!=row['source_sha256']:raise ValueError('Authentic source RGB bytes changed.')
    crops=[];gap=longest=0
    for path in row['frames']:
        frame=cv2.imread(path)
        if frame is None or frame.shape!=(480,640,3):raise ValueError('Invalid RGB input geometry.')
        result=tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        if result.face_landmarks:
            landmarks=np.array([(p.x,p.y,p.z) for p in result.face_landmarks[0]])
            crops.append(cv2.cvtColor(aligned_mouth_crop(frame,landmarks,size=96,margin=2.2),cv2.COLOR_BGR2GRAY));gap=0
        else:
            gap+=1
            if crops:longest=max(gap,longest)
    tracking={'source_frames':len(row['frames']),'tracked_frames':len(crops),'longest_gap':longest}
    if len(crops)<2 or len(crops)<.8*len(row['frames']) or longest>5:
        return None,tracking
    value=np.rint(resample_temporal(crops,29)).clip(0,255).astype(np.uint8)
    return value,tracking


def prepare(run,minutes=30):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    cv2.setNumThreads(1);run=Path(run);contract,rows=contract_for(run)
    directory=run/'rois';directory.mkdir(exist_ok=True);start=time.perf_counter();results=[]
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'frontend/public/models/face_landmarker.task')),
                                        running_mode=vision.RunningMode.IMAGE,num_faces=1,min_face_detection_confidence=.5,min_face_presence_confidence=.5)
    with vision.FaceLandmarker.create_from_options(options) as tracker:
        for index,row in enumerate(rows):
            key=fingerprint({'source':row['source_sha256'],'bounds':[row['start_frame'],row['end_frame']],
                             'preprocessing':PREPROCESS,'landmarker':contract['landmarker_sha256']})
            target=directory/(key+'.npy');receipt=target.with_suffix('.json')
            if receipt.exists():
                saved=json.loads(receipt.read_text())
                if saved['cache_key']!=key or source_digest(row)!=row['source_sha256']:raise ValueError('ROI source/cache mismatch.')
                if saved['usable']:
                    if sha256(target)!=saved['roi_sha256']:raise ValueError('ROI bytes changed.')
                    clip=np.load(target,allow_pickle=False)
                    if clip.shape!=(29,96,96) or clip.dtype!=np.uint8:raise ValueError('ROI shape/type mismatch.')
            else:
                if time.perf_counter()-start>minutes*60:raise TimeoutError('ROI budget reached; completed caches are resumable.')
                clip,tracking=roi(row,tracker,mp)
                saved={'cache_key':key,'usable':clip is not None,'tracking':tracking}
                if clip is not None:
                    if target.exists():raise FileExistsError('Incomplete unidentified ROI cache is protected.')
                    temporary=target.with_suffix('.npy.part')
                    with temporary.open('xb') as stream:np.save(stream,clip,allow_pickle=False)
                    temporary.rename(target);saved['roi_sha256']=sha256(target)
                atomic_json(receipt,saved)
            results.append({**row,**saved,'roi':str(target.resolve())})
            if (index+1)%48==0:print(f'ROI {index+1}/{len(rows)}; usable {sum(r["usable"] for r in results)}',flush=True)
    write_rows(run/'prepared.jsonl',results)
    report={'seconds':time.perf_counter()-start,'samples':len(results),'usable':sum(r['usable'] for r in results),
            'failures':[{'id':r['id'],'split':r['split'],'role':r['role'],'tracking':r['tracking']} for r in results if not r['usable']],
            'policy':'Tracking failures remain in evaluation denominators as rejected/incorrect; no fake crops or replacement clips.'}
    atomic_json(run/'preparation.json',report);print(json.dumps(report,indent=2));return report


def extract(run,minutes=45,limit=None):
    import torch
    from pipeline.avhubert.loader import AVHubertLoader
    run=Path(run);contract,rows=contract_for(run)
    if contract['encoder_sha256']!=sha256(ENCODER):raise ValueError('Encoder fingerprint changed.')
    prepared=read_rows(run/'prepared.jsonl')
    if {r['id'] for r in prepared}!={r['id'] for r in rows}:raise ValueError('Preparation is incomplete.')
    directory=run/'features';directory.mkdir(exist_ok=True);loader=AVHubertLoader(threads=1,low_memory=True).load()
    if loader.preprocessing()['image_mean']!=.421 or loader.preprocessing()['image_std']!=.165 or loader.preprocessing()['image_crop_size']!=88:raise ValueError('Official encoder preprocessing changed.')
    captured={};handles=[]
    def hook(layer):
        def capture(module,args,output):
            value=output[0] if isinstance(output,tuple) else output
            captured[layer]=value[:,0,:].detach().cpu().float().numpy().copy()
        return capture
    for layer in (6,9):handles.append(loader.models[0].encoder.layers[layer-1].register_forward_hook(hook(layer)))
    start=time.perf_counter();results=[];encoded=0;times=[]
    try:
        for index,row in enumerate(prepared):
            result={**row}
            if row['usable']:
                if sha256(row['roi'])!=row['roi_sha256']:raise ValueError('ROI changed before feature extraction.')
                key=fingerprint({'source':row['source_sha256'],'bounds':[row['start_frame'],row['end_frame']],
                                 'roi':row['roi_sha256'],'encoder':contract['encoder_sha256'],'layers':LAYERS,
                                 'preprocessing':PREPROCESS,'landmarker':contract['landmarker_sha256']})
                target=directory/(key+'.npy');receipt=target.with_suffix('.json')
                if receipt.exists():
                    saved=json.loads(receipt.read_text())
                    if saved['cache_key']!=key or saved['sha256']!=sha256(target):raise ValueError('Feature cache fingerprint mismatch.')
                    feature=np.load(target,allow_pickle=False)
                else:
                    if time.perf_counter()-start>minutes*60 or limit is not None and encoded>=limit:
                        atomic_json(run/'extraction_progress.json',{'encoded_this_invocation':encoded,'seconds':time.perf_counter()-start,'sample_seconds':times})
                        print('Feature budget reached; completed cache entries are resumable.',flush=True);return
                    tick=time.perf_counter();captured.clear()
                    final=loader.encode_frames(np.load(row['roi'],allow_pickle=False),source_fps=None,target_frames=29)
                    if set(captured)!={6,9}:raise ValueError('Expected real intermediate transformer states.')
                    feature=np.stack([captured[6],captured[9],final]).astype(np.float32)
                    if feature.shape!=(3,29,768) or not np.isfinite(feature).all():raise ValueError('Invalid pretrained temporal features.')
                    if target.exists():raise FileExistsError('Unidentified feature cache artifact is protected.')
                    temporary=target.with_suffix('.npy.part')
                    with temporary.open('xb') as stream:np.save(stream,feature,allow_pickle=False)
                    temporary.rename(target);saved={'cache_key':key,'sha256':sha256(target)};atomic_json(receipt,saved)
                    times.append(time.perf_counter()-tick);encoded+=1
                if feature.shape!=(3,29,768) or not np.isfinite(feature).all():raise ValueError('Cached feature geometry/finiteness mismatch.')
                result.update(feature=str(target.resolve()),feature_sha256=saved['sha256'])
            results.append(result)
            if (index+1)%48==0:print(f'Features {index+1}/{len(prepared)}, new {encoded}, mean {np.mean(times) if times else 0:.2f}s/clip',flush=True)
    finally:
        for handle in handles:handle.remove()
    write_rows(run/'features.jsonl',results)
    atomic_json(run/'extraction.json',{'seconds':time.perf_counter()-start,'encoded_this_invocation':encoded,'samples':len(results),
                                     'usable':sum(r['usable'] for r in results),'layers':LAYERS,'sample_seconds':times,'audio_input':None})


def pool(feature,kind):
    if kind=='mean':return feature.mean(0)
    if kind=='temporal3':return np.concatenate([part.mean(0) for part in np.array_split(feature,3)])
    if kind=='mean_std':return np.concatenate([feature.mean(0),feature.std(0)])
    raise ValueError('Unknown temporal pooling contract.')


def vectors(rows,words,layer,kind):
    x=[];y=[];kept=[]
    for row in rows:
        if not row['usable']:continue
        if sha256(row['feature'])!=row['feature_sha256']:raise ValueError('Feature bytes changed.')
        f=np.load(row['feature'],allow_pickle=False)
        if f.shape!=(3,29,768) or not np.isfinite(f).all():raise ValueError('Invalid frozen features.')
        x.append(pool(f[LAYERS.index(layer)],kind));y.append(words.index(row['word_label']) if row['role']=='known_word' else -1);kept.append(row)
    return np.stack(x),np.asarray(y),kept


def softmax(logits):
    values=logits-logits.max(axis=1,keepdims=True);p=np.exp(values);return p/p.sum(1,keepdims=True)


def train(run):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    run=Path(run);contract,_=contract_for(run)
    if (run/'selection.json').exists():raise FileExistsError('Selected head is protected; do not tune after test access.')
    rows=read_rows(run/'features.jsonl');words=contract['vocabulary']
    training=[r for r in rows if r['split']=='train' and r['role']=='known_word'];validation=[r for r in rows if r['split']=='valid' and r['role']=='known_word']
    if set(r['word_label'] for r in training if r['usable'])!=set(words):raise ValueError('Training vocabulary incomplete after tracking.')
    candidates=[];best=None;started=time.perf_counter()
    with threadpool_limits(limits=1):
        for layer in LAYERS:
            for kind in ('mean','temporal3','mean_std'):
                x,y,_=vectors(training,words,layer,kind);v,z,valid_rows=vectors(validation,words,layer,kind)
                scale=StandardScaler().fit(x);x=scale.transform(x);v=scale.transform(v)
                for regularization in (.01,.1,1.):
                    classifier=LogisticRegression(C=regularization,solver='lbfgs',max_iter=600,class_weight='balanced',random_state=42)
                    classifier.fit(x,y)
                    p=classifier.predict_proba(v)
                    if classifier.classes_.tolist()!=list(range(len(words))):raise ValueError('Classifier output order mismatch.')
                    correct=int((p.argmax(1)==z).sum());accuracy=correct/len(validation)
                    loss=float(-np.log(p[np.arange(len(z)),z].clip(1e-12)).mean())
                    record={'layer':layer,'pooling':kind,'C':regularization,'validation_accuracy':accuracy,'validation_correct':correct,'validation_total':len(validation),'validation_loss_usable':loss,'iterations':classifier.n_iter_.tolist()}
                    candidates.append(record);print(json.dumps(record),flush=True)
                    # Fixed lexicographic selection: accuracy incl tracking failures, then CE.
                    if best is None or (accuracy,-loss)>(best['record']['validation_accuracy'],-best['record']['validation_loss_usable']):
                        best={'record':record,'mean':scale.mean_.copy(),'scale':scale.scale_.copy(),'weight':classifier.coef_.copy(),'bias':classifier.intercept_.copy()}
    record=best['record'];artifact={'format':'lipnet_avhubert_word_linear_v1','contract':contract,'feature_layer':record['layer'],'pooling':record['pooling'],
                                 'vocabulary':words,'mean':torch.tensor(best['mean'],dtype=torch.float32),'scale':torch.tensor(best['scale'],dtype=torch.float32),
                                 'weight':torch.tensor(best['weight'],dtype=torch.float32),'bias':torch.tensor(best['bias'],dtype=torch.float32),
                                 'selection':record,'audio_inference':False}
    torch.save(artifact,run/'head.pt')
    selection={'selected':record,'candidates':candidates,'seconds':time.perf_counter()-started,'test_used':False,
               'head_sha256':sha256(run/'head.pt'),'features_manifest_sha256':sha256(run/'features.jsonl'),
               'training_samples':len(training),'training_usable':sum(r['usable'] for r in training),'validation_samples':len(validation),
               'limitations':'Frozen encoder, finite closed vocabulary. Webcam/open-set quality not established.'}
    atomic_json(run/'selection.json',selection);return selection


def predict_features(head,features):
    words=head['vocabulary'];layer=head['feature_layer'];kind=head['pooling']
    return sequence_probabilities(head,features[LAYERS.index(layer)])


def sequence_probabilities(head,feature):
    words=head['vocabulary'];kind=head['pooling']
    if feature.shape!=(29,768) or not np.isfinite(feature).all():raise ValueError('Finite [29,768] real encoder sequence required.')
    value=pool(feature,kind)
    mean=head['mean'].numpy();scale=head['scale'].numpy();weight=head['weight'].numpy();bias=head['bias'].numpy()
    if weight.shape!=(len(words),len(value)) or mean.shape!=value.shape or scale.shape!=value.shape or np.any(scale<=0):raise ValueError('Classifier/normalization geometry mismatch.')
    return softmax((((value-mean)/scale)@weight.T+bias)[None])[0]


def evaluate(run):
    import torch
    from sklearn.metrics import classification_report,confusion_matrix
    run=Path(run);contract,_=contract_for(run);selection=json.loads((run/'selection.json').read_text())
    if (run/'evaluation.json').exists():raise FileExistsError('Held-out evaluation is protected from repeated model selection.')
    if selection['head_sha256']!=sha256(run/'head.pt') or selection['features_manifest_sha256']!=sha256(run/'features.jsonl'):raise ValueError('Selected model/features changed.')
    head=torch.load(run/'head.pt',map_location='cpu',weights_only=True)
    if head['contract']!=contract:raise ValueError('Head/source contract mismatch.')
    rows=read_rows(run/'features.jsonl');words=contract['vocabulary']
    # Validation-only confidence policy; no test scores used to choose thresholds.
    valid=[r for r in rows if r['split']=='valid'];vknown=[r for r in valid if r['role']=='known_word'];vood=[r for r in valid if r['role']=='phrase_ood']
    def scores(group):
        result=[]
        for row in group:
            if not row['usable']:result.append((row,None));continue
            if sha256(row['feature'])!=row['feature_sha256']:raise ValueError('Features changed at evaluation.')
            result.append((row,predict_features(head,np.load(row['feature'],allow_pickle=False))))
        return result
    def accept(p,threshold,margin):
        if p is None:return False
        order=np.sort(p);return bool(order[-1]>=threshold and order[-1]-order[-2]>=margin)
    vk,vu=scores(vknown),scores(vood);policies=[]
    for threshold in (.5,.6,.7,.8,.9,.95,.99):
        for margin in (0.,.1,.2):
            good=sum(accept(p,threshold,margin) and words[int(p.argmax())]==r['word_label'] for r,p in vk)
            bad=sum(accept(p,threshold,margin) and words[int(p.argmax())]!=r['word_label'] for r,p in vk)
            false_accept=sum(accept(p,threshold,margin) for r,p in vu)
            utility=(good-2*bad)/len(vk)-2*false_accept/len(vu)
            policies.append((utility,threshold,margin))
    _,threshold,margin=max(policies)
    atomic_json(run/'rejection_policy.json',{'head_sha256':selection['head_sha256'],'threshold':threshold,'margin':margin,
                                           'source':'validation known-word and phrase samples only','general_oov_detection_validated':False})
    test=[r for r in rows if r['split']=='test' and r['role']=='known_word'];test_ood=[r for r in rows if r['split']=='test' and r['role']=='phrase_ood']
    predictions=[];correct=top5=accepted=accepted_correct=0;actual=[];guessed=[];confidence=[];speakers=defaultdict(lambda:[0,0]);matrix=np.zeros((len(words),len(words)),dtype=int);failures=Counter()
    for row,p in scores(test):
        target=words.index(row['word_label']);index=int(p.argmax()) if p is not None else -1
        ok=index==target;display=accept(p,threshold,margin);correct+=int(ok);accepted+=int(display);accepted_correct+=int(display and ok)
        speakers[row['speaker_id']][0]+=int(ok);speakers[row['speaker_id']][1]+=1
        if p is not None:
            top5+=int(target in np.argsort(-p)[:5]);confidence.append(float(p.max()));matrix[target,index]+=1
        else:failures[row['word_label']]+=1
        actual.append(target);guessed.append(index)
        predictions.append({'id':row['id'],'speaker':row['speaker_id'],'ground_truth':row['word_label'],'candidate':words[index] if index>=0 else None,
                            'word':words[index] if display else None,'uncertain':not display,'confidence':float(p.max()) if p is not None else None,
                            'probabilities':p.tolist() if p is not None else None,'correct_raw':ok,'tracking_usable':row['usable']})
    ood_false=sum(accept(p,threshold,margin) for r,p in scores(test_ood))
    report=classification_report(actual,guessed,labels=list(range(len(words))),target_names=words,output_dict=True,zero_division=0)
    perword={w:{**report[w],'correct':int(matrix[i,i]),'total':sum(r['word_label']==w for r in test),'tracking_failures':failures[w]} for i,w in enumerate(words)}
    summary={'task':'eight-class MIRACL silent whole-word recognition','vocabulary':words,'test_samples':len(test),'correct':correct,'top1_accuracy':correct/len(test),
             'top5_accuracy':top5/len(test),'macro_f1':report['macro avg']['f1-score'],'single_word_wer':1-correct/len(test),'per_word':perword,'classification_report':report,
             'confusion_matrix':matrix.tolist(),'matrix_policy':'Rows actual, columns predicted; tracking failures reported separately and included as errors in all totals.',
             'speakers':{s:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for s,v in speakers.items()},
             'confidence':{'mean':float(np.mean(confidence)),'min':min(confidence),'max':max(confidence)},
             'rejection_policy':{'threshold':threshold,'margin':margin,'selection':'validation known-word correctness and phrase OOD only; prespecified utility',
                                 'known_test_accepted':accepted,'known_test_coverage':accepted/len(test),'accepted_correct':accepted_correct,'accepted_accuracy':accepted_correct/accepted if accepted else None,
                                 'phrase_ood_test_samples':len(test_ood),'phrase_ood_false_accepts':ood_false,'phrase_ood_false_accept_rate':ood_false/len(test_ood),
                                 'limitation':'Multi-word phrase diagnostic only; arbitrary unseen isolated-word rejection is not established.'},
             'selected_head':selection['selected'],'head_sha256':selection['head_sha256'],'encoder_sha256':contract['encoder_sha256'],
             'test_policy':contract['test_policy'],'historical_test_clips_quarantined':48,'audio_inference':False,'production_ready':False}
    write_rows(run/'test_predictions.jsonl',predictions);np.savetxt(run/'confusion.csv',matrix,fmt='%d',delimiter=',',header=','.join(words))
    atomic_json(run/'evaluation.json',summary);print(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('init','prepare','extract','train','evaluate'));p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/processed/multidataset/miracl_v1.jsonl');p.add_argument('--minutes',type=float,default=30);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if not 0<args.minutes<=120:p.error('Choose a positive stage budget up to 120 minutes.')
    if args.stage=='init':print(initialize(args.manifest,args.run))
    elif args.stage=='prepare':prepare(args.run,args.minutes)
    elif args.stage=='extract':extract(args.run,args.minutes,args.limit)
    elif args.stage=='train':print(train(args.run))
    elif args.stage=='evaluate':evaluate(args.run)


if __name__=='__main__':main()
