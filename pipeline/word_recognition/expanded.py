"""Twenty real word classes: MIRACL plus all labeled allenye66 sequences.

Separate run, frozen encoder, validation-only selection, one protected test.
Upstream speaker identities are unknown: its split is clip-disjoint, NOT
speaker-independent. Existing MIRACL artifacts and live eight-word head remain.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .common import ROOT,atomic_json,fingerprint,read_rows,sha256,timestamp,write_rows
from .expanded_data import ADAPTER,BASE_RUN,COMMIT,SOURCE,WORDS,adapt_sequence
from .pretrained import ENCODER,LAYERS,PREPROCESS,pool,predict_features

DEFAULT_RUN=ROOT/'runs/word_accuracy/avhubert_combined20_v1'
FORMAT='lipnet_avhubert_expanded_word_linear_v1'


def feature_key(roi_sha,encoder_sha):
    return fingerprint({'roi':roi_sha,'adapter':ADAPTER,'encoder':encoder_sha,'layers':LAYERS,'preprocessing':PREPROCESS})


def precache(source=SOURCE,minutes=90):
    """Label-independent frozen extraction while verified downloads arrive.

    No head, predictions or split selection is performed here. Only source
    receipts completed atomically by the downloader enter this cache.
    """
    from pipeline.avhubert.loader import AVHubertLoader
    source=Path(source);directory=source/'frozen_features';directory.mkdir(parents=True,exist_ok=True)
    encoder_sha=sha256(ENCODER);loader=AVHubertLoader(threads=1,low_memory=True).load();captured={};handles=[]
    def hook(layer):
        def capture(module,args,output):
            value=output[0] if isinstance(output,tuple) else output
            captured[layer]=value[:,0,:].detach().cpu().float().numpy().copy()
        return capture
    for layer in (6,9):handles.append(loader.models[0].encoder.layers[layer-1].register_forward_hook(hook(layer)))
    done=set();start=time.perf_counter();times=[]
    try:
        while True:
            receipts=sorted((source/'original').glob('*/receipt.json'));pending=[p for p in receipts if p.parent.name not in done]
            for receipt in pending:
                row=json.loads(receipt.read_text())
                if sha256(row['array'])!=row['array_sha256']:raise ValueError('Verified publisher array changed.')
                clip=adapt_sequence(np.load(row['array'],allow_pickle=False))
                # NPY headers participate in the cache key, matching prepare().
                import io
                buffer=io.BytesIO();np.save(buffer,clip,allow_pickle=False)
                digest=hashlib.sha256(buffer.getvalue()).hexdigest();key=feature_key(digest,encoder_sha)
                target=directory/(key+'.npy');meta=target.with_suffix('.json')
                if meta.exists():
                    saved=json.loads(meta.read_text())
                    if saved['cache_key']!=key or sha256(target)!=saved['sha256']:raise ValueError('Frozen pre-cache changed.')
                else:
                    if time.perf_counter()-start>minutes*60:raise TimeoutError('Pre-cache stage budget reached; restart safely to resume.')
                    tick=time.perf_counter();captured.clear();last=loader.encode_frames(clip,source_fps=None,target_frames=29)
                    feature=np.stack([captured[6],captured[9],last]).astype(np.float32)
                    if feature.shape!=(3,29,768) or not np.isfinite(feature).all():raise ValueError('Invalid real frozen feature sequence.')
                    with target.open('xb') as stream:np.save(stream,feature,allow_pickle=False)
                    saved={'cache_key':key,'sha256':sha256(target),'seconds':time.perf_counter()-tick,'source_id':row['id'],
                           'source_array_sha256':row['array_sha256'],'encoder_sha256':encoder_sha,'roi_sha256':digest,'layers':list(LAYERS)}
                    atomic_json(meta,saved);times.append(saved['seconds'])
                done.add(receipt.parent.name)
                if len(done)%20==0:
                    progress={'verified_sequences_encoded':len(done),'new':len(times),'seconds':time.perf_counter()-start,
                              'mean_new_seconds':float(np.mean(times)) if times else None,'audio_used':False,'predictions_made':False}
                    atomic_json(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
            if (source/'acquisition.json').exists():
                acquisition=json.loads((source/'acquisition.json').read_text())
                if acquisition['errors']:raise RuntimeError('Acquisition incomplete; fix/resume before final training.')
                if len(done)==acquisition['acquired']:break
            if time.perf_counter()-start>minutes*60:raise TimeoutError('Pre-cache wait budget reached; resumable sources/features retained.')
            time.sleep(3)
    finally:
        for handle in handles:handle.remove()
    report={'sequences':len(done),'new':len(times),'seconds':time.perf_counter()-start,'sample_seconds':times,'encoder_sha256':encoder_sha,'predictions_made':False}
    atomic_json(directory/'complete.json',report);return {k:v for k,v in report.items() if k!='sample_seconds'}


def split_groups(rows):
    """Prespecified blocked index split; exact duplicated content stays together."""
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs cannot identify distinct utterances.')
    by_pixels=defaultdict(list)
    for row in rows:
        if row['word_label'] is not None:by_pixels[row['pixels_sha256']].append(row)
    grouped=defaultdict(list);duplicates=[]
    for digest,group in by_pixels.items():
        if len({r['word_label'] for r in group})!=1:
            raise ValueError('Conflicting labels for identical original pixels; adjudication required.')
        group.sort(key=lambda r:int(r['id'].rsplit('_',1)[1]));row=dict(group[0])
        row['duplicate_aliases']=[r['id'] for r in group[1:]]
        if len(group)>1:duplicates.append({'canonical':row['id'],'aliases':row['duplicate_aliases'],'pixels_sha256':digest})
        grouped[row['word_label']].append(row)
    selected=[]
    for word,group in sorted(grouped.items()):
        group.sort(key=lambda r:int(r['id'].rsplit('_',1)[1]))
        if len(group)<20:raise ValueError('Insufficient independent clips for a target word.')
        train_end=int(.7*len(group));valid_end=train_end+int(.15*len(group))
        for i,row in enumerate(group):
            selected.append({**row,'split':'train' if i<train_end else 'valid' if i<valid_end else 'test'})
    return selected,duplicates


def initialize(run=DEFAULT_RUN,source=SOURCE):
    run=Path(run);source=Path(source)
    if run.exists() and any(run.iterdir()):raise FileExistsError('Choose a new run; completed selection/evaluation is protected.')
    acquired=json.loads((source/'acquisition.json').read_text())
    inventory=read_rows(source/'inventory.jsonl')
    if acquired['errors'] or acquired['acquired']!=685 or len(inventory)!=685 or acquired['commit']!=COMMIT:
        raise ValueError('Complete pinned acquisition required, including unresolved rows.')
    rows,duplicates=split_groups(inventory)
    for row in rows:
        row.update(role='known_word',usable=True,input_kind='publisher_bgr_mouth_array',start_frame=0,end_frame=22)
    base_features=BASE_RUN/'features.jsonl'
    for base in read_rows(base_features):
        if base['role']!='known_word':continue
        rows.append({**base,'dataset':'miracl','input_kind':'measured_aligned96',
                     'pixels_sha256':base.get('roi_sha256'),'speaker_id':'miracl:'+base['speaker_id']})
    if tuple(sorted({r['word_label'] for r in rows}))!=WORDS:raise ValueError('Actual twenty-word union changed.')
    run.mkdir(parents=True,exist_ok=True);write_rows(run/'sources.jsonl',rows)
    contract={'schema':1,'task':'twenty_class_combined_isolated_words','created_at':timestamp(),
              'vocabulary':list(WORDS),'preprocessing':PREPROCESS,'source_adapters':{'allenye66':ADAPTER,'miracl':PREPROCESS},
              'encoder_sha256':sha256(ENCODER),'landmarker_sha256':sha256(ROOT/'frontend/public/models/face_landmarker.task'),
              'upstream_commit':COMMIT,'source_inventory':str((source/'inventory.jsonl').resolve()),
              'source_inventory_sha256':sha256(source/'inventory.jsonl'),
              'base_features_manifest':str(base_features.resolve()),'base_features_manifest_sha256':sha256(base_features),
              'sources_sha256':sha256(run/'sources.jsonl'),'layers':list(LAYERS),
              'upstream_split':'per-class ascending recording index, first floor(70%) train, next floor(15%) valid, remainder test; exact content groups indivisible',
              'speaker_independence':'MIRACL retains original speaker-disjoint partitions and pilot quarantine; upstream speakers/sessions unknown, clip-disjoint only',
              'selection':'maximize validation macro recall over all 20 classes; tie-break validation cross-entropy; no test-based selection',
              'search':{'layers':list(LAYERS),'pooling':['mean','temporal3','mean_std'],'C':[.01,.1,1.,10.]},
              'rejection':'validation known-word accuracy/coverage only; no claim of calibrated OOV detection',
              'duplicates':duplicates,'quarantined_unresolved_ids':[r['id'] for r in inventory if r['word_label'] is None],
              'counts':{s:dict(Counter(r['word_label'] for r in rows if r['split']==s)) for s in ('train','valid','test')},
              'historical_test_note':'MIRACL test speaker identities and earlier results already observed; no new population claim.',
              'audio_inference':False,'encoder_finetuned':False}
    atomic_json(run/'protocol.json',contract);return contract


def contract_for(run):
    run=Path(run);c=json.loads((run/'protocol.json').read_text())
    if c['vocabulary']!=list(WORDS) or c['preprocessing']!=PREPROCESS or c['source_adapters']!={'allenye66':ADAPTER,'miracl':PREPROCESS}:
        raise ValueError('Experiment vocabulary/preprocessing changed.')
    for key,file in (('sources_sha256',run/'sources.jsonl'),('source_inventory_sha256',c['source_inventory']),('base_features_manifest_sha256',c['base_features_manifest'])):
        if sha256(file)!=c[key]:raise ValueError('Frozen source manifest changed.')
    return c,read_rows(run/'sources.jsonl')


def prepare(run):
    cv2.setNumThreads(1);run=Path(run);c,rows=contract_for(run)
    directory=run/'rois';directory.mkdir(exist_ok=True);out=[];quality=[];start=time.perf_counter()
    for row in rows:
        if row['dataset']=='miracl':
            if row['usable'] and sha256(row['roi'])!=row['roi_sha256']:raise ValueError('Preserved MIRACL ROI changed.')
            out.append(row);continue
        if sha256(row['source_path'])!=row['source_sha256'] or sha256(row['array'])!=row['array_sha256']:
            raise ValueError('Publisher source changed.')
        value=np.load(row['array'],allow_pickle=False)
        if hashlib.sha256(value.tobytes()).hexdigest()!=row['pixels_sha256']:raise ValueError('Decoded source pixels changed.')
        clip=adapt_sequence(value);target=directory/(row['id'].split(':')[1]+'.npy')
        if target.exists():
            if not np.array_equal(np.load(target,allow_pickle=False),clip):raise ValueError('Protected prepared sequence changed.')
        else:
            with target.open('xb') as stream:np.save(stream,clip,allow_pickle=False)
        motion=float(np.abs(np.diff(clip.astype(np.float32),axis=0)).mean())
        q={'id':row['id'],'mean':float(clip.mean()),'std':float(clip.std()),'mean_frame_change':motion}
        quality.append(q);out.append({**row,'roi':str(target.resolve()),'roi_sha256':sha256(target),'quality':q,'alignment_measured':False})
    write_rows(run/'prepared.jsonl',out)
    report={'seconds':time.perf_counter()-start,'publisher_sequences':len(quality),'quality':quality,
            'policy':'All integrity-valid labeled utterances retained; low movement is diagnostic, not a label-dependent exclusion.',
            'publisher_native_geometry':[22,80,112,3],'encoder_roi_geometry':[29,96,96],'adapter':ADAPTER,
            'miracl_tracking_failures':sum(not r['usable'] for r in out if r['dataset']=='miracl')}
    atomic_json(run/'preparation.json',report);return {k:v for k,v in report.items() if k!='quality'}


def extract(run,minutes=90):
    from pipeline.avhubert.loader import AVHubertLoader
    run=Path(run);c,sources=contract_for(run);rows=read_rows(run/'prepared.jsonl')
    if {r['id'] for r in rows}!={r['id'] for r in sources}:raise ValueError('Incomplete preparation.')
    if sha256(ENCODER)!=c['encoder_sha256']:raise ValueError('Encoder changed.')
    loader=AVHubertLoader(threads=1,low_memory=True).load();captured={};handles=[]
    def hook(layer):
        def capture(module,args,output):
            value=output[0] if isinstance(output,tuple) else output
            captured[layer]=value[:,0,:].detach().cpu().float().numpy().copy()
        return capture
    for layer in (6,9):handles.append(loader.models[0].encoder.layers[layer-1].register_forward_hook(hook(layer)))
    directory=run/'features';directory.mkdir(exist_ok=True);out=[];times=[];start=time.perf_counter()
    try:
        for i,row in enumerate(rows):
            if row['dataset']=='miracl':
                if row['usable']:checked_feature(row)
                out.append(row);continue
            key=feature_key(row['roi_sha256'],c['encoder_sha256'])
            target=directory/(key+'.npy');receipt=target.with_suffix('.json')
            shared=Path(c['source_inventory']).parent/'frozen_features'/(key+'.npy')
            if not receipt.exists() and shared.with_suffix('.json').exists():
                target=shared;receipt=shared.with_suffix('.json')
            if receipt.exists():
                saved=json.loads(receipt.read_text())
                if saved['cache_key']!=key or sha256(target)!=saved['sha256']:raise ValueError('Feature cache changed.')
            else:
                if time.perf_counter()-start>minutes*60:raise TimeoutError('Feature stage budget reached; all completed cache entries are resumable.')
                if sha256(row['roi'])!=row['roi_sha256']:raise ValueError('Prepared pixels changed.')
                tick=time.perf_counter();captured.clear()
                last=loader.encode_frames(np.load(row['roi'],allow_pickle=False),source_fps=None,target_frames=29)
                f=np.stack([captured[6],captured[9],last]).astype(np.float32)
                if f.shape!=(3,29,768) or not np.isfinite(f).all():raise ValueError('Bad real encoder features.')
                with target.open('xb') as stream:np.save(stream,f,allow_pickle=False)
                saved={'cache_key':key,'sha256':sha256(target),'seconds':time.perf_counter()-tick}
                atomic_json(receipt,saved);times.append(saved['seconds'])
            out.append({**row,'feature':str(target.resolve()),'feature_sha256':saved['sha256']})
            if (i+1)%20==0:
                print(f'Real encoder {i+1}/{len(rows)}, new {len(times)}, mean {np.mean(times) if times else 0:.2f}s',flush=True)
                atomic_json(run/'extraction_progress.json',{'completed_upstream':i+1,'new':len(times),'seconds':time.perf_counter()-start})
    finally:
        for handle in handles:handle.remove()
    write_rows(run/'features.jsonl',out)
    report={'seconds':time.perf_counter()-start,'new_sequences':len(times),'reused_miracl_sequences':sum(r['dataset']=='miracl' and r['usable'] for r in out),
            'all_rows':len(out),'layers':list(LAYERS),'sample_seconds':times,'audio_used':False}
    atomic_json(run/'extraction.json',report);return {k:v for k,v in report.items() if k!='sample_seconds'}


def checked_feature(row):
    if sha256(row['feature'])!=row['feature_sha256']:raise ValueError('Frozen feature bytes changed.')
    f=np.load(row['feature'],allow_pickle=False)
    if f.shape!=(3,29,768) or not np.isfinite(f).all():raise ValueError('Real finite 3x29x768 features required.')
    return f


def validate_derived(sources, derived):
    """Derived artifacts may add fields but cannot change frozen sample identity."""
    original = {r['id']: r for r in sources}
    if len(original) != len(sources) or len({r['id'] for r in derived}) != len(derived):
        raise ValueError('Duplicate IDs in frozen or derived manifest.')
    if {r['id'] for r in derived} != set(original):
        raise ValueError('Derived manifest must contain every frozen source exactly once.')
    for row in derived:
        source = original[row['id']]
        if any(key not in row or row[key] != value for key, value in source.items()):
            raise ValueError('Derived manifest changed frozen source identity or labels.')


def load_features(run):
    c,sources=contract_for(run);rows=read_rows(Path(run)/'features.jsonl')
    validate_derived(sources, rows)
    features={r['id']:checked_feature(r) for r in rows if r['usable']}
    return c,rows,features


def train(run):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    run=Path(run)
    if any((run/n).exists() for n in ('head.pt','selection.json','test_access.json','evaluation.json')):
        raise FileExistsError('Selected/evaluated experiment protected; test scores cannot guide retuning.')
    c,rows,f=load_features(run);training=[r for r in rows if r['split']=='train' and r['usable']];valid=[r for r in rows if r['split']=='valid']
    usable_valid=[r for r in valid if r['usable']];y=np.array([WORDS.index(r['word_label']) for r in training]);z=np.array([WORDS.index(r['word_label']) for r in usable_valid])
    if set(y)!=set(range(20)):raise ValueError('Incomplete twenty-class training set.')
    totals=np.array([sum(r['word_label']==w for r in valid) for w in WORDS]);candidates=[];best=None;start=time.perf_counter()
    with threadpool_limits(limits=1):
        for layer in c['search']['layers']:
            for kind in c['search']['pooling']:
                x=np.stack([pool(f[r['id']][LAYERS.index(layer)],kind) for r in training])
                vx=np.stack([pool(f[r['id']][LAYERS.index(layer)],kind) for r in usable_valid])
                scaler=StandardScaler().fit(x);x=scaler.transform(x);vx=scaler.transform(vx)
                for regularization in c['search']['C']:
                    model=LogisticRegression(C=regularization,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=42).fit(x,y)
                    if model.classes_.tolist()!=list(range(20)):raise ValueError('Model output mapping differs from vocabulary.')
                    p=model.predict_proba(vx);correct=p.argmax(1)==z
                    recalls=np.bincount(z[correct],minlength=20)/totals
                    loss=float(-np.log(p[np.arange(len(z)),z].clip(1e-12)).mean())
                    record={'layer':layer,'pooling':kind,'C':regularization,'validation_macro_recall':float(recalls.mean()),
                            'validation_correct':int(correct.sum()),'validation_total':len(valid),'validation_accuracy':float(correct.sum()/len(valid)),
                            'validation_log_loss':loss,'minimum_class_recall':float(recalls.min()),'iterations':model.n_iter_.tolist()}
                    candidates.append(record);print(json.dumps(record),flush=True)
                    if best is None or (record['validation_macro_recall'],-loss)>(best['record']['validation_macro_recall'],-best['record']['validation_log_loss']):
                        best={'record':record,'mean':scaler.mean_.copy(),'scale':scaler.scale_.copy(),'weight':model.coef_.copy(),'bias':model.intercept_.copy()}
    record=best['record'];head={'format':FORMAT,'contract':c,'vocabulary':list(WORDS),'feature_layer':record['layer'],'pooling':record['pooling'],
                              'selection':record,'audio_inference':False}
    for key in ('mean','scale','weight','bias'):head[key]=torch.tensor(best[key],dtype=torch.float32)
    torch.save(head,run/'head.pt')
    selection={'selected':record,'candidates':candidates,'head_sha256':sha256(run/'head.pt'),'features_manifest_sha256':sha256(run/'features.jsonl'),
               'training_samples':len(training),'validation_samples':len(valid),'seconds':time.perf_counter()-start,'test_used':False,
               'encoder_finetuned':False,'learned_head_parameters':int(head['weight'].numel()+head['bias'].numel())}
    # Freeze a conservative known-word-only rejection policy before opening test.
    p=np.stack([predict_features(head,f[r['id']]) for r in usable_valid]);target=np.array([WORDS.index(r['word_label']) for r in usable_valid])
    policies=[]
    for threshold in (.5,.6,.7,.8,.9,.95,.99):
        for margin in (.1,.2,.3):
            sorted_p=np.sort(p,axis=1);accepted=(sorted_p[:,-1]>=threshold)&(sorted_p[:,-1]-sorted_p[:,-2]>=margin)
            right=(p.argmax(1)==target);good=int((accepted&right).sum());wrong=int((accepted&~right).sum())
            policies.append({'threshold':threshold,'margin':margin,'validation_accepted_correct':good,'validation_accepted_wrong':wrong,'utility':(good-3*wrong)/len(valid)})
    policy=max(policies,key=lambda r:(r['utility'],r['threshold'],r['margin']))
    atomic_json(run/'rejection_policy.json',{**policy,'head_sha256':selection['head_sha256'],'source':'known-word validation only; no test predictions',
                                             'general_oov_detection_validated':False,'webcam_accuracy_validated':False})
    atomic_json(run/'selection.json',selection);return selection['selected']


def evaluate(run):
    import torch
    from sklearn.metrics import classification_report,confusion_matrix
    run=Path(run)
    if (run/'test_access.json').exists() or (run/'evaluation.json').exists():raise FileExistsError('Held-out evaluation already opened; no repeated selection.')
    c,rows,f=load_features(run);selection=json.loads((run/'selection.json').read_text())
    if sha256(run/'head.pt')!=selection['head_sha256'] or sha256(run/'features.jsonl')!=selection['features_manifest_sha256']:
        raise ValueError('Selected model/features changed before locked evaluation.')
    head=torch.load(run/'head.pt',map_location='cpu',weights_only=True)
    if head['contract']!=c:raise ValueError('Head contract differs from frozen experiment.')
    policy=json.loads((run/'rejection_policy.json').read_text());test=[r for r in rows if r['split']=='test']
    atomic_json(run/'test_access.json',{'at':timestamp(),'selection_sha256':sha256(run/'selection.json'),'head_sha256':selection['head_sha256'],
                                      'policy_sha256':sha256(run/'rejection_policy.json'),'policy':'Model selection complete before final test inference.'})
    predictions=[];correct=top5=accepted=accepted_correct=0;actual=[];guessed=[];sources=defaultdict(lambda:[0,0]);speakers=defaultdict(lambda:[0,0])
    for row in test:
        p=predict_features(head,f[row['id']]) if row['usable'] else None
        target=WORDS.index(row['word_label']);index=int(p.argmax()) if p is not None else -1
        ok=target==index;actual.append(target);guessed.append(index);correct+=int(ok)
        order=np.argsort(-p) if p is not None else [];conf=float(p[index]) if p is not None else None
        margin=float(p[order[0]]-p[order[1]]) if p is not None else 0
        accept=p is not None and conf>=policy['threshold'] and margin>=policy['margin']
        top5+=int(target in order[:5]);accepted+=int(accept);accepted_correct+=int(accept and ok)
        sources[row['dataset']][0]+=int(ok);sources[row['dataset']][1]+=1
        if row.get('speaker_id'):speakers[row['speaker_id']][0]+=int(ok);speakers[row['speaker_id']][1]+=1
        predictions.append({'id':row['id'],'dataset':row['dataset'],'speaker_id':row.get('speaker_id'),'ground_truth':row['word_label'],
                            'candidate':WORDS[index] if index>=0 else None,'word':WORDS[index] if accept else None,'uncertain':not accept,
                            'correct_raw':bool(ok),'probabilities':p.tolist() if p is not None else None,'confidence':conf,'tracking_usable':row['usable']})
    report=classification_report(actual,guessed,labels=list(range(20)),target_names=list(WORDS),output_dict=True,zero_division=0)
    matrix=confusion_matrix(actual,guessed,labels=list(range(20)))
    summary={'task':'twenty-class combined silent isolated-word recognition','protocol':'combined20_blocked_clip_split_v1',
             'vocabulary':list(WORDS),'test_samples':len(test),'correct':correct,'top1_accuracy':correct/len(test),'top5_accuracy':top5/len(test),
             'macro_f1':report['macro avg']['f1-score'],'macro_recall':report['macro avg']['recall'],'single_word_wer':1-correct/len(test),
             'classification_report':report,'per_word':{w:{**report[w],'correct':int(matrix[i,i]),'total':int(sum(a==i for a in actual))} for i,w in enumerate(WORDS)},
             'confusion_matrix':matrix.tolist(),'matrix_policy':'actual rows / predicted columns; failed inputs count as errors',
             'by_source':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in sources.items()},
             'known_speakers':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in speakers.items()},
             'rejection':{'threshold':policy['threshold'],'margin':policy['margin'],'accepted':accepted,'accepted_correct':accepted_correct,
                          'coverage':accepted/len(test),'accepted_accuracy':accepted_correct/accepted if accepted else None,
                          'oov_validated':False},
             'head_sha256':selection['head_sha256'],'encoder_sha256':c['encoder_sha256'],'selected_head':selection['selected'],
             'encoder_finetuned':False,'audio_inference':False,'webcam_accuracy_validated':False,'production_ready':False,
             'limitations':[c['speaker_independence'],c['historical_test_note'],
                            'Two datasets differ in camera, speakers, crop and filtering; source-domain cues can inflate combined closed-set performance.',
                            'Publisher mouth crops lack original full-face eye landmarks; reflected letterbox is an explicit approximation.',
                            'No unseen isolated-word, arbitrary English, continuous-sentence or physical-webcam evaluation.']}
    summary['eligible_for_manual_research_activation']=summary['top1_accuracy']>=.7 and min(v['recall'] for v in summary['per_word'].values())>=.2
    write_rows(run/'test_predictions.jsonl',predictions);np.savetxt(run/'confusion.csv',matrix,fmt='%d',delimiter=',',header=','.join(WORDS))
    atomic_json(run/'evaluation.json',summary);return summary


def plots(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);e=json.loads((run/'evaluation.json').read_text());selection=json.loads((run/'selection.json').read_text())
    plt.rcParams.update({'font.size':10,'savefig.dpi':180})
    matrix=np.asarray(e['confusion_matrix']);fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(20),yticks=range(20),xticklabels=WORDS,yticklabels=WORDS,xlabel='Predicted word',ylabel='Actual word',title=f'Combined 20-word test: {e["correct"]}/{e["test_samples"]} correct')
    plt.setp(ax.get_xticklabels(),rotation=60,ha='right')
    for i in range(20):
        for j in range(20):
            if matrix[i,j]:ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    fig.colorbar(im,ax=ax,label='Utterances');fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(run/f'confusion.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5));recalls=[e['per_word'][w]['recall']*100 for w in WORDS]
    axes[0].barh(WORDS,recalls,color='#367d70');axes[0].set(xlim=(0,105),xlabel='Held-out recall (%)',title='Per-word recognition');axes[0].invert_yaxis()
    for i,v in enumerate(recalls):axes[0].text(v+.8,i,f'{e["per_word"][WORDS[i]]["correct"]}/{e["per_word"][WORDS[i]]["total"]}',va='center',fontsize=8)
    candidates=selection['candidates'];axes[1].plot(range(1,len(candidates)+1),[100*r['validation_macro_recall'] for r in candidates],'o-',ms=3)
    axes[1].set(xlabel='Prespecified head candidate',ylabel='Validation macro recall (%)',title='Validation-only search; no fabricated training curve')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(run/f'diagnostics.{ext}')
    plt.close(fig);return {'plots':['confusion.png','confusion.pdf','confusion.svg','diagnostics.png','diagnostics.pdf','diagnostics.svg']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('precache','init','prepare','extract','train','evaluate','plots'))
    p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--minutes',type=float,default=90)
    a=p.parse_args()
    if not 0<a.minutes<=120:p.error('Stage time budget must be within 120 minutes.')
    fn={'precache':lambda:precache(a.source,a.minutes),'init':lambda:initialize(a.run,a.source),'prepare':lambda:prepare(a.run),'extract':lambda:extract(a.run,a.minutes),
        'train':lambda:train(a.run),'evaluate':lambda:evaluate(a.run),'plots':lambda:plots(a.run)}[a.stage]
    print(json.dumps(fn(),indent=2),flush=True)


if __name__=='__main__':main()
