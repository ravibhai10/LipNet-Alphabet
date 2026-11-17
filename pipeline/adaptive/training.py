"""Batch fitting on approved feedback. Validation selects; held-out test is a separate final step."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

import numpy as np

from .models import BASE_WORD, BASE_ALPHABET, WORDS, TASK_LABELS, ModelRegistry, baseline_adapter
from .quality import canonical, digest
from .store import now


def metrics(p, labels, threshold=.95, margin=.2, words=None):
    from sklearn.metrics import classification_report, confusion_matrix
    words=words or WORDS
    p=np.asarray(p,dtype=float); p=p/p.sum(1,keepdims=True)
    y=np.asarray(labels); guess=p.argmax(1); order=np.argsort(-p,axis=1)
    confidence=p.max(1); margins=p[np.arange(len(y)),order[:,0]]-p[np.arange(len(y)),order[:,1]]
    accepted=(confidence>=threshold)&(margins>=margin)
    report=classification_report(y,guess,labels=list(range(len(words))),target_names=words,output_dict=True,zero_division=0)
    risk=[]
    for count in range(1,len(y)+1):
        indexes=np.argsort(-confidence,kind='stable')[:count]
        risk.append({'coverage':count/len(y),'risk':float(np.mean(guess[indexes]!=y[indexes])),
                     'minimum_score':float(confidence[indexes[-1]])})
    ece=0.
    for lower in np.linspace(0,.9,10):
        mask=(confidence>=lower)&((confidence<lower+.1) if lower<.9 else (confidence<=1))
        if mask.any(): ece+=float(mask.mean()*abs((guess[mask]==y[mask]).mean()-confidence[mask].mean()))
    return {'samples':len(y),'top1_accuracy':float(np.mean(guess==y)),
            'top5_accuracy':float(np.mean([target in row[:5] for target,row in zip(y,order)])),
            'macro_precision':report['macro avg']['precision'],'macro_recall':report['macro avg']['recall'],
            'macro_f1':report['macro avg']['f1-score'],'per_class':{w:report[w] for w in words},
            'confusion_matrix':confusion_matrix(y,guess,labels=list(range(len(words)))).tolist(),
            'coverage':float(accepted.mean()),'rejection_rate':float(1-accepted.mean()),
            'accepted_subset_accuracy':float(np.mean(guess[accepted]==y[accepted])) if accepted.any() else None,
            'false_acceptance_rate':None,'negative_evaluation':'No independent negative set in this version',
            'ece_10_bin':ece,'brier':float(np.mean(np.sum((p-np.eye(len(words))[y])**2,axis=1))),
            'risk_coverage':risk,'confidence_calibrated':False,
            'rejection_scope':'Score/margin only; live temporal agreement can reject further'}


def compare(champion, challenger, *, champion_id, latency_ms, memory_bytes, model_size_bytes):
    reasons=[]
    if challenger['top1_accuracy']<=champion['top1_accuracy']: reasons.append('Top-1 must strictly improve.')
    if challenger['macro_f1']<champion['macro_f1']: reasons.append('Macro F1 regressed.')
    if any(challenger['per_class'][w]['f1-score']<champion['per_class'][w]['f1-score']-.05 for w in challenger['per_class']):
        reasons.append('A class F1 regressed by more than .05.')
    if latency_ms is None or latency_ms>1500: reasons.append('CPU latency is missing or exceeds 1500 ms.')
    if memory_bytes is None or memory_bytes>2*1024**3: reasons.append('Process memory is missing or exceeds 2 GiB.')
    if model_size_bytes>2*1024**3: reasons.append('Artifact size exceeds 2 GiB.')
    if challenger['false_acceptance_rate'] is None:
        reasons.append('Independent negative false-acceptance evaluation is unavailable.')
    elif champion['false_acceptance_rate'] is None or challenger['false_acceptance_rate']>champion['false_acceptance_rate']+.02:
        reasons.append('Negative false acceptance regressed by more than .02 or champion evidence is missing.')
    if challenger.get('negative_samples',0) and challenger['negative_samples']<20:
        reasons.append('At least 20 independent validation negative recordings are required.')
    return {'eligible':not reasons,'reasons':reasons,'champion_id':champion_id,'validation_only':True,
            'limits':{'class_f1_regression':.05,'latency_ms':1500,'process_peak_bytes':2*1024**3,'model_bytes':2*1024**3},
            'champion':champion,'challenger':challenger}


def head_probabilities(head, features):
    from pipeline.word_recognition.pretrained import sequence_probabilities
    return sequence_probabilities(head,features)


def word_acceptance(head, features):
    from pipeline.word_recognition.consensus import temporal_consensus
    p=head_probabilities(head,features);order=np.argsort(-p)
    consensus=temporal_consensus(head,features,{'threshold':.95,'margin':.2})
    return bool(p[order[0]]>=.95 and p[order[0]]-p[order[1]]>=.2 and consensus['confirmed'] and
                consensus['views'][0]['candidate']==WORDS[int(order[0])])


class BatchTrainer:
    def __init__(self, store):
        self.store=store
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='feedback-training')
        self.lock=threading.Lock()

    def create(self, version_id, actor):
        version=self.store.version(version_id)
        if version['task'] not in TASK_LABELS or version['labels']!=TASK_LABELS[version['task']]:
            raise ValueError('Training requires the exact task-specific label mapping.')
        if not self.lock.acquire(blocking=False): raise BlockingIOError('A training or final evaluation job is already running.')
        identifier=uuid.uuid4().hex
        try:
            with self.store.db() as db:
                if db.execute('SELECT 1 FROM jobs WHERE version=?',(version_id,)).fetchone():
                    raise ValueError('This immutable dataset already has a training attempt. Preserve its evidence and collect a new version.')
                db.execute('INSERT INTO jobs VALUES(?,?,?,?,NULL,NULL)',(identifier,version_id,'QUEUED',now()))
                self.store.event(db,identifier,actor,'training_requested',{'dataset_version':version_id})
            self.executor.submit(self._worker,identifier,version)
        except BaseException:
            self.lock.release();raise
        return {'job_id':identifier,'status':'QUEUED'}

    def status(self, identifier):
        with self.store.db() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row: raise LookupError('Training job not found.')
            value=dict(row)
            if value['result']:value['result']=json.loads(value['result'])
            return value

    def _worker(self, identifier, version):
        try:
            with self.store.db() as db:db.execute("UPDATE jobs SET status='RUNNING' WHERE id=?",(identifier,))
            result=self.train(identifier,version)
            with self.store.db() as db:
                db.execute("UPDATE jobs SET status='COMPLETED', result=? WHERE id=?",(json.dumps(result),identifier))
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Feedback training failed; production was preserved')
            with self.store.db() as db:
                db.execute("UPDATE jobs SET status='FAILED',error=? WHERE id=?",('Training failed. Review the protected local training log; production is unchanged.',identifier))
        finally:self.lock.release()

    def features(self, version, split, encoder, directory, negative=False):
        if split not in {'train','valid','test'}:raise ValueError('Unknown partition.')
        values=[];labels=[];times=[]
        for record in version.get('negative_records',[]) if negative else version['records']:
            if record['split']!=split:continue
            raw=self.store.blob(record['id'],'rois')
            if digest(raw)!=record['roi_sha256']: raise ValueError('Approved ROI hash changed.')
            rois=np.load(io.BytesIO(raw),allow_pickle=False)
            # Match the deployed browser/service JPEG boundary before encoding.
            from .models import jpeg_frames
            from app.services.browser_inference import BrowserInference
            rois=np.stack(BrowserInference.decode_jpegs(jpeg_frames(rois)))
            if not hasattr(encoder,'_adaptive_checkpoint_sha256'):
                import hashlib
                with encoder.ckpt_path.open('rb') as stream:encoder._adaptive_checkpoint_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
            receipt={'source':record['source_sha256'],'roi':record['roi_sha256'],
                     'encoder_sha256':encoder._adaptive_checkpoint_sha256,'encoder_preprocessing':encoder.preprocessing(),
                     'layer':9,'preprocessing':record['preprocessing'],'label':record['label'],'split':split}
            key=digest(canonical(receipt))
            started=time.perf_counter()
            sequence=encoder.encode_frames(rois,source_fps=None,target_frames=29,output_layer=9)
            times.append(1000*(time.perf_counter()-started))
            if sequence.shape!=(29,768) or not np.isfinite(sequence).all():raise ValueError('Encoder feature contract failed.')
            np.save(directory/(key+'.npy'),sequence,allow_pickle=False)
            (directory/(key+'.json')).write_bytes(canonical({**receipt,'cache_key':key,'feature_sha256':digest((directory/(key+'.npy')).read_bytes())}))
            values.append(sequence);labels.append(-1 if negative else WORDS.index(record['label']))
        if not values:raise ValueError('Empty partition.')
        return np.stack(values),np.array(labels),times

    def train(self, identifier, version):
        if version['task']=='alphabet_recognition':
            from .alphabet_training import train_alphabet
            return train_alphabet(self,identifier,version)
        import torch
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits
        from pipeline.word_recognition.pretrained import pool
        registry=ModelRegistry(self.store); registry.seed()
        with self.store.db() as db:
            champion_id=db.execute("SELECT model FROM champions WHERE task='isolated_word'").fetchone()[0]
        champion_meta=registry.model(champion_id)
        path=self.store.root/champion_meta['checkpoint']
        if digest(path.read_bytes())!=champion_meta['checkpoint_sha256']:raise ValueError('Champion integrity failed.')
        champion=torch.load(path,map_location='cpu',weights_only=True)
        base=baseline_adapter('isolated_word').load().service.model
        output=self.store.root/'runs/adaptive'/identifier;output.mkdir(parents=True,exist_ok=False)
        (output/'dataset.json').write_bytes(canonical(version))
        with _training_encoder_lock():
            x,y,_=self.features(version,'train',base.encoder,output)
            vx,vy,latency=self.features(version,'valid',base.encoder,output)
        # No test features or test statistics are opened during selection.
        train_x=np.stack([pool(f,'mean_std') for f in x]); valid_x=np.stack([pool(f,'mean_std') for f in vx])
        scaler=StandardScaler().fit(train_x)
        with threadpool_limits(limits=1):
            classifier=LogisticRegression(C=1.,max_iter=600,class_weight='balanced',random_state=42)
            classifier.fit(scaler.transform(train_x),y)
        if classifier.classes_.tolist()!=list(range(8)):raise ValueError('Eight-class output contract failed.')
        head={**champion,'contract':{**champion['contract'],'adaptive_dataset_version':version['id'],
                                  'adaptive_manifest_sha256':version['manifest_sha256']},
              'mean':torch.tensor(scaler.mean_,dtype=torch.float32),
              'scale':torch.tensor(scaler.scale_,dtype=torch.float32),
              'weight':torch.tensor(classifier.coef_,dtype=torch.float32),'bias':torch.tensor(classifier.intercept_,dtype=torch.float32)}
        torch.save(head,output/'head.pt')
        torch.save(head,output/'final.pt')
        challenger_report=metrics(np.stack([head_probabilities(head,f) for f in vx]),vy)
        champion_report=metrics(np.stack([head_probabilities(champion,f) for f in vx]),vy)
        from pipeline.word_recognition.consensus import temporal_consensus
        for h,r in ((champion,champion_report),(head,challenger_report)):
            scores=np.stack([head_probabilities(h,f) for f in vx]);order=np.argsort(-scores,axis=1)
            accepted=np.array([word_acceptance(h,f) for f in vx])
            r.update(coverage=float(accepted.mean()),rejection_rate=float(1-accepted.mean()),
                     accepted_subset_accuracy=float(np.mean(scores[accepted].argmax(1)==vy[accepted])) if accepted.any() else None,
                     rejection_scope='Fixed score/margin gate plus overlapping temporal agreement')
        negatives=[r for r in version.get('negative_records',[]) if r['split']=='valid']
        if negatives:
            with _training_encoder_lock():nx,_,_=self.features(version,'valid',base.encoder,output,negative=True)
            for h,r in ((champion,champion_report),(head,challenger_report)):
                accepted=[]
                for f in nx:
                    accepted.append(word_acceptance(h,f))
                r.update(false_acceptance_rate=float(np.mean(accepted)),negative_samples=len(nx),
                         negative_evaluation='Reviewer-approved UNKNOWN clips on validation speakers; exact frozen IDs in dataset version')
        try:
            import psutil
            memory=psutil.Process().memory_info()
            peak=getattr(memory,'peak_wset',memory.rss)
        except ImportError:peak=None
        comparison=compare(champion_report,challenger_report,champion_id=champion_id,
                           latency_ms=float(np.mean(latency)),memory_bytes=peak,
                           model_size_bytes=(output/'head.pt').stat().st_size)
        configuration={'seed':42,'classifier':'LogisticRegression','C':1.,'max_iter':600,'class_weight':'balanced',
                       'scaler_fit':'train only','feature_layer':9,'pooling':'mean_std','encoder_frozen':True,
                       'test_used_for_selection':False,'dataset_manifest_sha256':version['manifest_sha256']}
        value={'id':identifier,'task':'isolated_word','version':digest((output/'head.pt').read_bytes()),
               'checkpoint':str((output/'head.pt').relative_to(self.store.root)),
               'checkpoint_sha256':digest((output/'head.pt').read_bytes()),'vocabulary':WORDS,
               'training_dataset_version':version['id'],'training_configuration':configuration,
               'input_spec':champion_meta['input_spec'],'preprocessing':champion_meta['preprocessing'],
               'validation_report':challenger_report,'test_report':None,'comparison':comparison,
               'cpu_latency_ms':float(np.mean(latency)),'latency_scope':'Frozen encoder extraction on approved validation ROIs',
               'process_peak_bytes':peak,'memory_scope':'Cumulative process peak; shared encoder',
               'model_size_bytes':(output/'head.pt').stat().st_size,'encoder_size_bytes':base.encoder.ckpt_path.stat().st_size,
               'policy':{'threshold':.95,'margin':.2},'deployment_status':'challenger',
               'regression_passed':False,'created':now()}
        (output/'comparison.json').write_bytes(canonical(comparison));(output/'config.json').write_bytes(canonical(configuration))
        (output/'history.json').write_bytes(canonical({'optimizer_iterations':classifier.n_iter_.tolist(),
            'validation_top1_accuracy':challenger_report['top1_accuracy'],'validation_macro_f1':challenger_report['macro_f1']}))
        with self.store.db() as db:
            db.execute('INSERT INTO models VALUES(?,?,?)',(identifier,'isolated_word',json.dumps(value)))
            self.store.event(db,identifier,'training-worker','challenger_saved',{'checkpoint_sha256':value['checkpoint_sha256']})
        return {'model_id':identifier,'comparison':comparison,'model_activated':False}

    def finalize(self, identifier, actor):
        model=ModelRegistry(self.store).model(identifier)
        if model['id'] in {BASE_WORD,BASE_ALPHABET} or model.get('test_report') is not None:
            raise ValueError('Final held-out evaluation is one-time and only available for new challengers.')
        version=self.store.version(model['training_dataset_version'])
        if not self.lock.acquire(blocking=False):raise BlockingIOError('A batch job is already running.')
        try:
            output=self.store.root/Path(model['checkpoint']).parent
            self.store.claim_test(version['id'],actor)
            # Claim before opening held-out values. A failure cannot be used to retune.
            with (output/'test_access.json').open('xb') as stream:
                stream.write(canonical({'actor':actor,'model':identifier,'at':now(),'checkpoint_sha256':model['checkpoint_sha256']}))
            model['final_evaluation_status']='RUNNING'
            with self.store.db() as db:
                db.execute('UPDATE models SET payload=? WHERE id=?',(json.dumps(model),identifier))
            self.executor.submit(self._final_worker,model,version,output)
        except BaseException:self.lock.release();raise
        return {'model_id':identifier,'status':'FINAL_EVALUATION_RUNNING'}

    def _final_worker(self, model, version, output):
        try:
            import torch
            head=torch.load(self.store.root/model['checkpoint'],map_location='cpu',weights_only=True)
            if version['task']=='alphabet_recognition':
                from .alphabet_training import load_alphabet, load_partition, predict
                net=load_alphabet(head)
                with _training_encoder_lock():
                    x,y=load_partition(self.store,version,'test')
                    probabilities,_=predict(net,x)
                report=metrics(probabilities,y,.5,.05,version['labels'])
            else:
                encoder=baseline_adapter('isolated_word').load().service.model.encoder
                with _training_encoder_lock():x,y,_=self.features(version,'test',encoder,output)
                probabilities=np.stack([head_probabilities(head,f) for f in x])
                report=metrics(probabilities,y)
                accepted=np.array([word_acceptance(head,f) for f in x])
                report.update(coverage=float(accepted.mean()),rejection_rate=float(1-accepted.mean()),
                    accepted_subset_accuracy=float(np.mean(probabilities[accepted].argmax(1)==y[accepted])) if accepted.any() else None,
                    rejection_scope='Fixed score/margin plus overlapping temporal agreement')
            negatives=[r for r in version.get('negative_records',[]) if r['split']=='test']
            if negatives:
                if version['task']=='alphabet_recognition':
                    with _training_encoder_lock():nx,_=load_partition(self.store,version,'test',negative=True);npred,_=predict(net,nx)
                    order=np.argsort(-npred,axis=1)
                    accepted=[p[o[0]]>=.5 and p[o[0]]-p[o[1]]>=.05 for p,o in zip(npred,order)]
                else:
                    from pipeline.word_recognition.consensus import temporal_consensus
                    with _training_encoder_lock():nx,_,_=self.features(version,'test',encoder,output,negative=True)
                    accepted=[]
                    for f in nx:
                        accepted.append(word_acceptance(head,f))
                report.update(false_acceptance_rate=float(np.mean(accepted)),negative_samples=len(negatives),
                              negative_evaluation='Frozen held-out UNKNOWN recordings; not used for selection')
            report.update(dataset_version=version['id'],scope='Consent-reviewed held-out recording speakers; not general physical webcam accuracy',
                          checkpoint_sha256=model['checkpoint_sha256'],test_used_for_selection=False)
            (output/'test_report.json').write_bytes(canonical(report));model['test_report']=report
            interpreter=self.store.root/'.venv-word-recognition/Scripts/python.exe'
            if not interpreter.is_file():raise RuntimeError('Verified regression environment is unavailable.')
            with (output/'regression.txt').open('w',encoding='utf-8') as stream:
                result=subprocess.run([str(interpreter),'-m','pytest','tests','pipeline/eval/tests','pipeline/grid/tests','pipeline/avhubert/tests'],
                                      cwd=self.store.root,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
            model['regression_passed']=result.returncode==0
            model['integration_passed']=bool(report['samples'] and np.isfinite(report['top1_accuracy']))
            model['regression_log_sha256']=digest((output/'regression.txt').read_bytes())
            model['final_evaluation_status']='COMPLETED'
            with self.store.db() as db:
                db.execute('UPDATE models SET payload=? WHERE id=?',(json.dumps(model),model['id']))
                self.store.event(db,model['id'],'evaluation-worker','final_evaluation',{'regression_passed':model['regression_passed']})
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Final feedback evaluation failed')
            model['final_evaluation_status']='FAILED';model['regression_passed']=False
            with self.store.db() as db:
                db.execute('UPDATE models SET payload=? WHERE id=?',(json.dumps(model),model['id']))
                self.store.event(db,model['id'],'evaluation-worker','final_evaluation_failed',{'production_changed':False})
        finally:self.lock.release()


def _training_encoder_lock():
    # Same resource lock used by new feedback inference; no concurrent encoder jobs.
    from app.services.word_compute import word_cpu_lock
    return word_cpu_lock
