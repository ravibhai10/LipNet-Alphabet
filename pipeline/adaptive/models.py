"""Explicit task adapters and an evidence-bound champion registry."""
from __future__ import annotations
import json
from pathlib import Path
import threading
import numpy as np

from .quality import canonical, digest
from .store import now

WORDS = ['choose','connection','hello','navigation','next','previous','start','stop']
TASK_LABELS = {'alphabet_recognition':[chr(65+i) for i in range(26)], 'isolated_word':WORDS}
BASE_WORD = 'avhubert-miracl8-preserved'
BASE_ALPHABET = 'alphabet-cnn-bilstm-preserved'
_word_adapter_lock = threading.Lock()
_alphabet_models = {}


class TaskAdapter:
    def __init__(self, task, service, model_id):
        if task not in TASK_LABELS: raise ValueError('Unsupported recognition task.')
        self.task, self.service, self.model_id = task, service, model_id

    def load(self):
        if not self.service.load(): raise RuntimeError('Selected task model is unavailable.')
        return self

    def predict(self, frames):
        self.load()
        result = self.service.predict(frames)
        if self.task == 'isolated_word':
            head = self.service.model.head
            if head['vocabulary'] != TASK_LABELS[self.task]: raise ValueError('Word adapter vocabulary mismatch.')
            candidate = result['predicted_word']
            prediction = None if result['uncertain'] else result.get('word')
            top = [{'label':p['word'],'score':p['prob']} for p in result['top_predictions']]
            version = self.service.model.head_sha256
        else:
            candidate = result['letter']
            prediction = None if result.get('uncertain',False) else candidate
            top = [{'label':p['letter'],'score':p['prob']} for p in result['top3']]
            version = self.service.model_version
        return {'task':self.task,'prediction':prediction,'candidate':candidate,'confidence':result['confidence'],
                'top_k':top,'model_id':self.model_id,'model_version':version,'vocabulary':TASK_LABELS[self.task],
                'status':'UNCERTAIN' if prediction is None else 'PREDICTED','latency_ms':result.get('latency_ms'),
                'confidence_calibrated':False}


def baseline_adapter(task):
    if task == 'isolated_word':
        # Dedicated explicit AV-HuBERT path, independent of the legacy default.
        from app.services.pretrained_word_inference import word_inference
        return TaskAdapter(task,word_inference,BASE_WORD)
    if task == 'alphabet_recognition':
        from app.services.browser_inference import browser_inference
        if browser_inference.backend != 'alphabet':
            raise RuntimeError('This feedback protocol requires the preserved alphabet CNN-BiLSTM.')
        return TaskAdapter(task,browser_inference,BASE_ALPHABET)
    raise ValueError('Unsupported recognition task.')


def jpeg_frames(rois):
    import base64
    import cv2
    return ['data:image/jpeg;base64,'+base64.b64encode(cv2.imencode('.jpg',roi,[cv2.IMWRITE_JPEG_QUALITY,75])[1]).decode() for roi in rois]


class ModelRegistry:
    def __init__(self, store):
        self.store = store

    def seed(self):
        specs = [(BASE_WORD,'isolated_word','runs/word_accuracy/avhubert_miracl8_v1/head.pt',
                  {'frames':29,'spatial':[96,96],'encoder_crop':88,'mean':.421,'std':.165},
                  'Frozen visual AV-HuBERT layer 9 + mean/std logistic regression'),
                 (BASE_ALPHABET,'alphabet_recognition','models/alphabet_classifier.pt',
                  {'frames':30,'channels':1,'spatial':[140,46],'range':[0,1]},'Frame CNN + bidirectional LSTM')]
        for identifier,task,relative,spec,architecture in specs:
            with self.store.db() as db:
                if db.execute('SELECT 1 FROM models WHERE id=?',(identifier,)).fetchone(): continue
            path = self.store.root/relative
            if not path.is_file(): continue
            value = {'id':identifier,'task':task,'version':digest(path.read_bytes()),'checkpoint':relative,
                     'checkpoint_sha256':digest(path.read_bytes()),'model_size_bytes':path.stat().st_size,
                     'vocabulary':TASK_LABELS[task],'input_spec':spec,'preprocessing':spec,'architecture':architecture,
                     'training_dataset_version':'historical MIRACL' if task=='isolated_word' else 'historical alphabet; provenance incomplete',
                     'training_configuration':None,'validation_report':None,'test_report':None,'cpu_latency_ms':None,
                     'deployment_status':'preserved_research_baseline','accuracy_scope':'Physical webcam accuracy unproven',
                     'evaluation_links':['RELIABILITY_RESULTS.md','.qa/reliability_20261109/final_frozen_evaluation.json'] if task=='isolated_word' else ['MODEL_AVAILABILITY.md']}
            report = self.store.root/'.qa/reliability_20261109/final_frozen_evaluation.json'
            if task=='isolated_word' and report.is_file():
                evidence = json.loads(report.read_text())
                value['validation_report']=evidence['partitions']['valid']
                value['test_report']=evidence['partitions']['test']
                value['cpu_latency_ms']=444.01
                value['latency_scope']='Historical ASGI n=8, JPEG to response; not a fresh measurement'
            with self.store.db() as db:
                db.execute('INSERT OR IGNORE INTO models VALUES(?,?,?)',(identifier,task,json.dumps(value)))
                db.execute('INSERT OR IGNORE INTO champions VALUES(?,?,NULL)',(task,identifier))

    def list(self):
        self.seed()
        with self.store.db() as db:
            models = [json.loads(r[0]) for r in db.execute('SELECT payload FROM models')]
            champions = {r['task']:dict(r) for r in db.execute('SELECT * FROM champions')}
        for model in models: model.pop('checkpoint',None)
        return {'models':models,'champions':champions,
                'physical_webcam_evaluation':None,'webcam_replay_evaluation':None,
                'alphabet_offline_evaluation':None,'word_offline_evaluation':'Historical MIRACL, original protocol limitations apply'}

    def model(self, identifier):
        self.seed()
        with self.store.db() as db:
            row = db.execute('SELECT payload FROM models WHERE id=?',(identifier,)).fetchone()
            if not row: raise LookupError('Model not found.')
            return json.loads(row[0])

    def predict(self, task, rois):
        self.seed()
        with self.store.db() as db:
            row = db.execute('SELECT model FROM champions WHERE task=?',(task,)).fetchone()
        identifier = row[0] if row else BASE_WORD if task=='isolated_word' else BASE_ALPHABET
        if identifier in {BASE_WORD,BASE_ALPHABET}:
            value=baseline_adapter(task).predict(jpeg_frames(rois))
            if value['model_version']!=self.model(identifier)['checkpoint_sha256']:
                raise ValueError('Selected backend differs from the registered preserved model.')
            return value
        meta = self.model(identifier)
        if meta['task']!=task or meta['vocabulary']!=TASK_LABELS[task]: raise ValueError('Champion task contract mismatch.')
        self.store.version(meta['training_dataset_version'])
        if task=='alphabet_recognition':
            from .alphabet_training import load_alphabet
            from app.services.browser_inference import BrowserInference
            import torch
            path=self.store.root/meta['checkpoint']
            if digest(path.read_bytes())!=meta['checkpoint_sha256']:raise ValueError('Alphabet challenger checkpoint changed.')
            if identifier not in _alphabet_models:
                _alphabet_models[identifier]=load_alphabet(torch.load(path,map_location='cpu',weights_only=True))
            with torch.inference_mode():
                p=_alphabet_models[identifier](BrowserInference.decode_frames(jpeg_frames(rois))).softmax(-1)[0].numpy()
            order=np.argsort(-p);vocab=TASK_LABELS[task]
            accepted=p[order[0]]>=.5 and p[order[0]]-p[order[1]]>=.05
            return {'task':task,'prediction':vocab[order[0]] if accepted else None,'candidate':vocab[order[0]],
                    'confidence':float(p[order[0]]),'top_k':[{'label':vocab[i],'score':float(p[i])} for i in order[:5]],
                    'model_id':identifier,'model_version':meta['version'],'vocabulary':vocab,'latency_ms':None,
                    'status':'PREDICTED' if accepted else 'UNCERTAIN','confidence_calibrated':False}
        from .training import head_probabilities
        from pipeline.word_recognition.consensus import temporal_consensus
        import torch
        path=self.store.root/meta['checkpoint']
        if digest(path.read_bytes())!=meta['checkpoint_sha256']: raise ValueError('Champion checkpoint changed.')
        # Share the existing singleton encoder; challenger head work stays under one CPU lock.
        with _word_adapter_lock:
            base=baseline_adapter(task).load().service.model
            features=base.encoder.encode_frames(rois,source_fps=None,target_frames=29,output_layer=9)
            head=torch.load(path,map_location='cpu',weights_only=True)
            p=head_probabilities(head,features);order=np.argsort(-p);policy=meta['policy']
            consensus=temporal_consensus(head,features,policy)
            consistent=consensus['confirmed'] and consensus['views'][0]['candidate']==WORDS[int(order[0])]
        accepted=p[order[0]]>=policy['threshold'] and p[order[0]]-p[order[1]]>=policy['margin'] and consistent
        return {'task':task,'prediction':WORDS[int(order[0])] if accepted else None,'candidate':WORDS[int(order[0])],
                'confidence':float(p[order[0]]),'top_k':[{'label':WORDS[i],'score':float(p[i])} for i in order[:5]],
                'model_id':identifier,'model_version':meta['version'],'vocabulary':WORDS,'latency_ms':None,
                'status':'PREDICTED' if accepted else 'UNCERTAIN','confidence_calibrated':False}

    def deploy(self, identifier, actor, approved):
        if approved is not True: raise ValueError('Explicit reviewer approval is required.')
        model = self.model(identifier)
        gates = model.get('comparison',{})
        if gates.get('eligible') is not True or model.get('test_report') is None or model.get('regression_passed') is not True or model.get('integration_passed') is not True:
            raise ValueError('Challenger lacks passing validation comparison, final held-out evidence or regression results.')
        if model['test_report'].get('negative_samples',0)<20:
            raise ValueError('The final protocol needs at least 20 held-out negative recordings. Test metrics never select the challenger.')
        self.store.version(model['training_dataset_version'])
        path = self.store.root/model['checkpoint']
        if digest(path.read_bytes())!=model['checkpoint_sha256']: raise ValueError('Checkpoint integrity failed.')
        with self.store.db() as db:
            prior = db.execute('SELECT model FROM champions WHERE task=?',(model['task'],)).fetchone()
            if not prior or prior[0]!=gates['champion_id']: raise ValueError('Champion changed; a new paired comparison is required.')
            db.execute('UPDATE champions SET model=?,previous=? WHERE task=?',(identifier,prior[0],model['task']))
            self.store.event(db,identifier,actor,'deploy',{'previous':prior[0],'comparison_sha256':digest(canonical(gates))})
        return {'champion':identifier,'previous':prior[0],'task':model['task']}

    def rollback(self, task, actor):
        with self.store.db() as db:
            row=db.execute('SELECT * FROM champions WHERE task=?',(task,)).fetchone()
            if not row or not row['previous']: raise ValueError('No previous champion is available.')
            model_row=db.execute('SELECT payload FROM models WHERE id=?',(row['previous'],)).fetchone()
            model=json.loads(model_row[0]);path=self.store.root/model['checkpoint']
            if digest(path.read_bytes())!=model['checkpoint_sha256']: raise ValueError('Rollback checkpoint integrity failed.')
            if model['deployment_status']=='challenger':
                version=db.execute('SELECT payload FROM versions WHERE id=?',(model['training_dataset_version'],)).fetchone()
                if not version:raise ValueError('Rollback dataset lineage is unavailable.')
                identifiers=json.loads(version[0])['record_ids']
                for identifier in identifiers:
                    if self.store.row(db,identifier)['status']=='WITHDRAWN':
                        raise ValueError('Rollback model contains withdrawn feedback; review its lineage first.')
            db.execute('UPDATE champions SET model=?,previous=? WHERE task=?',(row['previous'],row['model'],task))
            self.store.event(db,row['previous'],actor,'rollback',{'from':row['model'],'task':task})
        return {'champion':row['previous'],'previous':row['model'],'task':task}
