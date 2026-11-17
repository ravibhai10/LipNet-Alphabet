"""Preserved CNN-BiLSTM architecture, explicit frozen speaker partitions, new artifacts only."""
from __future__ import annotations
from copy import deepcopy
import io
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .quality import canonical, digest
from .models import TASK_LABELS, ModelRegistry, jpeg_frames
from .store import now


def load_alphabet(checkpoint):
    from models.alphabet_model import AlphabetCNNBiLSTM
    if checkpoint.get('vocabulary',TASK_LABELS['alphabet_recognition'])!=TASK_LABELS['alphabet_recognition']:
        raise ValueError('Alphabet vocabulary mismatch.')
    config=checkpoint.get('config',{})
    if any(config.get(k,v)!=v for k,v in {'num_classes':26,'t':30,'h':140,'w':46}.items()):
        raise ValueError('Alphabet checkpoint geometry mismatch.')
    net=AlphabetCNNBiLSTM(num_classes=26)
    net.load_state_dict(checkpoint.get('model_state_dict',checkpoint.get('state_dict')),strict=True)
    return net.eval()


def load_partition(store,version,split,negative=False):
    from app.services.browser_inference import BrowserInference
    values=[];labels=[]
    for row in version.get('negative_records',[]) if negative else version['records']:
        if row['split']!=split:continue
        raw=store.blob(row['id'],'rois')
        if digest(raw)!=row['roi_sha256']:raise ValueError('Alphabet ROI source integrity failed.')
        rois=np.load(io.BytesIO(raw),allow_pickle=False)
        values.append(BrowserInference.decode_frames(jpeg_frames(rois))[0])
        labels.append(-1 if negative else version['labels'].index(row['label']))
    if not values:raise ValueError('Empty alphabet partition.')
    return torch.stack(values),np.asarray(labels,dtype=np.int64)


def predict(net,x):
    values=[];times=[]
    with torch.inference_mode():
        for clip in x:
            started=time.perf_counter();values.append(net(clip[None]).softmax(-1)[0].numpy())
            times.append(1000*(time.perf_counter()-started))
    return np.stack(values),times


def train_alphabet(trainer,identifier,version):
    from .training import metrics,compare,_training_encoder_lock
    registry=ModelRegistry(trainer.store);registry.seed()
    with trainer.store.db() as db:
        from .models import champion_key,BASE_ALPHABET
        current=db.execute('SELECT model FROM champions WHERE task=?',(champion_key('alphabet_recognition',version.get('owner')),)).fetchone()
        champion_id=current[0] if current else BASE_ALPHABET
    champion_meta=registry.model(champion_id);source=trainer.store.root/champion_meta['checkpoint']
    if digest(source.read_bytes())!=champion_meta['checkpoint_sha256']:raise ValueError('Alphabet champion changed.')
    torch.manual_seed(42);np.random.seed(42)
    champion=torch.load(source,map_location='cpu',weights_only=True);net=load_alphabet(champion)
    output=trainer.store.directory/'derivatives'/identifier;output.mkdir(parents=True,exist_ok=False)
    (output/'dataset.json').write_bytes(canonical(version))
    history=[];best_loss=float('inf');best_state=None;stale=0
    with _training_encoder_lock():
        x,y=load_partition(trainer.store,version,'train');vx,vy=load_partition(trainer.store,version,'valid')
        before,baseline_latency=predict(net,vx)
        loader=DataLoader(TensorDataset(x,torch.from_numpy(y)),batch_size=4,shuffle=True,
                          generator=torch.Generator().manual_seed(42))
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.0001)
        for epoch in range(10):
            net.train();total=0.
            for batch,targets in loader:
                optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(net(batch),targets,label_smoothing=.05)
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();total+=loss.item()*len(batch)
            net.eval();probabilities,_=predict(net,vx)
            loss=float(-np.log(probabilities[np.arange(len(vy)),vy].clip(1e-12)).mean())
            history.append({'epoch':epoch+1,'train_loss':total/len(y),'validation_loss':loss})
            if loss<best_loss:
                best_loss=loss;best_state=deepcopy(net.state_dict());stale=0
            else:stale+=1
            if stale>=3:break
        configuration={'seed':42,'epochs_max':10,'batch_size':4,'optimizer':'AdamW','lr':.0001,
                       'early_stopping_patience':3,'selection':'validation cross entropy','test_used_for_selection':False,
                       'architecture':'Preserved AlphabetCNNBiLSTM','input':[30,1,140,46],'range':[0,1],
                       'dataset_manifest_sha256':version['manifest_sha256']}
        final={'model_state_dict':net.state_dict(),'config':{'num_classes':26,'t':30,'h':140,'w':46},
               'vocabulary':version['labels'],'training':configuration}
        torch.save(final,output/'final.pt');net.load_state_dict(best_state);final['model_state_dict']=best_state
        torch.save(final,output/'head.pt');after,latency=predict(net,vx)
    champion_report=metrics(before,vy,.5,.05,version['labels']);report=metrics(after,vy,.5,.05,version['labels'])
    negatives=[r for r in version.get('negative_records',[]) if r['split']=='valid']
    if negatives:
        with _training_encoder_lock():
            nx,_=load_partition(trainer.store,version,'valid',negative=True)
            baseline=load_alphabet(champion);negative_before,_=predict(baseline,nx);negative_after,_=predict(net,nx)
        for r,probabilities in ((champion_report,negative_before),(report,negative_after)):
            order=np.argsort(-probabilities,axis=1)
            accepted=[p[o[0]]>=.5 and p[o[0]]-p[o[1]]>=.05 for p,o in zip(probabilities,order)]
            r.update(false_acceptance_rate=float(np.mean(accepted)),negative_samples=len(negatives),
                     negative_evaluation='Reviewer-approved UNKNOWN validation recordings')
    try:
        import psutil
        memory=psutil.Process().memory_info();peak=getattr(memory,'peak_wset',memory.rss)
    except ImportError:peak=None
    comparison=compare(champion_report,report,champion_id=champion_id,latency_ms=float(np.mean(latency)),
                       memory_bytes=peak,model_size_bytes=(output/'head.pt').stat().st_size)
    value={'id':identifier,'task':'alphabet_recognition','version':digest((output/'head.pt').read_bytes()),
           'checkpoint':str((output/'head.pt').relative_to(trainer.store.root)),
           'checkpoint_sha256':digest((output/'head.pt').read_bytes()),'vocabulary':version['labels'],
           'training_dataset_version':version['id'],'owner':version.get('owner'),'protocol':version.get('protocol'),'preprocessing_id':'adaptive-eyeaxis96-jpeg75-v2','training_configuration':configuration,
           'input_spec':champion_meta['input_spec'],'preprocessing':champion_meta['preprocessing'],
           'validation_report':report,'test_report':None,'comparison':comparison,'cpu_latency_ms':float(np.mean(latency)),
           'latency_scope':'Alphabet model forward pass on approved validation inputs; preprocessing excluded',
           'process_peak_bytes':peak,'model_size_bytes':(output/'head.pt').stat().st_size,
           'deployment_status':'challenger','regression_passed':False,'created':now()}
    for name,data in [('config',configuration),('history',history),('comparison',comparison)]:
        (output/(name+'.json')).write_bytes(canonical(data))
    with trainer.store.db() as db:
        db.execute('INSERT INTO models VALUES(?,?,?)',(identifier,'alphabet_recognition',json.dumps(value)))
        trainer.store.event(db,identifier,'training-worker','challenger_saved',{'checkpoint_sha256':value['checkpoint_sha256']})
    return {'model_id':identifier,'comparison':comparison,'model_activated':False}
