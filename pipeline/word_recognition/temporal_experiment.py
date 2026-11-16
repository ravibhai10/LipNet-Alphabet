"""Controlled CPU BiLSTM experiments on verified frozen visual sequences.

Only training/validation feature files are opened. No test evaluation or model
promotion happens here. This module does not alter any active checkpoint.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from .common import ROOT, read_rows, sha256
from .pretrained import LAYERS, predict_features


class TemporalHead(nn.Module):
    def __init__(self, classes, hidden=32, projection=64, dropout=.2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(768, projection), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(projection, hidden, batch_first=True, bidirectional=True)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 768:
            raise ValueError('Expected [batch, time, 768] frozen visual features.')
        encoded, _ = self.lstm(self.encoder(x))
        return self.output(encoded.mean(1))


def load_partition(rows, split, words, layer):
    if split not in {'train', 'valid'}:
        raise ValueError('This experiment cannot open test features.')
    values, labels, kept = [], [], []
    selected = [r for r in rows if r['split'] == split and r['role'] == 'known_word']
    for row in selected:
        if not row['usable']: continue
        if sha256(row['feature']) != row['feature_sha256']:
            raise ValueError('Frozen feature integrity mismatch.')
        x = np.load(row['feature'], allow_pickle=False)
        if x.shape != (3,29,768) or not np.isfinite(x).all():
            raise ValueError('Invalid finite visual sequence.')
        values.append(x[LAYERS.index(layer)]); labels.append(words.index(row['word_label'])); kept.append(row)
    if not values: raise ValueError('No usable training/validation sequences.')
    return np.stack(values), np.array(labels, dtype=np.int64), kept, len(selected)


def metrics(probabilities, targets, words, total):
    # Float32 softmax roundoff must not be mistaken for invalid probabilities by
    # float64 metric implementations. This does not alter model predictions.
    probabilities=np.asarray(probabilities,dtype=np.float64)
    probabilities=probabilities/probabilities.sum(axis=1,keepdims=True)
    predicted = probabilities.argmax(1)
    report = classification_report(targets, predicted, labels=list(range(len(words))),
                                   target_names=words, output_dict=True, zero_division=0)
    return {'samples': total, 'usable': len(targets), 'correct': int((predicted==targets).sum()),
            'top1_accuracy': float((predicted==targets).sum()/total),
            'top5_accuracy': float(sum(y in order for y, order in zip(targets, np.argsort(-probabilities,axis=1)[:,:5]))/total),
            'macro_f1_usable': report['macro avg']['f1-score'], 'per_class': report,
            'confusion_matrix': confusion_matrix(targets,predicted,labels=list(range(len(words)))).tolist(),
            'log_loss_usable': float(log_loss(targets,probabilities,labels=list(range(len(words)))))}


def run_experiment(source, output, *, epochs=25, lengths=(15,29), seed=42):
    output=Path(output); source=Path(source)
    if output.exists(): raise FileExistsError('Choose a new experiment output; artifacts are protected.')
    if epochs<1 or not lengths or any(n not in range(8,30) for n in lengths): raise ValueError('Invalid temporal experiment settings.')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    head=torch.load(source/'head.pt',map_location='cpu',weights_only=True)
    rows=read_rows(source/'features.jsonl'); words=head['vocabulary']; layer=head['feature_layer']
    x,y,training,train_total=load_partition(rows,'train',words,layer)
    v,z,validation,valid_total=load_partition(rows,'valid',words,layer)
    if {r['speaker_id'] for r in training} & {r['speaker_id'] for r in validation}:
        raise ValueError('Speaker leakage between training and validation.')
    if {r['source_sha256'] for r in training} & {r['source_sha256'] for r in validation}:
        raise ValueError('Source leakage between training and validation.')
    if set(y)!=set(range(len(words))): raise ValueError('Incomplete training vocabulary.')
    output.mkdir(parents=True)
    # Baseline geometry is identical to the existing head, using its frozen scaler.
    baseline=np.stack([predict_features(head,np.stack([f]*3)) for f in v])
    baseline_metrics=metrics(baseline,z,words,valid_total)
    mean=x.mean((0,1)); scale=x.std((0,1)).clip(1e-5)
    x=(x-mean)/scale; v=(v-mean)/scale
    config={'seed':seed,'epochs':epochs,'patience':5,'optimizer':'AdamW','lr':.001,
            'weight_decay':.0001,'batch_size':32,'hidden':32,'projection':64,'dropout':.2,
            'feature_layer':layer,'vocabulary':words,'sequence_lengths':list(lengths),
            'source_head_sha256':sha256(source/'head.pt'), 'source_features_sha256':sha256(source/'features.jsonl'),
            'preprocessing':head['contract']['preprocessing'],'test_used':False,
            'protocol':'Fixed visual encoder/layer; vary sequence length only. Select best epoch using validation CE.'}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    candidates=[]
    for length in lengths:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        indexes=np.linspace(0,28,length).round().astype(int)
        tx=torch.from_numpy(x[:,indexes]); vx=torch.from_numpy(v[:,indexes]); ty=torch.from_numpy(y)
        model=TemporalHead(len(words)); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best_loss=float('inf'); best_state=None; history=[]; stale=0; started=time.perf_counter()
        directory=output/f'bilstm_t{length}'; directory.mkdir()
        for epoch in range(1,epochs+1):
            model.train(); loss_sum=0.; order=torch.randperm(len(tx))
            for batch in order.split(32):
                optimizer.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(tx[batch]),ty[batch])
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); loss_sum+=loss.item()*len(batch)
            model.eval()
            with torch.inference_mode():
                logits=torch.cat([model(b) for b in vx.split(32)]); val_loss=nn.functional.cross_entropy(logits,torch.from_numpy(z)).item()
            history.append({'epoch':epoch,'train_loss':loss_sum/len(tx),'validation_loss':val_loss,
                            'validation_accuracy':float((logits.argmax(1).numpy()==z).sum()/valid_total)})
            if val_loss<best_loss:
                best_loss=val_loss; best_state=deepcopy(model.state_dict()); best_epoch=epoch; stale=0
            else: stale+=1
            if stale>=5: break
        artifact={'config':config,'sequence_length':length,'mean':torch.from_numpy(mean),
                  'scale':torch.from_numpy(scale),'model_state_dict':model.state_dict()}
        torch.save(artifact,directory/'last.pt')
        model.load_state_dict(best_state); artifact['model_state_dict']=best_state; artifact['best_epoch']=best_epoch
        torch.save(artifact,directory/'best.pt')
        with torch.inference_mode():
            p=torch.cat([model(b) for b in vx.split(32)]).softmax(-1).numpy()
        measured=metrics(p,z,words,valid_total)
        measured.update(name=f'BiLSTM T={length}',epochs=len(history),best_epoch=best_epoch,
                        training_seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),
                        checkpoint_sha256=sha256(directory/'best.pt'),checkpoint_bytes=(directory/'best.pt').stat().st_size)
        (directory/'history.json').write_text(json.dumps(history,indent=2))
        (directory/'validation.json').write_text(json.dumps(measured,indent=2))
        np.savetxt(directory/'validation_confusion.csv',measured['confusion_matrix'],fmt='%d',delimiter=',',header=','.join(words))
        candidates.append(measured); print(json.dumps({k:measured[k] for k in ['name','top1_accuracy','training_seconds']}),flush=True)
    summary={'baseline_validation':baseline_metrics,'candidates':candidates,'test_used':False,
             'promoted':False,'active_checkpoint_changed':False,
             'conclusion':'Validation comparison only. Retain the existing model until a candidate passes a prespecified held-out protocol.'}
    (output/'comparison.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/word_accuracy/avhubert_miracl8_v1')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--epochs',type=int,default=25)
    args=parser.parse_args();run_experiment(args.source,args.out,epochs=args.epochs)
