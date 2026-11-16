"""Personal 8-word adaptation: import -> extract -> train -> one locked evaluation.

Uses the existing frozen layer-9 AV-HuBERT features and 8-way linear head.
The original MIRACL model and its historical test results are never overwritten.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from .common import atomic_json, read_rows, sha256, write_rows
from .personal_data import WORDS, ingest, extract, load_contract
from .pretrained import pool, sequence_probabilities


def feature(row):
    if sha256(row['feature']) != row['feature_sha256']:
        raise ValueError('Personal feature bytes changed.')
    x = np.load(row['feature'],allow_pickle=False)
    if x.shape != (29,768) or not np.isfinite(x).all():
        raise ValueError('Invalid personal encoder features.')
    return x


def baseline_for(contract):
    head = torch.load(contract['baseline_head'],map_location='cpu',weights_only=True)
    if head['format'] != 'lipnet_avhubert_word_linear_v1' or head['feature_layer'] != 9 or head['pooling'] != 'mean_std' or head['vocabulary'] != WORDS:
        raise ValueError('This personal workflow requires the verified existing layer-9 mean/std eight-word baseline.')
    return head


def score(head, rows):
    probabilities = np.stack([sequence_probabilities(head,feature(row)) for row in rows])
    labels = np.array([WORDS.index(row['label']) for row in rows])
    correct = int((probabilities.argmax(1)==labels).sum())
    loss = float(-np.log(probabilities[np.arange(len(labels)),labels].clip(1e-12)).mean())
    return probabilities,labels,{'correct':correct,'samples':len(rows),'accuracy':correct/len(rows),'log_loss':loss}


def train(run):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    run = Path(run);contract,_ = load_contract(run)
    if any((run/name).exists() for name in ('selection.json','head.pt','test_access.json','evaluation.json')):
        raise FileExistsError('Selected/evaluated artifacts are protected. Do not retune against the held-out test.')
    baseline = baseline_for(contract);rows = read_rows(run/'features.jsonl')
    if {r['id'] for r in rows} != {r['id'] for r in read_rows(run/'recordings.jsonl')}:
        raise ValueError('Personal feature extraction is incomplete.')
    training = [r for r in rows if r['split']=='train'];valid = [r for r in rows if r['split']=='valid']
    # Reuse only genuine MIRACL TRAINING features, never old validation/test rows.
    original = [r for r in read_rows(contract['base_training_features']) if r['split']=='train' and r['role']=='known_word' and r['usable']]
    x=[];labels=[]
    for row in original:
        if sha256(row['feature']) != row['feature_sha256']:
            raise ValueError('Preserved MIRACL training features changed.')
        value = np.load(row['feature'],allow_pickle=False)
        if value.shape != (3,29,768) or not np.isfinite(value).all():
            raise ValueError('Invalid baseline training cache.')
        x.append(pool(value[1],'mean_std'));labels.append(WORDS.index(row['word_label']))
    if not x or set(labels) != set(range(8)):
        raise ValueError('All eight genuine MIRACL training classes are required.')
    old_count = len(x)
    x.extend(pool(feature(r),'mean_std') for r in training);labels.extend(WORDS.index(r['label']) for r in training)
    x=np.stack(x);labels=np.asarray(labels)
    valid_x=np.stack([pool(feature(r),'mean_std') for r in valid]);valid_y=np.array([WORDS.index(r['label']) for r in valid])
    _,_,base_validation=score(baseline,valid)
    selected={'source':'preserved_baseline','C':None,'personal_weight':0,**base_validation}
    candidates=[dict(selected)];best_head=baseline;start=time.perf_counter()
    with threadpool_limits(limits=1):
        scaler=StandardScaler().fit(x);scaled=scaler.transform(x);vx=scaler.transform(valid_x)
        for c in (.1,1.,10.):
            for personal_weight in (1.,4.,10.):
                sample_weight=np.ones(len(x));sample_weight[old_count:]=personal_weight
                classifier=LogisticRegression(C=c,solver='lbfgs',max_iter=600,class_weight='balanced',random_state=42)
                classifier.fit(scaled,labels,sample_weight=sample_weight)
                if classifier.classes_.tolist()!=list(range(8)):
                    raise ValueError('Wrong trained output order.')
                p=classifier.predict_proba(vx);correct=int((p.argmax(1)==valid_y).sum())
                record={'source':'personal_plus_miracl_train','C':c,'personal_weight':personal_weight,'correct':correct,'samples':len(valid),
                        'accuracy':correct/len(valid),'log_loss':float(-np.log(p[np.arange(len(valid_y)),valid_y].clip(1e-12)).mean())}
                candidates.append(record);print(json.dumps(record),flush=True)
                if (record['accuracy'],-record['log_loss'])>(selected['accuracy'],-selected['log_loss']):
                    selected=record
                    best_head={**baseline,'mean':torch.tensor(scaler.mean_,dtype=torch.float32),'scale':torch.tensor(scaler.scale_,dtype=torch.float32),
                               'weight':torch.tensor(classifier.coef_,dtype=torch.float32),'bias':torch.tensor(classifier.intercept_,dtype=torch.float32)}
    artifact={**best_head,'contract':{**baseline['contract'],'personal_protocol':contract,'task':'personal_cross_session'},'selection':selected}
    # Check the actual serialized float32 geometry before freezing selection.
    sequence_probabilities(artifact,feature(valid[0]))
    torch.save(artifact,run/'head.pt')
    policy=json.loads((Path(contract['baseline_head']).parent/'rejection_policy.json').read_text())
    atomic_json(run/'rejection_policy.json',{**policy,'head_sha256':sha256(run/'head.pt'),
        'source':'Inherited baseline thresholds, not retuned on personal test; uncalibrated and OOV unverified.'})
    result={'selected':selected,'candidates':candidates,'test_used':False,'head_sha256':sha256(run/'head.pt'),
            'features_manifest_sha256':sha256(run/'features.jsonl'),'personal_protocol_sha256':sha256(run/'personal_protocol.json'),
            'baseline_validation':base_validation,'original_training':old_count,'personal_training':len(training),'seconds':time.perf_counter()-start}
    atomic_json(run/'selection.json',result)
    return result


def metrics(probabilities, labels):
    from sklearn.metrics import classification_report,confusion_matrix
    guess=probabilities.argmax(1);correct=int((guess==labels).sum())
    report=classification_report(labels,guess,labels=list(range(8)),target_names=WORDS,output_dict=True,zero_division=0)
    matrix=confusion_matrix(labels,guess,labels=list(range(8)))
    return {'test_samples':len(labels),'correct':correct,'top1_accuracy':correct/len(labels),
            'top5_accuracy':float(np.mean([y in np.argsort(-p)[:5] for p,y in zip(probabilities,labels)])),
            'macro_f1':report['macro avg']['f1-score'],'single_word_wer':1-correct/len(labels),
            'confusion_matrix':matrix.tolist(),'classification_report':report,
            'per_word':{w:{**report[w],'correct':int(matrix[i,i]),'total':int((labels==i).sum())} for i,w in enumerate(WORDS)}}


def evaluate(run):
    run=Path(run);contract,_=load_contract(run);selection=json.loads((run/'selection.json').read_text())
    if (run/'evaluation.json').exists():
        raise FileExistsError('This held-out evaluation was already accessed. Preserve it; do not retune.')
    if selection['head_sha256']!=sha256(run/'head.pt') or selection['features_manifest_sha256']!=sha256(run/'features.jsonl') or selection['personal_protocol_sha256']!=sha256(run/'personal_protocol.json'):
        raise ValueError('Frozen selection/protocol/features changed.')
    # Claim the one-time test before reading any test feature values.
    access={'head_sha256':selection['head_sha256'],'selection_sha256':sha256(run/'selection.json'),
            'features_manifest_sha256':selection['features_manifest_sha256'],'purpose':'One final comparison on untouched personal recording sessions.'}
    if (run/'test_access.json').exists():
        if json.loads((run/'test_access.json').read_text())!=access:
            raise ValueError('Cannot resume evaluation after changing the frozen selection/features.')
    else:atomic_json(run/'test_access.json',access)
    head=torch.load(run/'head.pt',map_location='cpu',weights_only=True);baseline=baseline_for(contract)
    test=[r for r in read_rows(run/'features.jsonl') if r['split']=='test']
    actual,y,_=score(head,test);before,by,_=score(baseline,test)
    if not np.array_equal(y,by):
        raise ValueError('Paired evaluation targets differ.')
    result=metrics(actual,y);baseline_report=metrics(before,y)
    eligible=result['test_samples']>=100 and result['top1_accuracy']>=.7 and all(v['recall']>=.2 for v in result['per_word'].values()) and result['correct']>=baseline_report['correct']
    report={**result,'task':'Personalized eight-word visual recognition; same participant(s), separate test sessions',
            'vocabulary':WORDS,'protocol':'personal_cross_session','speaker_independent':False,'participants':contract['audit']['participants'],
            'head_sha256':selection['head_sha256'],'baseline_head_sha256':contract['baseline_head_sha256'],
            'baseline':baseline_report,'difference_percentage_points':100*(result['top1_accuracy']-baseline_report['top1_accuracy']),
            'test_used_for_selection':False,'eligible_for_manual_research_activation':eligible,'webcam_accuracy_scope':'Only these consented held-out sessions; not unseen-speaker accuracy.',
            'calibrated_confidence':False,'oov_rejection_validated':False,'audio_inference':False,'production_ready':False}
    write_rows(run/'test_predictions.jsonl',[{'id':r['id'],'session':r['session_id'],'speaker':r['speaker_id'],'ground_truth':r['label'],
        'candidate':WORDS[int(p.argmax())],'probabilities':p.tolist(),'baseline_candidate':WORDS[int(b.argmax())],'baseline_probabilities':b.tolist()} for r,p,b in zip(test,actual,before)])
    atomic_json(run/'baseline_evaluation.json',baseline_report)
    for name,data in [('confusion',result),('baseline_confusion',baseline_report)]:
        with (run/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['true/predicted',*WORDS])
            writer.writerows([w,*values] for w,values in zip(WORDS,data['confusion_matrix']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    vmax=max(max(max(row) for row in result['confusion_matrix']),max(max(row) for row in baseline_report['confusion_matrix']))
    for ax,data,title in zip(axes,[baseline_report,result],['Preserved baseline','Validation-selected personal model']):
        ax.imshow(data['confusion_matrix'],cmap='Blues',vmin=0,vmax=vmax)
        ax.set(xticks=range(8),yticks=range(8),xticklabels=WORDS,yticklabels=WORDS,xlabel='Predicted word',ylabel='Actual word',title=f'{title}\n{data["correct"]}/{data["test_samples"]} correct')
        ax.tick_params(axis='x',rotation=55)
        for i,row in enumerate(data['confusion_matrix']):
            for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>vmax*.5 else 'black')
    fig.tight_layout();fig.savefig(run/'paired_confusion.png',dpi=220);fig.savefig(run/'paired_confusion.svg');plt.close(fig)
    atomic_json(run/'evaluation.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['import','extract','train','evaluate'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--bundles',type=Path,nargs='+')
    parser.add_argument('--minutes',type=float,default=30)
    args=parser.parse_args()
    if not 0<args.minutes<=120:parser.error('Use a stage budget of 0–120 minutes.')
    if args.stage=='import':
        if not args.bundles:parser.error('Pass the actual exported recording JSON with --bundles.')
        result=ingest(args.bundles,args.run)
    elif args.stage=='extract':result=extract(args.run,args.minutes)
    elif args.stage=='train':result=train(args.run)
    else:result=evaluate(args.run)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
