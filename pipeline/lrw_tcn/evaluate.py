"""Immutable, provenance-checked 500-output evaluation; partial coverage stays explicit."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time
import cv2
import numpy as np
import torch
from pipeline.lrw_tcn.inference import LRWTemporalClassifier
from pipeline.lrw_tcn.preprocessing import align_sequence,MP68,WEB_CONVENTION
from pipeline.lrw_tcn.diagnostics import Measurement
from pipeline.word_recognition.common import sha256,ROOT


def load_source(row,base):
    """Only genuine local bytes: video, ordered RGB files or aligned uint8 ROI NPY."""
    if row.get('data_source') not in ('official_public','authorized_dataset','user_recording'):
        raise ValueError('Explicit authentic-source provenance required.')
    kind=row.get('input_kind')
    paths=[(base/p).resolve() for p in row['frames']] if kind=='image_sequence' else [(base/row['path']).resolve()]
    if not paths or any(not p.is_file() for p in paths):raise ValueError('Source input is missing.')
    hashes=[sha256(p) for p in paths]
    import hashlib
    fingerprint=hashes[0] if len(hashes)==1 else hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest()
    if row.get('source_sha256')!=fingerprint:raise ValueError('Source hash mismatch; annotate actual input bytes before evaluation.')
    if kind=='aligned_mouth_npy':
        frames=np.load(paths[0],allow_pickle=False)
        if frames.dtype!=np.uint8 or frames.ndim!=3 or frames.shape[1:]!=(96,96):raise ValueError('Expected uint8 [T,96,96] mouth ROIs.')
        return frames,fingerprint,{'alignment':'caller-declared','publisher_alignment_verified':False}
    frames=[];fps=None
    if kind=='image_sequence':
        frames=[cv2.imread(str(p)) for p in paths]
        if any(f is None for f in frames):raise ValueError('An image did not decode.')
        fps=row.get('source_fps')
    elif kind in ('raw_video','lrw_centered_video'):
        cap=cv2.VideoCapture(str(paths[0]))
        try:
            fps=float(cap.get(cv2.CAP_PROP_FPS))
            while True:
                ok,frame=cap.read()
                if not ok:break
                frames.append(frame)
                if len(frames)>76:raise ValueError('Only isolated clips of at most 76 frames are supported.')
        finally:cap.release()
    else:raise ValueError('Unsupported evaluation input kind.')
    if not 8<=len(frames)<=76:raise ValueError('Incomplete isolated-word clip.')
    if kind=='lrw_centered_video':
        if fps!=25 or len(frames)!=29 or any(f.shape[:2]!=(256,256) for f in frames):raise ValueError('Fixed-crop shortcut requires standard 29x256x256 25-FPS LRW video.')
        return np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)[115:211,79:175] for f in frames]),fingerprint,{'alignment':'fixed LRW-centered crop; not publisher mean-face alignment','publisher_alignment_verified':False}
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'frontend/public/models/face_landmarker.task')),
                                        running_mode=vision.RunningMode.IMAGE,num_faces=2)
    points=[]
    with vision.FaceLandmarker.create_from_options(options) as tracker:
        for frame in frames:
            result=tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            if len(result.face_landmarks)!=1:raise ValueError('Exactly one measured face required; no manufactured landmarks.')
            points.append([[result.face_landmarks[0][i].x,result.face_landmarks[0][i].y] for i in MP68])
    crops,meta=align_sequence([cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames],points)
    meta['source_fps']=fps;meta['temporal_policy']='Preserved source frame order/count; unknown native FPS is not invented.'
    return crops,fingerprint,meta


def metrics(predictions,words):
    matrix=np.zeros((len(words),len(words)),dtype=int);total=correct=top5=failed=0
    accepted=accepted_correct=unknowns=false_accepts=0;per_word={}
    for row in predictions:
        if row['unknown']:
            unknowns+=1;false_accepts+=int(row.get('word') is not None);continue
        total+=1
        if 'error' in row:failed+=1;continue
        actual=words.index(row['label']);predicted=words.index(row['candidate']);matrix[actual,predicted]+=1
        hit=actual==predicted;correct+=int(hit);top5+=int(row['label'] in row['top5'])
        if row['word'] is not None:accepted+=1;accepted_correct+=int(hit)
    for i,word in enumerate(words):
        count=sum(not r['unknown'] and r['label']==word for r in predictions)
        tp=int(matrix[i,i]);fp=int(matrix[:,i].sum())-tp
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/count if count else 0
        per_word[word]={'correct':tp,'total':count,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0}
    active=[m for m in per_word.values() if m['total']]
    return {'known_samples':total,'correct':correct,'top1_on_available_samples':correct/total if total else None,
            'top5_on_available_samples':top5/total if total else None,'failed_inputs':failed,
            'covered_classes':len(active),'total_model_classes':len(words),'full_500_class_benchmark':False,
            'macro_f1_supported_classes':float(np.mean([r['f1'] for r in active])) if active else None,
            'per_word':per_word,'unknown_rejection':{'samples':unknowns,'false_accepts':false_accepts,
            'false_accept_rate':false_accepts/unknowns if unknowns else None,'accepted_known':accepted,'accepted_known_correct':accepted_correct,
            'validated':False},'scope':'Available-sample diagnostic; no overall LRW or webcam accuracy claim.'},matrix


def validate_manifest(rows,words):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Nonempty unique source IDs required.')
    if any(r.get('split')!='test' or not r.get('source_sha256') or not r.get('provenance') or not isinstance(r.get('unknown'),bool) for r in rows):
        raise ValueError('Each row requires split=test, source_sha256, provenance, and explicit boolean unknown.')
    if len({r['source_sha256'] for r in rows})!=len(rows):raise ValueError('Repeated input bytes cannot inflate evaluation counts.')
    if any(not r['unknown'] and r.get('label') not in words for r in rows):raise ValueError('Known labels must match the actual ordered LRW vocabulary.')
    if any(r['unknown'] and r.get('label') in words for r in rows):raise ValueError('A supported vocabulary word cannot be labeled unknown.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Evaluation output is protected; choose a new directory.')
    rows=[json.loads(line) for line in args.manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    words=(Path(__file__).with_name('vocabulary.txt')).read_text().splitlines()
    validate_manifest(rows,words)
    # No fitting or threshold search is implemented here. Historical publisher
    # training provenance cannot be reconstructed from local evaluation inputs.
    torch.set_num_threads(1);model=LRWTemporalClassifier().load();predictions=[]
    for row in rows:
        result={'id':row['id'],'label':row.get('label'),'unknown':row['unknown'],'speaker_id':row.get('speaker_id')}
        measurement=Measurement()
        try:
            rois,fingerprint,preprocessing=load_source(row,args.manifest.parent)
            prediction=model.predict_rois(rois)
            result.update(candidate=prediction['candidate'],word=prediction['word'],confidence=prediction['confidence'],
                          top5=[x['word'] for x in prediction['top5']],source_sha256=fingerprint,preprocessing=preprocessing)
        except (ValueError,OSError) as error:result['error']=str(error)
        result['performance']=measurement.finish();predictions.append(result)
    report,matrix=metrics(predictions,words)
    durations=[r['performance']['wall_ms'] for r in predictions]
    report.update({'checkpoint_sha256':model.checkpoint_sha256,'manifest_sha256':sha256(args.manifest),
                   'speakers':sorted({r['speaker_id'] for r in rows if r.get('speaker_id')}),
                   'unknown_speaker_rows':sum(not r.get('speaker_id') for r in rows),
                   'speaker_disjoint_from_publisher_training_verified':False,
                   'selection':'No training, tuning, calibration or checkpoint selection uses these rows.',
                   'latency_ms':{'median_end_to_end':float(np.median(durations)),'p95_end_to_end':float(np.percentile(durations,95))},
                   'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':torch.get_num_threads()},
                   'audio_inference':False})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions),encoding='utf-8')
    with (args.output/'confusion.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['actual/predicted',*words])
        for word,counts in zip(words,matrix):writer.writerow([word,*counts.tolist()])
    print(json.dumps({k:v for k,v in report.items() if k!='per_word'},indent=2))


if __name__=='__main__':main()
