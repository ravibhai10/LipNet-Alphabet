"""Evaluate preserved scratch weights on the EXACT fresh test clips, after selection."""
from pathlib import Path
import argparse
import json
import math
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.word_recognition.common import read_rows,sha256,atomic_json,write_rows
from pipeline.word_recognition.legacy_word_model import WordVideoNet
from pipeline.multidataset.experiment import tracked_clip,PREPROCESS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    if (run/'paired_comparison.json').exists():raise FileExistsError('Paired comparison is protected.')
    current=json.loads((run/'evaluation.json').read_text());rows=[r for r in read_rows(run/'sources.jsonl') if r['role']=='known_word' and r['split']=='test']
    path=ROOT/'runs/multidataset/miracl8_cpu_v2/best.pt';baseline=torch.load(path,map_location='cpu',weights_only=False)
    if sha256(path)!='3c7c420d30ac111d4b752322318d92c3b9dfa2077ba2f3d04bc6a71b6aa937ed' or baseline['contract']['model_source_sha256']!=sha256(ROOT/'pipeline/word_recognition/legacy_word_model.py') or baseline['contract']['preprocessing']!=PREPROCESS:raise ValueError('Preserved scratch baseline contract changed.')
    words=baseline['vocabulary']
    if words!=current['vocabulary']:raise ValueError('Model vocabularies differ; no paired comparison possible.')
    torch.set_num_threads(1);model=WordVideoNet(len(words)).eval();model.load_state_dict(baseline['model_state_dict'],strict=True)
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'frontend/public/models/face_landmarker.task')),running_mode=vision.RunningMode.IMAGE,num_faces=1)
    predictions=[];correct=0;started=time.perf_counter()
    with vision.FaceLandmarker.create_from_options(options) as tracker:
        for i,row in enumerate(rows):
            try:
                clip,tracking=tracked_clip(row,tracker,mp)
                with torch.inference_mode():probability=model(torch.from_numpy(clip).float()[None,None]/255.).softmax(-1)[0]
                prediction=words[int(probability.argmax())];ok=prediction==row['word_label']
            except ValueError:
                prediction=None;ok=False
            correct+=int(ok);predictions.append({'id':row['id'],'ground_truth':row['word_label'],'candidate':prediction,'correct_raw':ok})
            if (i+1)%48==0:print(f'Preserved baseline {i+1}/{len(rows)}',flush=True)
    fresh={r['id']:r for r in read_rows(run/'test_predictions.jsonl')}
    if set(fresh)!={r['id'] for r in predictions}:raise ValueError('Test ID mismatch.')
    better=sum(not r['correct_raw'] and fresh[r['id']]['correct_raw'] for r in predictions)
    worse=sum(r['correct_raw'] and not fresh[r['id']]['correct_raw'] for r in predictions)
    report={'same_test_clips':len(rows),'same_vocabulary':words,'preserved_scratch_correct':correct,'preserved_scratch_accuracy':correct/len(rows),
            'pretrained_correct':current['correct'],'pretrained_accuracy':current['top1_accuracy'],
            'accuracy_change_percentage_points':100*(current['top1_accuracy']-correct/len(rows)),
            'old_wrong_new_correct':better,'old_correct_new_wrong':worse,'seconds':time.perf_counter()-started,
            'scratch_checkpoint_sha256':sha256(path),'pretrained_head_sha256':current['head_sha256'],
            'scratch_model_definition':'pipeline/word_recognition/legacy_word_model.py','scratch_model_source_sha256':sha256(ROOT/'pipeline/word_recognition/legacy_word_model.py'),
            'policy':'Same source IDs and native RGB inputs; each pipeline uses its actual trained spatial/temporal preprocessing. Tracking failure counts as incorrect.',
            'limitation':'Three held-out MIRACL speakers, eight words; no broad English or physical webcam accuracy claim.'}
    write_rows(run/'preserved_scratch_test_predictions.jsonl',predictions);atomic_json(run/'paired_comparison.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
