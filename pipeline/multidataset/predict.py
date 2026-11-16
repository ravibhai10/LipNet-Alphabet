"""Offline word inference from genuine MIRACL RGB source frames, not saved logits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from pipeline.word_recognition.common import ROOT,read_rows,sha256
from pipeline.word_recognition.model import WordVideoNet
from .experiment import PREPROCESS,tracked_clip,uncertainty


def predict(run,manifest,sample_id):
    run=Path(run);checkpoint=run/'best.pt'
    # Local trusted training artifact only; never accepts uploaded checkpoint files.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);contract=saved['contract']
    if contract['task']!='isolated_word' or contract['dataset']!='miracl' or contract['preprocessing']!=PREPROCESS:
        raise ValueError('Word model preprocessing/task mismatch.')
    if contract['manifest_sha256']!=sha256(manifest) or contract['landmarker_sha256']!=sha256(ROOT/'frontend/public/models/face_landmarker.task') or contract['model_source_sha256']!=sha256(ROOT/'pipeline/word_recognition/model.py'):
        raise ValueError('Source/landmarker/model contract mismatch.')
    words=saved['vocabulary']
    if words!=contract['vocabulary'] or len(set(words))!=len(words) or len(words)!=8:raise ValueError('Invalid model output vocabulary.')
    candidates=[r for r in read_rows(manifest) if r['id']==sample_id and r['task']=='isolated_word']
    if len(candidates)!=1:raise ValueError('Unique isolated-word source ID required.')
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'frontend/public/models/face_landmarker.task')),
                                        running_mode=vision.RunningMode.IMAGE,num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as tracker:clip,tracking=tracked_clip(candidates[0],tracker,mp)
    torch.set_num_threads(1);model=WordVideoNet(len(words)).eval();model.load_state_dict(saved['model_state_dict'],strict=True)
    with torch.inference_mode():probabilities=model(torch.from_numpy(clip).float()[None,None]/255.).softmax(-1)[0].tolist()
    return {'id':sample_id,'task':'isolated_word','audio_inference':False,'vocabulary':words,'probabilities':probabilities,
            'prediction':uncertainty(probabilities,words),'tracking':tracking,'checkpoint_sha256':sha256(checkpoint),
            'input_shape':[1,1,24,64,64],'production_ready':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--id',required=True)
    args=parser.parse_args();print(json.dumps(predict(args.run,args.manifest,args.id),indent=2))
