"""Strictly load the supervised AV-HuBERT word head and run real video-only inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .common import ROOT,read_rows,sha256
from .pretrained import ENCODER,PREPROCESS,LAYERS,sequence_probabilities,roi,contract_for
from pipeline.avhubert.loader import AVHubertLoader


class AVHubertWordClassifier:
    def __init__(self,head_path,*,expected_vocabulary_size=None,encoder=None):
        if expected_vocabulary_size is not None and (not isinstance(expected_vocabulary_size, int) or expected_vocabulary_size < 2):
            raise ValueError('Vocabulary size must be at least 2.')
        self.path=Path(head_path);self.head=None;self.encoder=None;self.policy=None
        self.expected_vocabulary_size=expected_vocabulary_size;self._provided_encoder=encoder
    def load(self):
        if self.head is not None:return self
        head=torch.load(self.path,map_location='cpu',weights_only=True)
        words=head['vocabulary']
        if self.expected_vocabulary_size is None and head.get('format') == 'lipnet_avhubert_expanded_word_linear_v1':
            raise ValueError('Word-head contract mismatch: select the separate expanded loader explicitly.')
        if self.expected_vocabulary_size is not None and len(words)!=self.expected_vocabulary_size:
            raise ValueError(f'Word vocabulary size mismatch: expected {self.expected_vocabulary_size}, got {len(words)}.')
        if words!=head['contract']['vocabulary'] or len(set(words))!=len(words) or not all(isinstance(w,str) and w.strip() and len(w.split())==1 for w in words):
            raise ValueError('Invalid actual ordered word vocabulary.')
        if self.expected_vocabulary_size==20:
            from .expanded_data import WORDS,ADAPTER,COMMIT
            if words!=list(WORDS) or head['contract'].get('source_adapters')!={'allenye66':ADAPTER,'miracl':PREPROCESS} or head['contract'].get('upstream_commit')!=COMMIT:
                raise ValueError('Expanded head source/ROI adapter contract mismatch.')
        if head['feature_layer'] not in LAYERS or head['pooling'] not in {'mean','temporal3','mean_std'}:raise ValueError('Unrecognized encoder feature contract.')
        for key in ('mean','scale','weight','bias'):
            if not torch.isfinite(head[key]).all():raise ValueError('Head parameters are non-finite.')
        if (head['scale']<=0).any():raise ValueError('Invalid feature scaling.')
        sequence_probabilities(head,np.zeros((29,768),dtype=np.float32))  # Geometry check only, never recognition evidence.
        if sha256(ENCODER)!=head['contract']['encoder_sha256'] or sha256(ROOT/'frontend/public/models/face_landmarker.task')!=head['contract']['landmarker_sha256']:raise ValueError('Encoder/landmarker fingerprint mismatch.')
        policy=json.loads((self.path.parent/'rejection_policy.json').read_text())
        if policy['head_sha256']!=sha256(self.path) or not 0<=policy['threshold']<=1 or not 0<=policy['margin']<=1:raise ValueError('Confidence-policy checkpoint mismatch.')
        encoder=self._provided_encoder or AVHubertLoader(threads=__import__('app.settings',fromlist=['settings']).settings().cpu_threads,low_memory=True).load()
        if encoder.preprocessing()['image_crop_size']!=88 or encoder.preprocessing()['image_mean']!=.421 or encoder.preprocessing()['image_std']!=.165:raise ValueError('Publisher encoder normalization changed.')
        # Initialize CPU kernels before the first live utterance. This discarded
        # zero-input warm-up is a performance operation, never recognition evidence.
        if self._provided_encoder is None:
            encoder.encode_frames(np.zeros((29,96,96),dtype=np.uint8),source_fps=None,target_frames=29,
                                  output_layer=None if head['feature_layer']==12 else head['feature_layer'])
        self.head,self.encoder,self.policy=head,encoder,policy
        self.head_sha256=sha256(self.path)
        return self
    def predict_rois(self,rois):
        self.load();rois=np.asarray(rois)
        if rois.ndim!=3 or rois.shape[1:]!=(96,96) or rois.dtype!=np.uint8 or not 8<=len(rois)<=76:raise ValueError('8..76 measured eye-axis-aligned uint8 96px mouth crops required.')
        started=time.perf_counter();layer=self.head['feature_layer']
        features=self.encoder.encode_frames(rois,source_fps=None,target_frames=29,output_layer=None if layer==12 else layer)
        encoded=time.perf_counter()
        probabilities=sequence_probabilities(self.head,features);order=np.argsort(-probabilities);confidence=float(probabilities[order[0]]);margin=float(probabilities[order[0]]-probabilities[order[1]])
        accepted=confidence>=self.policy['threshold'] and margin>=self.policy['margin'];words=self.head['vocabulary']
        from .consensus import temporal_consensus
        consistency=temporal_consensus(self.head,features,self.policy)
        consistency['confirmed']=bool(consistency['confirmed'] and consistency['views'][0]['candidate']==words[int(order[0])])
        return {'word':words[int(order[0])] if accepted else None,'candidate':words[int(order[0])],'uncertain':not accepted,'confidence':confidence,
                'margin':margin,'probabilities':probabilities.tolist(),'vocabulary':words,'audio_inference':False,
                'head_sha256':self.head_sha256,'input_shape':[1,1,len(rois),96,96],'model_frames':29,
                'timings_ms':{**getattr(self.encoder,'last_timings_ms',{}),
                              'classification':(time.perf_counter()-encoded)*1000},
                'feature_layer':layer,'inference_seconds':time.perf_counter()-started,
                'temporal_consensus':consistency,
                'limitation':f'{len(words)} supported words; unseen isolated-word rejection and webcam recognition accuracy are not established.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--id',required=True)
    args=p.parse_args();contract,rows=contract_for(args.run);matches=[r for r in rows if r['id']==args.id]
    if len(matches)!=1:raise ValueError('Unique source ID required.')
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'frontend/public/models/face_landmarker.task')),running_mode=vision.RunningMode.IMAGE,num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as tracker:clip,tracking=roi(matches[0],tracker,mp)
    if clip is None:raise ValueError('Real source face tracking failed; no prediction.')
    result=AVHubertWordClassifier(args.run/'head.pt').predict_rois(clip);result.update(source_id=args.id,tracking=tracking)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
