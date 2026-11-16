"""Read-only CPU inference using the author's actual 500-word research model.

No LRW dataset is downloaded or required for weight loading. Its research-only
license and publisher preprocessing requirements remain applicable. This is not
a measured LRW accuracy benchmark or a validated arbitrary-webcam recognizer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from .vendor.model import Lipreading
from pipeline.word_recognition.common import ROOT,sha256

PACKAGE=Path(__file__).parent
CHECKPOINT=ROOT/'models/external/lrw_tcn/resnet18_dctcn_video.pth'


def build(config):
    if config.get('use_boundary') or config.get('backbone_type')!='resnet':raise ValueError('Expected the published video-only non-boundary ResNet18 DC-TCN.')
    options={name:config['densetcn_'+key] for name,key in (
        ('block_config','block_config'),('growth_rate_set','growth_rate_set'),('reduced_size','reduced_size'),
        ('kernel_size_set','kernel_size_set'),('dilation_size_set','dilation_size_set'),
        ('squeeze_excitation','se'),('dropout','dropout'))}
    return Lipreading(modality='video',num_classes=500,backbone_type='resnet',relu_type=config['relu_type'],
                      width_mult=config['width_mult'],use_boundary=False,tcn_options={},densetcn_options=options)


class LRWTemporalClassifier:
    def __init__(self,path=CHECKPOINT):self.path=Path(path);self.model=None;self.vocabulary=[]
    def load(self):
        if self.model is not None:return self
        receipt=json.loads((self.path.parent/'acquisition.json').read_text())
        if sha256(self.path)!=receipt['sha256']:raise ValueError('Author checkpoint fingerprint mismatch.')
        if sha256(PACKAGE/'config.json')!=receipt['config_sha256'] or sha256(PACKAGE/'vocabulary.txt')!=receipt['labels_sha256']:
            raise ValueError('Publisher configuration/ordered labels mismatch.')
        words=[s.strip() for s in (PACKAGE/'vocabulary.txt').read_text().splitlines() if s.strip()]
        if len(words)!=500 or len(set(words))!=500:raise ValueError('Exactly 500 unique ordered publisher labels required.')
        checkpoint=torch.load(str(self.path),map_location='cpu',weights_only=False,mmap=True)
        with torch.device('meta'):model=build(json.loads((PACKAGE/'config.json').read_text()))
        state=checkpoint['model_state_dict']
        if state['tcn.tcn_output.weight'].shape[0]!=len(words):raise ValueError('Checkpoint classifier output does not match the label mapping.')
        torch.nn.Module.load_state_dict(model,state,strict=True,assign=True)
        if any(t.device.type!='cpu' for t in (*model.parameters(),*model.buffers())):raise ValueError('Model has unmaterialized or non-CPU tensors.')
        self.model=model.eval().requires_grad_(False);self.vocabulary=words;self.checkpoint_sha256=receipt['sha256']
        return self
    def predict_rois(self,rois,*,threshold=.7,margin_threshold=.15):
        if not 0<=threshold<=1 or not 0<=margin_threshold<=1:raise ValueError('Rejection thresholds must be in [0,1].')
        self.load();frames=np.asarray(rois)
        if frames.ndim!=3 or not 8<=len(frames)<=76 or min(frames.shape[1:])<88 or not np.isfinite(frames).all():raise ValueError('8..76 measured grayscale mouth ROIs >=88px required.')
        if frames.dtype!=np.uint8:raise ValueError('Use uint8 grayscale pixels, before publisher normalization.')
        h,w=frames.shape[1:];frames=frames[:,(h-88)//2:(h-88)//2+88,(w-88)//2:(w-88)//2+88]
        tensor=torch.from_numpy(np.ascontiguousarray((frames.astype(np.float32)/255.-.421)/.165))[None,None]
        tick=time.perf_counter()
        with torch.inference_mode():p=self.model(tensor,lengths=[len(frames)]).softmax(-1)[0]
        if p.shape!=(500,) or not torch.isfinite(p).all():raise ValueError('Invalid genuine 500-class output.')
        values,indices=p.topk(5);score=float(values[0]);margin=float(values[0]-values[1]);accepted=score>=threshold and margin>=margin_threshold
        return {'model':'author ResNet18 + DC-TCN','task':'isolated_word','vocabulary_size':500,'candidate':self.vocabulary[int(indices[0])],
                'word':self.vocabulary[int(indices[0])] if accepted else None,'uncertain':not accepted,'confidence':score,'margin':margin,
                'probabilities':p.tolist(),
                'top5':[{'word':self.vocabulary[int(i)],'probability':float(v)} for i,v in zip(indices,values)],
                'inference_seconds':time.perf_counter()-tick,'input_shape':list(tensor.shape),'audio_inference':False,
                'checkpoint_sha256':self.checkpoint_sha256,'accuracy_evaluated_locally':False,
                'rejection':'Uncalibrated confidence/margin heuristic; reliable OOV rejection not established.',
                'usage':'Individual research/comparative prototype under author LICENSE; no production/commercial deployment claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);q=p.add_mutually_exclusive_group(required=True);q.add_argument('--mouth-npy',type=Path);q.add_argument('--lrw-centered-video',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    if args.mouth_npy:
        rois=np.load(args.mouth_npy,allow_pickle=False);preprocessing='Caller-provided aligned mouth ROIs; publisher alignment must be established by caller.'
    else:
        cap=cv2.VideoCapture(str(args.lrw_centered_video));frames=[]
        try:
            if cap.get(cv2.CAP_PROP_FPS)!=25:raise ValueError('Fixed crop applies only to standardized 25 FPS LRW input.')
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame.shape[:2]!=(256,256) or len(frames)>=29:raise ValueError('Only native 29x256x256 standardized LRW clips accepted.')
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[115:211,79:175])
        finally:cap.release()
        if len(frames)!=29:raise ValueError('Expected exactly 29 decoded LRW frames.')
        rois=np.stack(frames);preprocessing='Fixed crop of dataset-centered LRW input. Publisher mean-face alignment is not reproduced; this is a smoke test, not benchmark preprocessing.'
    result=LRWTemporalClassifier().predict_rois(rois);result['preprocessing_status']=preprocessing
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
