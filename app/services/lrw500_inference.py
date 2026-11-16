"""Separate selectable research adapter; never replaces the eight-word model."""
import os
import logging
import json
import threading
import numpy as np
from pipeline.lrw_tcn.inference import LRWTemporalClassifier, PACKAGE, CHECKPOINT
from pipeline.lrw_tcn.preprocessing import prepare_webcam, WEB_CONVENTION
from pipeline.lrw_tcn.diagnostics import Measurement
from app.services.browser_inference import BrowserInference
from pipeline.word_recognition.common import ROOT


class LRW500Inference:
    def __init__(self):
        self.model=None;self.error=None;self.state='not_loaded';self._load_lock=threading.Lock();self._predict_lock=threading.Lock()
        self.threshold=float(os.environ.get('LIPNET_LRW_CONFIDENCE_THRESHOLD','.7'))
        self.margin=float(os.environ.get('LIPNET_LRW_MARGIN_THRESHOLD','.15'))
        if not 0<=self.threshold<=1 or not 0<=self.margin<=1:raise ValueError('LRW rejection thresholds must be in [0,1].')
        self.words=[w.strip() for w in (PACKAGE/'vocabulary.txt').read_text().splitlines() if w.strip()]
    def load(self):
        with self._load_lock:
            if self.model is not None:return True
            if self.state=='failed':return False
            self.state='loading'
            try:
                import torch
                torch.set_num_threads(1)
                measure=Measurement();self.model=LRWTemporalClassifier().load()
                # Discarded initialization only, never prediction/evaluation evidence.
                self.model.predict_rois(np.full((12,96,96),128,dtype=np.uint8),threshold=self.threshold,margin_threshold=self.margin)
                self.load_diagnostics=measure.finish();self.state='ready';return True
            except Exception as error:
                logging.getLogger(__name__).warning('Publisher word model could not load: %s', error)
                self.model=None;self.error=str(error);self.state='failed';return False
    def health(self,load=False):
        available=CHECKPOINT.is_file() and (CHECKPOINT.parent/'acquisition.json').is_file()
        if load and available and self.state=='not_loaded':
            # Claim loading before launch so concurrent health probes cannot spawn duplicates.
            self.state='loading';threading.Thread(target=self.load,daemon=True,name='lrw500-load').start()
        diagnostic_path=ROOT/'runs/lrw500/available_samples_v1/evaluation/evaluation.json'
        diagnostic=None
        if diagnostic_path.is_file():
            try:
                data=json.loads(diagnostic_path.read_text())
                diagnostic={key:data[key] for key in ('known_samples','correct','covered_classes','failed_inputs','unknown_rejection','scope')}
            except (OSError,ValueError,KeyError):pass
        return {'model_loaded':self.model is not None and self.state=='ready','inference_ready':self.state=='ready',
                'available':available,'state':self.state,'model_kind':'Publisher ResNet18 + DC-TCN (500-word research)',
                'vocabulary':self.words,'vocabulary_size':len(self.words),'checkpoint':CHECKPOINT.name,
                'error':'The 500-word model is unavailable. Check the local backend log.' if self.error else None,'audio_inference':False,'production_ready':False,'held_out_accuracy':None,
                'webcam_accuracy_validated':False,'alignment_convention':WEB_CONVENTION,
                'live_policy':'Completed utterances only in the browser; two distinct temporal views must both pass rejection and agree.',
                'accuracy_note':'No local 500-class benchmark. The 94.79% MIRACL result applies only to the separate eight-word model.',
                'available_diagnostic':diagnostic,
                'rejection':{'confidence_threshold':self.threshold,'margin_threshold':self.margin,'validated':False},
                'load_diagnostics':getattr(self,'load_diagnostics',None)}
    def predict(self,frames,*,input_kind='webcam',landmarks=None,timestamps_ms=None,face_counts=None,convention=None,final=False):
        if not self._predict_lock.acquire(blocking=False):raise BlockingIOError('500-word CPU inference is busy; keep only the latest utterance.')
        try:
            measurement=Measurement();images=BrowserInference.decode_jpegs(frames)
            if input_kind=='webcam':
                if convention!=WEB_CONVENTION:raise ValueError('Use the versioned LRW mean-face capture convention, not the eight-word mouth crop.')
                rois,preprocessing=prepare_webcam(images,landmarks or [],timestamps_ms or [],face_counts or [])
            elif input_kind=='aligned_mouth':
                if convention!='caller-aligned-mouth-v1':raise ValueError('Aligned input requires an explicit preprocessing provenance declaration.')
                if any(i.shape!=(96,96) for i in images):raise ValueError('Aligned mouth input must be 96x96 uint8 grayscale.')
                rois=np.stack(images);preprocessing={'alignment_equivalence_verified':False,'limitation':'Caller-provided mouth ROIs; publisher alignment not verified by this endpoint.'}
            else:raise ValueError('Unsupported input kind.')
            if self.state=='loading':raise RuntimeError('500-word model is loading; retry when health reports ready.')
            if not self.load():raise RuntimeError(self.error or '500-word checkpoint unavailable.')
            result=self.model.predict_rois(rois,threshold=self.threshold,margin_threshold=self.margin)
            consensus=None
            if input_kind=='webcam' and final:
                # Two actual temporal contexts from one measured utterance.
                # Never count a duplicated identical request as confirmation.
                peer=self.model.predict_rois(rois[1:-1],threshold=self.threshold,margin_threshold=self.margin) if len(rois)>=10 else None
                agreed=peer is not None and not result['uncertain'] and not peer['uncertain'] and peer['candidate']==result['candidate']
                consensus={'confirmed':bool(agreed),'views':[len(rois),len(rois)-2] if peer else [len(rois)],
                           'peer_candidate':peer['candidate'] if peer else None,'peer_confidence':peer['confidence'] if peer else None,
                           'limitation':'Context-consistency heuristic; no measured accuracy improvement.'}
                if not agreed:result['word']=None;result['uncertain']=True
            metrics=measurement.finish()
            return {'task':'isolated_word','mode':'lrw500','predicted_word':result['candidate'],'word':result['word'],
                    'display_word':result['word'] or 'Uncertain','confidence':result['confidence'],'uncertain':result['uncertain'],
                    'top_predictions':[{'word':i['word'],'prob':i['probability']} for i in result['top5']],
                    'all_probabilities':[{'word':w,'prob':p} for w,p in zip(self.model.vocabulary,result['probabilities'])],
                    'diagnostics':{'margin':result['margin'],'confidence_threshold':self.threshold,'margin_threshold':self.margin,
                                   'rejection_status':'UNCERTAIN' if result['uncertain'] else 'ACCEPTED_UNCALIBRATED','rejection_validated':False,
                                   'temporal_consensus':consensus},
                    'preprocessing':preprocessing,'temporal':{'received_frames':len(frames),'model_target_frames':len(rois)},
                    'latency_ms':metrics['wall_ms'],'performance':metrics,'model_name':result['model'],'vocabulary_size':500,
                    'audio_inference':False,'webcam_accuracy_validated':False,'local_accuracy':None,
                    'checkpoint_sha256':result['checkpoint_sha256']}
        finally:self._predict_lock.release()


lrw500_inference=LRW500Inference()
