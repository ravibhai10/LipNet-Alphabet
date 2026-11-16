"""Opt-in research word backend; preserves the alphabet model and legacy selection."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import threading
import time
import numpy as np
from pipeline.word_recognition.common import ROOT,sha256
from pipeline.word_recognition.pretrained_inference import AVHubertWordClassifier
from .browser_inference import BrowserInference


class PretrainedWordInference:
    def __init__(self,path=None):
        self.path=Path(path or os.environ.get('LIPNET_AVHUBERT_WORD_HEAD',ROOT/'runs/word_accuracy/avhubert_miracl8_v1/head.pt'))
        self.model=None;self.error=None;self._load_lock=threading.Lock();self._predict_lock=threading.Lock();self._attempted=False
    def _make_classifier(self):
        return AVHubertWordClassifier(self.path)
    def load(self):
        with self._load_lock:
            if self._attempted:return self.model is not None
            self._attempted=True
            try:
                report=json.loads((self.path.parent/'evaluation.json').read_text())
                personal = self.path.name == 'head.pt' and self.path.parent.name == 'personal_words'
                minimum_samples = 4 if personal else 100
                minimum_accuracy = .5 if personal else .7
                if personal and report.get('eligible_for_manual_research_activation') is not True:
                    raise ValueError('Personal head lacks a successful paired held-out activation decision.')
                if report['head_sha256']!=sha256(self.path) or report['test_samples']<minimum_samples or report['top1_accuracy']<minimum_accuracy:
                    raise ValueError('Word-head held-out evidence does not pass the research activation gate.')
                if any(v['recall']<.2 for v in report['per_word'].values()):raise ValueError('A supported word lacks acceptable held-out recall.')
                self.model=self._make_classifier().load();self.report=report
                frozen_policy=report.get('rejection_policy',{})
                if 'threshold' in frozen_policy and 'margin' in frozen_policy:
                    self.policy_source='frozen_evaluation'
                    self.policy_file_differs=any(self.model.policy[k]!=frozen_policy[k] for k in ('threshold','margin'))
                    if any(not 0<=frozen_policy[k]<=1 for k in ('threshold','margin')):
                        raise ValueError('Invalid evaluated rejection policy.')
                    self.model.policy={**self.model.policy,**{k:frozen_policy[k] for k in ('threshold','margin')}}
                else:
                    self.policy_source='head_bound_policy_file';self.policy_file_differs=False
                return True
            except Exception as error:
                logging.getLogger(__name__).warning('Pretrained word model could not load: %s', error)
                self.model=None;self.error=str(error);return False
    def health(self):
        ready=self.load();words=self.model.head['vocabulary'] if ready else []
        return {'model_loaded':ready,'inference_ready':ready,'model_kind':'AV-HuBERT + supervised whole-word head (research)',
                'task':'isolated_word','device':'cpu','vocabulary':words,'vocabulary_size':len(words),'checkpoint':self.path.name,
                'error':'The pretrained word model is unavailable. Check the local backend log.' if not ready else None,'production_ready':False,'audio_inference':False,
                'held_out_accuracy':self.report['top1_accuracy'] if ready else None,
                'evaluation_protocol':self.report.get('protocol','miracl_original_held_out') if ready else None,
                'evaluation_samples':self.report['test_samples'] if ready else None,
                'evaluation_correct':self.report.get('correct') if ready else None,
                'research_activation_gate_passed':ready,
                'policy_source':getattr(self,'policy_source',None),
                'policy_file_differs_from_evaluation':getattr(self,'policy_file_differs',False),
                'rejection_policy':getattr(self.model,'policy',None) if ready else None,
                'out_of_vocabulary_detection':'Validation phrase rejection only; unseen isolated-word and webcam accuracy remain unverified.'}
    def predict(self,frames):
        if not self._predict_lock.acquire(blocking=False):raise BlockingIOError('Word inference busy; retain only the latest capture window.')
        request_started=time.perf_counter()
        try:
            crops=BrowserInference.decode_jpegs(frames)
            if any(c.shape!=(96,96) for c in crops):raise ValueError('This head requires the measured 96px alignment convention used during training.')
            if not self.load():raise RuntimeError(self.error or 'Word inference unavailable.')
            started=time.perf_counter();result=self.model.predict_rois(np.stack(crops));p=np.asarray(result['probabilities']);words=result['vocabulary'];order=np.argsort(-p)
            candidate = result['candidate']
            active_word = result['word']
            consensus = result.get('temporal_consensus')
            if consensus is not None and not consensus['confirmed']:
                active_word = None
            uncertain = result['uncertain'] or active_word is None
            return {'task':'isolated_word','predicted_word':candidate,'word':active_word,
                    'display_word':candidate if active_word else 'UNCERTAIN — PLEASE TRY AGAIN','confidence':result['confidence'],'uncertain':uncertain,
                    'top_predictions':[{'word':words[i],'prob':float(p[i])} for i in order[:5]],
                    'all_probabilities':[{'word':w,'prob':float(v)} for w,v in zip(words,p)],
                    'diagnostics':{'margin':result['margin'],'entropy':float(-(p*np.log(p.clip(1e-12))).sum()),
                                   'confidence_threshold':self.model.policy['threshold'],'rejection_status':'REJECTED_UNCERTAIN' if uncertain else 'ACCEPTED',
                                   'temporal_consensus':consensus},
                    'temporal':{'received_frames':len(frames),'model_target_frames':29,'input_spatial':[96,96]},
                    'latency_ms':round(1000*(time.perf_counter()-request_started),2),'model_name':'AV-HuBERT whole-word research head',
                    'model_version':result.get('head_sha256'),
                    'timings_ms':{**result.get('timings_ms',{}),'jpeg_decode_and_readiness':(started-request_started)*1000},
                    'vocabulary':words,'audio_inference':False,'webcam_accuracy_validated':False}
        finally:self._predict_lock.release()


word_inference=PretrainedWordInference()
