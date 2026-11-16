"""Explicit twenty-word experiment; preserves the original eight-word route."""
import logging
import os
from pathlib import Path
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from app.services.pretrained_word_inference import PretrainedWordInference,word_inference
from app.services.word_compute import word_cpu_lock
from pipeline.word_recognition.common import ROOT
from pipeline.word_recognition.pretrained_inference import AVHubertWordClassifier


class ExpandedWordInference(PretrainedWordInference):
    def __init__(self,path=None):
        super().__init__(path or os.environ.get('LIPNET_EXPANDED_WORD_HEAD',ROOT/'runs/word_accuracy/avhubert_combined20_v1/head.pt'))
    def _make_classifier(self):
        # Both heads share identical frozen weights. Reuse the ready encoder to
        # avoid another 400MB allocation on the target 8GB Windows machine.
        shared=word_inference.model.encoder if word_inference.load() else None
        return AVHubertWordClassifier(self.path,expected_vocabulary_size=20,encoder=shared)
    def health(self):
        result=super().health()
        result.update(model_kind='AV-HuBERT + combined 20-word head (research)',
                      out_of_vocabulary_detection='Not validated. Upstream speaker IDs are unknown; clip-disjoint evaluation only.',
                      webcam_accuracy_validated=False)
        if self.model is not None:result['evaluation_by_source']=self.report.get('by_source')
        return result


expanded_word_inference=ExpandedWordInference()
router=APIRouter()


class ExpandedWordRequest(BaseModel):
    frames:list[str]=Field(min_length=8,max_length=76)


@router.get('/api/word/expanded/health')
def expanded_word_health():
    return expanded_word_inference.health()


@router.post('/api/predict/word/expanded')
def predict_expanded_word(req:ExpandedWordRequest):
    if not word_cpu_lock.acquire(blocking=False):raise HTTPException(429,'A word model is already processing a CPU request.')
    try:
        result=expanded_word_inference.predict(req.frames)
        # Abstention must remain explicit; a candidate is not an accepted word.
        if result['uncertain']:result['word']=None;result['display_word']=''
        result['model_name']='AV-HuBERT combined 20-word research head'
        result['evaluation_scope']='MIRACL original partitions plus blocked clip-disjoint upstream split; webcam unmeasured'
        return result
    except ValueError as error:raise HTTPException(422,str(error)) from error
    except BlockingIOError as error:raise HTTPException(429,str(error)) from error
    except RuntimeError as error:
        logging.getLogger(__name__).warning('Expanded word inference unavailable: %s',error)
        raise HTTPException(503,'Twenty-word research model unavailable; inspect its health endpoint.') from error
    finally:word_cpu_lock.release()
