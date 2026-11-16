"""Research-only 500-word routes; legacy and eight-word routes stay intact."""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.services.lrw500_inference import lrw500_inference
from app.services.word_compute import word_cpu_lock

router=APIRouter()


class LRW500Request(BaseModel):
    frames:list[str]=Field(min_length=8,max_length=76)
    input_kind:Literal['webcam','aligned_mouth']='webcam'
    convention:str=Field(max_length=80)
    landmarks:list[list[list[float]]]=Field(default_factory=list,max_length=76)
    timestamps_ms:list[float]=Field(default_factory=list,max_length=76)
    face_counts:list[int]=Field(default_factory=list,max_length=76)
    final:bool=False


@router.get('/api/word/lrw500/health')
def lrw500_health(load:bool=True):
    return lrw500_inference.health(load=load)


@router.post('/api/predict/word/lrw500')
def predict_lrw500(req:LRW500Request):
    if not word_cpu_lock.acquire(blocking=False):raise HTTPException(429,'A word model is already processing a CPU request.')
    try:
        result=lrw500_inference.predict(req.frames,input_kind=req.input_kind,landmarks=req.landmarks,
                    timestamps_ms=req.timestamps_ms,face_counts=req.face_counts,convention=req.convention,final=req.final)
        result['utterance_final']=req.final
        if not req.final:result['utterance_warning']='Partial utterance window; boundaries are heuristic.'
        return result
    except ValueError as error:raise HTTPException(422,str(error)) from error
    except BlockingIOError as error:raise HTTPException(429,str(error)) from error
    except RuntimeError as error:
        logging.getLogger(__name__).warning('LRW research inference: %s',error)
        raise HTTPException(503,'500-word model unavailable or loading. Inspect its readiness endpoint.') from error
    except Exception as error:
        logging.getLogger(__name__).exception('Unexpected 500-word inference failure')
        raise HTTPException(500,'Word analysis failed. Retry the word or check the local backend log.') from error
    finally:word_cpu_lock.release()
