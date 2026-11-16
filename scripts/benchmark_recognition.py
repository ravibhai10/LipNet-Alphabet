"""Real preserved CPU word model + JPEG/ASGI integration on validation clips.

One encoder is loaded once. Baseline/source adapters share those exact weights.
This is an offline clip runtime experiment, not a physical-camera benchmark.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import time
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Choose a new measurement file.')
    from app.services.pretrained_word_inference import PretrainedWordInference
    from app.services.browser_inference import BrowserInference
    from pipeline.word_recognition.common import read_rows,sha256
    from pipeline.lrw_tcn.diagnostics import memory_bytes
    from app import main as api
    # The encoder environment may use a newer Starlette TestClient dependency.
    # Drive the standard ASGI interface directly without altering that runtime.
    class ASGIClient:
        def post(self, path, *, json):
            import json as codec
            payload=codec.dumps(json).encode();messages=[];consumed=False
            async def receive():
                nonlocal consumed
                if consumed:return {'type':'http.disconnect'}
                consumed=True
                return {'type':'http.request','body':payload,'more_body':False}
            async def send(message):messages.append(message)
            scope={'type':'http','method':'POST','path':path,'raw_path':path.encode(),
                   'query_string':b'','headers':[(b'content-type',b'application/json')],
                   'scheme':'http','http_version':'1.1','root_path':'',
                   'server':('127.0.0.1',8000),'client':('127.0.0.1',12345)}
            asyncio.run(api.app(scope,receive,send))
            status=next(m['status'] for m in messages if m['type']=='http.response.start')
            body=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body').decode()
            return SimpleNamespace(status_code=status,text=body,json=lambda:codec.loads(body))
    start=time.perf_counter(); service=PretrainedWordInference()
    if not service.load():raise RuntimeError(service.error)
    load_seconds=time.perf_counter()-start
    rows=read_rows(ROOT/'runs/word_accuracy/avhubert_miracl8_v1/prepared.jsonl')
    selected={}
    for row in rows:
        if row['split']=='valid' and row['role']=='known_word' and row['usable']:
            selected.setdefault(row['word_label'],row)
    # Execute the backed-up service adapter against the SAME in-memory encoder.
    backup=ROOT/'.qa/reliability_20261109/targeted_source_before.zip'
    namespace={'__name__':'app.services._benchmark_before','__package__':'app.services'}
    with zipfile.ZipFile(backup) as archive:
        exec(compile(archive.read('app/services/pretrained_word_inference.py'),'baseline_service','exec'),namespace)
    before=namespace['PretrainedWordInference']();before.model=service.model;before.report=service.report;before._attempted=True
    api.word_inference=service
    observations=[]
    with nullcontext(ASGIClient()) as client:
        for label,row in selected.items():
            if sha256(row['roi'])!=row['roi_sha256']:raise ValueError('Validation ROI changed.')
            rois=np.load(row['roi'],allow_pickle=False);frames=[]
            for roi in rois:
                ok,jpeg=cv2.imencode('.jpg',roi,[cv2.IMWRITE_JPEG_QUALITY,75])
                if not ok:raise RuntimeError('JPEG encoding failed.')
                frames.append(base64.b64encode(jpeg).decode())
            # Alternate order to reduce systematic warm/cache ordering bias.
            entries={}
            def baseline():
                frozen=service.model.policy
                service.model.policy=json.loads((service.path.parent/'rejection_policy.json').read_text())
                api.word_inference=before
                tick=time.perf_counter()
                try:
                    response=client.post('/api/predict/word',json={'frames':frames})
                    if response.status_code!=200:raise RuntimeError(response.text)
                    result=response.json()
                finally:
                    service.model.policy=frozen
                    api.word_inference=service
                entries['before']={'wall_ms':(time.perf_counter()-tick)*1000,'result':result}
            def improved():
                tick=time.perf_counter();response=client.post('/api/predict/word',json={'frames':frames})
                if response.status_code!=200:raise RuntimeError(response.text)
                entries['after']={'wall_ms':(time.perf_counter()-tick)*1000,'result':response.json()}
            for operation in ([baseline,improved] if len(observations)%2==0 else [improved,baseline]):operation()
            equality=entries['before']['result']['all_probabilities']==entries['after']['result']['all_probabilities']
            if not equality:raise ValueError('Protected model probabilities changed.')
            observations.append({'source_id':row['id'],'source_sha256':row['source_sha256'],'label':label,
                                 'probabilities_identical':equality,**entries})
            print(label,entries['after']['wall_ms'],flush=True)
        invalid=client.post('/api/predict/word',json={'frames':['corrupted']*8})
        assert invalid.status_code==422
    alpha=BrowserInference();tick=time.perf_counter();alpha_ready=alpha.load();alpha_load=(time.perf_counter()-tick)*1000
    alphabet=[]
    if alpha_ready:
        for _ in range(3):alphabet.append(alpha.predict(frames))
    report={'scope':'Eight genuine validation mouth clips; backed-up service vs updated service through the same real ASGI route, shared frozen model. '
                    'No physical camera, no accuracy claim for these selected smoke samples. Alphabet uses word ROIs only as a runtime probe.',
            'model_load_seconds_including_warmup':load_seconds,'observations':observations,
            'mean_before_adapter_ms':float(np.mean([r['before']['wall_ms'] for r in observations])),
            'mean_after_asgi_ms':float(np.mean([r['after']['wall_ms'] for r in observations])),
            'p95_after_asgi_ms':float(np.percentile([r['after']['wall_ms'] for r in observations],95)),
            'memory':memory_bytes(),'head_sha256':sha256(service.path),
            'alphabet_load_ms':alpha_load,'alphabet_probes':alphabet,'invalid_video_status':invalid.status_code,
            'encoder_bytes':(ROOT/'models/external/avhubert/base_vox_iter5.pt').stat().st_size,
            'head_bytes':service.path.stat().st_size,'confidence_calibrated':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'observations','alphabet_probes'}},indent=2))


if __name__=='__main__':main()
