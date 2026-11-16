"""Real source replay through independent tracking and task adapters; never a feedback dataset import."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from pipeline.adaptive.quality import validate_capture
    from pipeline.adaptive.models import baseline_adapter,jpeg_frames
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing probe results; choose a new output path.')
    manifest=ROOT/'data/processed/multidataset/miracl_v1.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    row=next(r for r in rows if r['task']=='isolated_word' and r['split']=='valid' and r['word_label']=='hello' and len(r['frames'])>=8)
    frames=[]
    for name in row['frames']:
        image=cv2.imread(str(name))
        if image is None:raise ValueError('Genuine source frame is unavailable.')
        frames.append('data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',image)[1]).decode())
    # MIRACL native FPS is unknown. 100ms is a replay scheduling interval only.
    capture={'id':'qa-replay-only','task':'isolated_word','speaker_id':'qa-historical-speaker','session_id':'qa-replay',
             'recorded_at':datetime.now(timezone.utc).isoformat(),'capture_kind':'webcam_replay','consent':True,
             'frames':frames,'timestamps_ms':[100*i for i in range(len(frames))]}
    started=time.perf_counter();meta,archive,roi_bytes=validate_capture(capture,ROOT)
    tracking_ms=1000*(time.perf_counter()-started)
    import io
    rois=np.load(io.BytesIO(roi_bytes),allow_pickle=False)
    outputs={}
    for task in ('isolated_word','alphabet_recognition'):
        adapter=baseline_adapter(task);started=time.perf_counter();adapter.load();load_ms=1000*(time.perf_counter()-started)
        predictions=[adapter.predict(jpeg_frames(rois)) for _ in range(3)]
        outputs[task]={'load_ms':load_ms,'responses':predictions,'mean_adapter_latency_ms':float(np.mean([p['latency_ms'] for p in predictions]))}
    result={'scope':'One genuine historical validation HELLO replay; real independent MediaPipe and model adapters. Not physical-camera or new-population accuracy.',
            'source_id':row['id'],'source_label':row['word_label'],'source_native_fps':None,
            'replay_schedule_ms':100,'feedback_stored':False,'tracking_ms':tracking_ms,
            'source_sha256':meta['source_sha256'],'png_archive_bytes':len(archive),'quality_warnings':meta['warnings'],
            'frames':len(frames),'outputs':outputs}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({task:value['mean_adapter_latency_ms'] for task,value in outputs.items()}))


if __name__=='__main__':main()
