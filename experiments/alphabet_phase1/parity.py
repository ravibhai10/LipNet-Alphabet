"""Recorded-source replay preparation and measured browser/OpenCV comparison.

No physical camera, recognizer inference, training or frontend writes.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from .resources import guard

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def ordinal(frames, count=30):
    import numpy as np
    frames = np.asarray(frames, dtype=np.float32)
    positions = np.linspace(0, len(frames)-1, count)
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo+1, len(frames)-1)
    weight = (positions-lo).reshape((count,) + (1,)*(frames.ndim-1))
    return (frames[lo]*(1-weight)+frames[hi]*weight).astype(np.float32)


def pixel_metrics(first, second):
    """SSIM: local 11x11 Gaussian, sigma=1.5, data range 255, valid center."""
    import cv2
    import numpy as np
    a, b = np.asarray(first,dtype=np.float64), np.asarray(second,dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 11:
        raise ValueError('Matching grayscale images >=11px required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite pixels required')
    blur = lambda value: cv2.GaussianBlur(value,(11,11),1.5)[5:-5,5:-5]
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    ssim = ((2*ma*mb+c1)*(2*cov+c2)/((ma*ma+mb*mb+c1)*(va+vb+c2))).mean()
    delta = np.abs(a-b)
    return {'mae_0_255':float(delta.mean()),'max_error_0_255':float(delta.max()),'ssim':float(ssim)}


def prepare(output):
    resources = guard(512)
    if output.exists():
        raise FileExistsError('Existing replay outputs are protected')
    output.mkdir(parents=True)
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from tracking.mouth_canonicalization import aligned_mouth_crop
    cv2.setNumThreads(1)
    grid_base = ROOT/'data/processed/grid_multispeaker_v1'
    grid = [json.loads(line) for line in (grid_base/'train.jsonl').open()]
    miracl = [json.loads(line) for line in (ROOT/'data/processed/multidataset/miracl_v1.jsonl').open()]
    selected = []
    for speaker in ['grid_T01','grid_T06']:
        row = max((r for r in grid if r['speaker_id']==speaker),key=lambda r:r['end_frame']-r['start_frame'])
        selected.append(('grid',row))
    for speaker, word in [('F01','choose'),('F02','previous')]:
        selected.append(('miracl',next(r for r in miracl if r['speaker_id']==speaker and r['word_label']==word and r['repetition']==1)))
    sequences = []
    model = ROOT/'frontend/public/models/face_landmarker.task'
    for dataset, row in selected:
        guard(256)
        key = row['id'].replace(':','_').replace('/','_')
        directory = output/key
        directory.mkdir()
        if dataset == 'grid':
            source = (grid_base/row['video']).resolve()
            if sha(source) != row['video_sha256']:
                raise ValueError('GRID source hash changed')
            cap = cv2.VideoCapture(str(source))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames, decoder_times = [], []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                decoder_times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
                if len(frames)>100:
                    raise ValueError('Unexpected long GRID source')
            cap.release()
            if len(frames)!=75 or fps!=25:
                raise ValueError('Replay GRID decode contract changed')
            original_hash = row['video_sha256']
        else:
            frames = [cv2.imread(path) for path in row['frames']]
            if any(f is None for f in frames):
                raise ValueError('MIRACL source did not decode')
            digest = hashlib.sha256()
            for path in row['frames']:
                digest.update(Path(path).name.encode())
                digest.update(bytes.fromhex(sha(path)))
            if digest.hexdigest()!=row['source_sha256']:
                raise ValueError('MIRACL sequence hash changed')
            original_hash = row['source_sha256']
            fps, decoder_times = None, [None]*len(frames)
        mode = vision.RunningMode.VIDEO if dataset=='grid' else vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=mode,num_faces=1,min_face_detection_confidence=.5,min_face_presence_confidence=.5,min_tracking_confidence=.5)
        entries = []
        with vision.FaceLandmarker.create_from_options(options) as tracker:
            for index, frame in enumerate(frames):
                timestamp = index*40 if fps else round(index*1000/15)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                result = tracker.detect_for_video(image,timestamp) if dataset=='grid' else tracker.detect(image)
                points = np.array([(p.x,p.y,p.z) for p in result.face_landmarks[0]]) if result.face_landmarks else None
                image_path = directory/f'frame_{index:03d}.png'
                if not cv2.imwrite(str(image_path),frame):
                    raise OSError('Replay PNG write failed')
                crop_name = None
                if points is not None:
                    crop = cv2.cvtColor(aligned_mouth_crop(frame,points),cv2.COLOR_BGR2GRAY)
                    crop_path = directory/f'offline_{index:03d}.png'
                    if not cv2.imwrite(str(crop_path),crop):
                        raise OSError('Offline crop write failed')
                    crop_name = crop_path.relative_to(output).as_posix()
                entries.append({'index':index,'frame':image_path.relative_to(output).as_posix(),
                    'frame_sha256':sha(image_path),'width':frame.shape[1],'height':frame.shape[0],
                    'native_nominal_timestamp_ms':index*1000/fps if fps else None,
                    'decoder_timestamp_ms':decoder_times[index],'replay_timestamp_ms':timestamp,
                    'offline_landmarks':points.tolist() if points is not None else None,'offline_crop':crop_name})
        sequences.append({'key':key,'id':row['id'],'dataset':dataset,'speaker':row['speaker_id'],
            'split':'train','source_sha256':original_hash,'native_fps':fps,
            'segment':[row['start_frame'],row['end_frame']],
            'timing_policy':'native nominal 25Hz' if fps else 'controlled 15Hz image replay; native FPS unknown',
            'offline_detector_mode':mode.name,'frames':entries})
        print(f'Prepared {key}: {len(entries)} authentic frames',flush=True)
    save(output/'manifest.json',{'scope':'Four deterministic training recordings, deliberately long GRID letter slots; not representative accuracy data',
        'resources':resources,'mediapipe_python_version':mp.__version__,'landmarker_sha256':sha(model),
        'source_code_hashes':{name:sha(ROOT/name) for name in ['tracking/mouth_canonicalization.py','frontend/src/live-demo/mouthUtils.ts','frontend/src/live-demo/useDemoPipeline.ts','app/services/browser_inference.py']},
        'sequences':sequences})


def analyze(output):
    import cv2
    import numpy as np
    manifest = json.loads((output/'manifest.json').read_text())
    results, summaries = [], []
    def decode(value):
        if not value:
            return None
        return cv2.imdecode(np.frombuffer(base64.b64decode(value.split(',',1)[1]),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    for sequence in manifest['sequences']:
        replay = json.loads((output/f"browser_{sequence['key']}.json").read_text())
        displacement, invalid_offline, invalid_browser = [], 0, 0
        comparisons = {'shared_landmarks_canvas_png':[], 'shared_landmarks_canvas_jpeg':[], 'actual_browser_jpeg':[]}
        offline_segment, browser_segment = [], []
        start, end = sequence['segment']
        frame_reports = []
        for source, browser in zip(sequence['frames'], replay['frames'], strict=True):
            if source['index']!=browser['index']:
                raise ValueError('Frame ordering changed')
            offline = cv2.imread(str(output/source['offline_crop']),cv2.IMREAD_GRAYSCALE) if source['offline_crop'] else None
            actual = decode(browser.get('actual_jpeg'))
            invalid_offline += offline is None
            invalid_browser += actual is None
            frame_result = {'index':source['index'],'offline_valid':offline is not None,'browser_valid':actual is not None}
            for key, value in [('shared_landmarks_canvas_png','shared_png'),('shared_landmarks_canvas_jpeg','shared_jpeg'),('actual_browser_jpeg','actual_jpeg')]:
                image = decode(browser.get(value))
                if offline is not None and image is not None:
                    metric = pixel_metrics(offline,image)
                    comparisons[key].append(metric)
                    frame_result[key] = metric
            a, b = source['offline_landmarks'], browser.get('landmarks')
            if a is not None and b is not None and len(a)==len(b):
                distance = np.linalg.norm((np.asarray(a)[:,:2]-np.asarray(b)[:,:2])*[source['width'],source['height']],axis=1)
                displacement.extend(distance.tolist())
            if start<=source['index']<end and offline is not None and actual is not None:
                offline_segment.append(cv2.resize(offline,(46,140),interpolation=cv2.INTER_LINEAR))
                browser_segment.append(cv2.resize(actual,(46,140),interpolation=cv2.INTER_AREA))
            if offline is not None:
                frame_result['offline_intensity'] = {'mean':float(offline.mean()),'std':float(offline.std())}
            if actual is not None:
                frame_result['browser_intensity'] = {'mean':float(actual.mean()),'std':float(actual.std())}
            frame_reports.append(frame_result)
        aggregated = {key:({'paired_frames':len(values),'mae_0_255':float(np.mean([v['mae_0_255'] for v in values])),
            'max_error_0_255':max(v['max_error_0_255'] for v in values),'mean_ssim':float(np.mean([v['ssim'] for v in values]))} if values else None) for key,values in comparisons.items()}
        nominal = [r['replay_timestamp_ms'] for r in sequence['frames']]
        used, last = [], float('-inf')
        for index, timestamp in enumerate(nominal):
            if timestamp-last>=66:
                used.append(index);last=timestamp
        segment_samples = sum(start<=index<end for index in used)
        tensor_metrics = None
        if offline_segment:
            off = np.rint(ordinal(offline_segment)).clip(0,255)
            live = ordinal(browser_segment)
            delta = np.abs(off-live)
            tensor_metrics = {'mae_0_255':float(delta.mean()),'max_error_0_255':float(delta.max()),
                'paired_valid_source_frames':len(offline_segment),'target_frames':30,
                'scope':'Paired indices only, offline linear resize/rounded temporal versus live area resize/unrounded temporal; not full model inference'}
        summaries.append({'id':sequence['id'],'dataset':sequence['dataset'],'frames':len(frame_reports),
            'invalid_offline_percent':100*invalid_offline/len(frame_reports),
            'invalid_browser_percent':100*invalid_browser/len(frame_reports),
            'landmark_mean_displacement_pixels':float(np.mean(displacement)) if displacement else None,
            'landmark_max_displacement_pixels':float(np.max(displacement)) if displacement else None,
            'pixel_comparisons':aggregated,'alphabet_tensor_comparison':tensor_metrics,
            'timing':{'native_fps':sequence['native_fps'],'replay_timing_policy':sequence['timing_policy'],
                'nominal_browser_minus_replay_timestamp_max_ms':max(abs(a['replay_timestamp_ms']-b['timestamp_ms']) for a,b in zip(sequence['frames'],replay['frames'])),
                'scripted_66ms_cadence_segment_frames':segment_samples,'native_segment_frames':end-start,
                'api_minimum_8_frames_passes_at_scripted_cadence':segment_samples>=8,
                'not_measured':'Physical camera delivery, browser long tasks, live FPS or wall-clock replay cadence'} })
        results.append({'id':sequence['id'],'frames':frame_reports})
    save(output/'frame_metrics.json',results)
    save(output/'parity_report.json',{'scope':'Actual native MediaPipe/OpenCV versus headless Chrome MediaPipe/WASM/Canvas on identical decoded training images. Not a webcam or accuracy evaluation.',
        'clips':summaries,'training_inference_compatible':False,
        'reason':'Legacy training crop corpus unavailable; detector/raster/timing differences need validation, W missing. Pixel similarity alone cannot certify a correct recognition contract.',
        'ssim_definition':'Gaussian local 11x11 sigma1.5, population variance, valid center, data range255',
        'all_source_frames_retained':True,'metrics_denominator':'All source frames for invalid percentages; paired-valid frames for pixel/landmark errors'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.output.resolve()) if args.stage=='prepare' else analyze(args.output.resolve())
