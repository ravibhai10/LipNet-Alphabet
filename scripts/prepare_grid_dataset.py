"""Fully decode authentic GRID video, inventory annotations, and split by speaker.

Frame-bounded raw-video annotations avoid re-encoding or copying source video.
Only the annotated GRID letter slot is classified: usually 25 letters, not A-Z.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_grid_official import sha256, atomic_json
from pipeline.grid.schema import GridManifestEntry


def parse_alignment(path):
    tokens, previous_end = [], 0
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 3:
            raise ValueError(f'Malformed alignment line {number}')
        start, end, word = int(values[0]), int(values[1]), values[2].lower()
        if start < previous_end or end <= start:
            raise ValueError(f'Nonmonotonic alignment line {number}')
        previous_end = end
        if word not in {'sil', 'sp'}:
            tokens.append({'start_units': start, 'end_units': end, 'word': word})
    if len(tokens) != 6 or tokens[0]['word'] not in {'bin','lay','place','set'} or tokens[1]['word'] not in {'blue','green','red','white'} or tokens[2]['word'] not in {'at','by','in','with'} or len(tokens[3]['word']) != 1 or not 'a' <= tokens[3]['word'] <= 'z' or tokens[4]['word'] not in {'zero','one','two','three','four','five','six','seven','eight','nine'} or tokens[5]['word'] not in {'again','now','please','soon'}:
        raise ValueError('Alignment does not match the six-token GRID grammar')
    return tokens, previous_end


def validate_frame_count(decoded, reported):
    # Container duration-based estimates can undercount complete MPEG streams.
    # Actual successful reads control segment bounds; premature EOF is rejected.
    if decoded < 2 or (reported > 0 and decoded < reported):
        raise ValueError(f'Decoding count mismatch: decoded {decoded}, metadata {reported}')


def probe_video(job):
    import cv2
    video, alignment, speaker, root, cached = job
    row = {'speaker': speaker, 'speaker_id': f'grid_T{int(speaker[1:]):02d}',
        'utterance_id': f'T{int(speaker[1:]):02d}_U{video.stem}',
        'video': video.relative_to(root).as_posix(), 'alignment': alignment.relative_to(root).as_posix(),
        'status': 'invalid', 'bytes': video.stat().st_size}
    try:
        row['video_sha256'] = sha256(video)
        if not alignment.is_file():
            raise ValueError('Missing matching annotation')
        row['alignment_sha256'] = sha256(alignment)
        if cached and cached.get('validation_version') == 1 and cached.get('video_sha256') == row['video_sha256'] and cached.get('alignment_sha256') == row['alignment_sha256']:
            return cached
        tokens, annotated_end = parse_alignment(alignment)
        params = [cv2.CAP_PROP_N_THREADS, 1] if hasattr(cv2, 'CAP_PROP_N_THREADS') else []
        cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, params)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError('Video cannot be opened with a valid frame rate')
            visual = hashlib.sha256()
            count, shape = 0, None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.size == 0 or count >= 500:
                    raise ValueError('Invalid frame or unexpectedly long GRID clip')
                shape = list(frame.shape[:2])
                visual.update(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64,64), interpolation=cv2.INTER_AREA).tobytes())
                count += 1
            validate_frame_count(count, reported)
            if annotated_end / 25000 > count / fps + 1 / fps:
                raise ValueError('Annotation extends beyond the decoded video')
        finally:
            cap.release()
        letter = tokens[3]
        first, last = math.floor(letter['start_units']/25000*fps), math.ceil(letter['end_units']/25000*fps)
        if not 0 <= first < last <= count or last-first < 2:
            raise ValueError('Letter segment has fewer than two frames or exceeds source')
        row.update(status='valid', decoded_frames=count, reported_frames=reported, fps=fps, frame_shape=shape,
            metadata_frame_count_matches=count==reported,
            decoded_thumbnail_sha256=visual.hexdigest(), label=letter['word'].upper(),
            start_frame=first, end_frame=last, alignment_timebase_hz=25000,
            start_units=letter['start_units'], end_units=letter['end_units'],
            transcript=' '.join(token['word'] for token in tokens), validation_version=1)
    except Exception as error:
        row['error'] = str(error)
    return row


def deduplicate(rows):
    """Keep deterministic representatives; quarantine conflicting labels entirely."""
    excluded, findings = set(), []
    for field in ('video_sha256', 'decoded_thumbnail_sha256'):
        groups = defaultdict(list)
        for row in rows:
            if row['status']=='valid':
                groups[row[field]].append(row)
        for digest, group in groups.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda row: row['video'])
            conflict = len({row['transcript'] for row in group}) > 1
            removed = group if conflict else group[1:]
            excluded.update(row['video'] for row in removed)
            findings.append({'kind': field, 'fingerprint': digest, 'conflicting_annotations': conflict,
                'videos': [row['video'] for row in group], 'excluded': [row['video'] for row in removed],
                'interpretation': 'byte-identical sources' if field=='video_sha256' else 'matching decoded 64x64 grayscale sequences; conservatively excluded, not asserted identical at native resolution'})
    return [row for row in rows if row['status']=='valid' and row['video'] not in excluded], findings


def assign_speakers(rows):
    speakers = sorted({row['speaker'] for row in rows}, key=lambda value:int(value[1:]))
    if len(speakers) < 3:
        raise ValueError('At least three usable speakers are required before preparing splits')
    count = max(1, round(len(speakers)*0.15))
    train, valid, test = speakers[:-2*count], speakers[-2*count:-count], speakers[-count:]
    assignments = {speaker: split for split, values in [('train',train),('valid',valid),('test',test)] for speaker in values}
    training_labels = {row['label'] for row in rows if assignments[row['speaker']]=='train'}
    for split in ('valid','test'):
        labels = {row['label'] for row in rows if assignments[row['speaker']]==split}
        if labels != training_labels:
            raise ValueError(f'{split} labels differ from training; inspect coverage before choosing speakers')
    return assignments


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row)+'\n')


def prepare(root, output, cache_root, workers=4, inventory_only=False):
    import cv2
    cv2.setNumThreads(1)
    root, output, cache_root = Path(root).resolve(), Path(output).resolve(), Path(cache_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing dataset outputs are protected; select a new output directory')
    output.mkdir(parents=True, exist_ok=True); cache_root.mkdir(parents=True, exist_ok=True)
    all_rows, speakers = [], []
    started = time.perf_counter()
    download_manifest = root / 'download_manifest.json'
    assets = json.loads(download_manifest.read_text()).get('assets', {}) if download_manifest.exists() else {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for directory in sorted(root.glob('s[0-9]*'), key=lambda path:int(path.name[1:])):
            # Do not snapshot a directory while another process is extracting it.
            if assets and any(assets.get(directory.name+'/'+kind,{}).get('status') != 'verified_and_extracted' for kind in ('alignment','video_normal')):
                continue
            videos = sorted((directory/'corpus').rglob('*.mpg'))
            if not videos:
                continue
            speaker = directory.name
            aligns = defaultdict(list)
            for path in (directory/'corpus').rglob('*.align'):
                aligns[path.stem].append(path)
            cache_file = cache_root / (speaker+'.json')
            cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
            jobs = []
            for video in videos:
                candidates = aligns.get(video.stem, [])
                if len(candidates)>1:
                    raise ValueError('Duplicate annotation IDs: '+speaker+'/'+video.stem)
                alignment = candidates[0] if candidates else directory/'corpus'/'missing'/f'{video.stem}.align'
                key = video.relative_to(root).as_posix()
                jobs.append((video,alignment,speaker,root,cache.get(key)))
            rows = []
            for index, row in enumerate(pool.map(probe_video, jobs),1):
                rows.append(row)
                if index%250==0:
                    print(f'{speaker}: fully decoded/checked {index}/{len(videos)}',flush=True)
            atomic_json(cache_file,{row['video']:row for row in rows})
            all_rows.extend(rows)
            video_ids = {video.stem for video in videos}
            speakers.append({'speaker':speaker,'speaker_id':f'grid_T{int(speaker[1:]):02d}',
                'videos':len(videos),'valid':sum(row['status']=='valid' for row in rows),
                'invalid':sum(row['status']!='valid' for row in rows),'missing_annotations':sum(not aligns.get(video.stem) for video in videos),
                'orphan_annotations':sum(stem not in video_ids for stem in aligns),
                'label_counts':dict(sorted(Counter(row['label'] for row in rows if row['status']=='valid').items()))})
    eligible, duplicates = deduplicate(all_rows)
    for entry in speakers:
        entry['eligible_after_deduplication'] = sum(row['speaker']==entry['speaker'] for row in eligible)
    assignments = assign_speakers(eligible) if not inventory_only else {}
    annotations, manifest = [], []
    for row in eligible:
        manifest.append(asdict(GridManifestEntry(talk_id=f'T{int(row["speaker"][1:]):02d}',speaker_id=row['speaker_id'],utterance_id=row['utterance_id'],
            video_path=row['video'],audio_path='',alignment_path=row['alignment'],frame_count=row['decoded_frames'],fps=row['fps'],
            transcript=row['transcript'],value=row['transcript'])))
        if assignments:
            annotations.append({'id':row['speaker']+'/'+row['utterance_id'],'video':os.path.relpath(root/row['video'],output).replace('\\','/'),
                'label':row['label'],'speaker_id':row['speaker_id'],'split':assignments[row['speaker']],
                'data_source':'real_camera','input_kind':'raw_video','video_sha256':row['video_sha256'],
                'start_frame':row['start_frame'],'end_frame':row['end_frame'],
                'source_alignment':os.path.relpath(root/row['alignment'],output).replace('\\','/'),
                'alignment_sha256':row['alignment_sha256'],'alignment_timebase_hz':25000,
                'start_units':row['start_units'],'end_units':row['end_units'],
                'task':'GRID annotated letter slot; coarticulated, not a general isolated A-Z dataset'})
    audit = None
    if annotations:
        from pipeline.avhubert.classification import audit_records
        audit = audit_records(annotations)
        write_jsonl(output/'avhubert_letters.jsonl',annotations)
        for split in ('train','valid','test'):
            write_jsonl(output/(split+'.jsonl'),[row for row in annotations if row['split']==split])
        # One real, identically labeled sample from each split: smoke extraction, never training data replacement.
        smoke = []
        for split in ('train','valid','test'):
            candidates = [row for row in annotations if row['split']==split and row['label']=='A']
            smoke.append(sorted(candidates,key=lambda row:row['end_frame']-row['start_frame'],reverse=True)[0])
        write_jsonl(output/'smoke_annotations.jsonl',smoke)
    # Existing s1 manifests remain byte-for-byte intact. New speaker summaries
    # use the same per-speaker corpus-relative schema as the existing service.
    for entry in speakers:
        local_manifest = root / entry['speaker'] / 'grid_manifest.jsonl'
        if local_manifest.exists():
            continue
        local_rows = []
        for row in manifest:
            if row['speaker_id'] != entry['speaker_id']:
                continue
            prefix = entry['speaker'] + '/corpus/'
            local_rows.append({**row,'video_path':row['video_path'].removeprefix(prefix),
                'alignment_path':row['alignment_path'].removeprefix(prefix)})
        write_jsonl(local_manifest,local_rows)
    write_jsonl(output/'grid_manifest.jsonl',manifest)
    write_jsonl(output/'video_inventory.jsonl',all_rows)
    write_jsonl(output/'duplicates.jsonl',duplicates)
    report={'client_audit_date':'2026-10-29','source':'https://spandh.dcs.shef.ac.uk/gridcorpus/',
        'corpus_root':str(root),'audio_required':False,'validation':'all source frames decoded; byte SHA-256 and decoded-thumbnail fingerprints; strict six-token word annotations in 1/25000 second units',
        'speakers':speakers,'total_videos':len(all_rows),'valid_video_annotation_pairs':sum(row['status']=='valid' for row in all_rows),
        'invalid_video_annotation_pairs':sum(row['status']!='valid' for row in all_rows),
        'eligible_after_deduplication':len(eligible),'duplicate_groups':len(duplicates),
        'assignments':assignments,'class_labels':sorted({row['label'] for row in eligible}),
        'missing_AZ_letters':sorted(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')-{row['label'] for row in eligible}),
        'split_audit':audit,'elapsed_seconds':round(time.perf_counter()-started,2),
        'remaining_speakers':[f's{i}' for i in range(1,35) if i!=21 and f's{i}' not in {entry['speaker'] for entry in speakers}],
        'unavailable_video_speakers':['s21'],'recognition_accuracy_measured':False}
    atomic_json(output/'inventory.json',report)
    lines=['# GRID dataset inventory','', 'Client audit date: 2026-10-29. Visual-only official GRID video and word alignments. No training or recognition accuracy measurement.','',
        '| Speaker | Videos | Valid pairs | Invalid | Missing annotations | Orphan annotations | Eligible after deduplication |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for entry in speakers:
        lines.append('| '+ ' | '.join(str(entry[key]) for key in ['speaker','videos','valid','invalid','missing_annotations','orphan_annotations','eligible_after_deduplication'])+' |')
    lines+=['',f'Total videos: {len(all_rows):,}. Eligible pairs after duplicate screening: {len(eligible):,}.',
        '', 'Task labels: '+', '.join(report['class_labels'])+'. Missing A–Z letters: '+', '.join(report['missing_AZ_letters'])+'.',
        '', 'The 64×64 decoded fingerprint screens for matching visual content; matching groups are conservatively excluded. It is not a claim of native-resolution identity. Repeated transcripts across different speakers are legitimate label overlap, not duplicate video.',
        '', 'Exact speaker assignments:']
    for split in ('train','valid','test'):
        lines.append('- '+split+': '+', '.join(speaker for speaker,value in assignments.items() if value==split))
    lines+=['','Remaining speakers with no extracted videos: '+', '.join(report['remaining_speakers'])+'. s21 has no official video.']
    (output/'DATASET_INVENTORY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['total_videos','valid_video_annotation_pairs','invalid_video_annotation_pairs','eligible_after_deduplication','remaining_speakers','elapsed_seconds']},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'data/external/grid')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'.qa/grid-validation-cache')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--inventory-only',action='store_true')
    args=parser.parse_args()
    if args.workers<1: parser.error('Workers must be positive')
    prepare(args.root,args.output,args.cache,args.workers,args.inventory_only)


if __name__=='__main__':
    main()
