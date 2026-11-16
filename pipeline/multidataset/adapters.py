"""Dataset-specific adapters. Missing corpus knowledge is an explicit blocker.

GRID timestamps are known. MIRACL frames are ordered without inventing an FPS.
TCD uses an authorized, reviewed index with declared view/timebase. MuAViC uses
its actual TSV/transcript pair plus explicit speaker identities. LRW preserves
its publisher protocol instead of fabricating disjoint speaker IDs.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
import cv2
import numpy as np
from pipeline.word_recognition.common import sha256,atomic_json,write_rows

WORD_TASKS={'isolated_word','aligned_word'}
TASKS=WORD_TASKS|{'phrase','sentence','phonetic_image_auxiliary'}
MIRACL_WORDS=['Hegin','Choose','Connection','Navigation','Next','Previous','Start','Stop','Hello','Well']
MIRACL_PHRASES=['Stop navigation.','Excuse me.','I am sorry.','Thank you.','Good bye.','I love this game.','Nice to meet you.','You are welcome.','How are you?','Have a good time']
MIRACL_UNRESOLVED={1,10}


def normalize(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Nonempty text required.')
    return ' '.join(unicodedata.normalize('NFC',text).strip().split()).casefold()


def video_info(path,max_frames=10000):
    capture=cv2.VideoCapture(str(path));fps=float(capture.get(cv2.CAP_PROP_FPS));reported=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));n=0;shape=None
    try:
        if not capture.isOpened() or not math.isfinite(fps) or fps<=0:raise ValueError('Cannot decode video with a valid native FPS.')
        while True:
            ok,frame=capture.read()
            if not ok:break
            n+=1;shape=frame.shape
            if n>max_frames:raise ValueError('Video exceeds adapter safety bound; segment with verified timestamps.')
    finally:capture.release()
    if n<2 or n<reported:raise ValueError('Empty/truncated video.')
    return {'frames':n,'fps':fps,'shape':list(shape)}


def validate(records,*,speaker_protocol='disjoint'):
    if not records:raise ValueError('No authentic normalized samples.')
    ids=set();speakers={};sources={};segments={};classes=defaultdict(lambda:defaultdict(set));counts=Counter()
    for row in records:
        if any(k in row for k in ('audio','audio_path','audio_features','waveform')):
            raise ValueError('Audio inputs are forbidden in visual-only manifests.')
        if row.get('task') not in TASKS or row.get('split') not in {'train','valid','test'} or row.get('validation_status')!='valid':raise ValueError('Invalid task, split, or validation status.')
        if row['id'] in ids:raise ValueError('Duplicate sample ID.')
        ids.add(row['id'])
        if not re.fullmatch('[a-f0-9]{64}',row.get('source_sha256','')):raise ValueError('Missing source fingerprint.')
        source=row['source_sha256'];split=row['split'];speaker=row.get('speaker_id')
        if speaker_protocol=='disjoint' and (not isinstance(speaker,str) or not speaker):raise ValueError('Explicit speaker identities required for this protocol.')
        if speaker:
            key=(row['dataset'],speaker)
            if key in speakers and speakers[key]!=split:raise ValueError('Speaker overlap across partitions.')
            speakers[key]=split
        if source in sources and sources[source]!=split:raise ValueError('Duplicate source content crosses partitions.')
        sources[source]=split
        first,last=row.get('start_frame'),row.get('end_frame')
        if row['task']!='phonetic_image_auxiliary':
            if type(first) is not int or type(last) is not int or first<0 or last-first<2 or last>row['source_frames']:raise ValueError('Invalid frame interval.')
        key=(source,first,last,row['task'])
        if key in segments:raise ValueError('Repeated source interval.')
        segments[key]=row['id']
        transcript=normalize(row['transcript'])
        if row['task'] in WORD_TASKS:
            if not row.get('alignment_verified') or row.get('word_label')!=transcript or len(transcript.split())!=1:raise ValueError('Word labels require verified single-word supervision.')
            classes[row['dataset'],row['task']][split].add(transcript)
        elif row.get('word_label') is not None:raise ValueError('Phrase, sentence, and phonetic-image labels cannot become word classes.')
        counts[row['dataset'],row['task'],split]+=1
    for group,partitions in classes.items():
        if set(partitions)!={'train','valid','test'} or (partitions['valid']|partitions['test'])-partitions['train']:raise ValueError('Word partitions/classes incomplete.')
    for dataset,task in {(r['dataset'],r['task']) for r in records}:
        if {fold for (ds,kind,fold),count in counts.items() if ds==dataset and kind==task}!={'train','valid','test'}:
            raise ValueError('Every normalized task needs all three partitions before evaluation.')
    return {'samples':len(records),'speaker_protocol':speaker_protocol,'speaker_independent':True if speaker_protocol=='disjoint' else None,
            'counts':{'/'.join(k):v for k,v in counts.items()},
            'speaker_assignments':{d:{s:sorted(sp for (ds,sp),fold in speakers.items() if ds==d and fold==s) for s in ('train','valid','test')} for d in {r['dataset'] for r in records}},
            'vocabularies':{'/'.join(k):sorted(set.union(*v.values())) for k,v in classes.items()},
            'identity_scope':'Speaker IDs are namespaced by corpus; unknown cross-corpus human identity is not claimed disjoint.'}


def export(records,output,*,speaker_protocol='disjoint'):
    output=Path(output)
    if output.exists():raise FileExistsError('Normalized manifest already exists; choose a new output.')
    report=validate(records,speaker_protocol=speaker_protocol)
    output.parent.mkdir(parents=True,exist_ok=True);write_rows(output,records)
    report['manifest_sha256']=sha256(output);atomic_json(output.with_suffix('.inventory.json'),report)
    return report


def verify_sources(records):
    """Rehash actual bytes, including ordered color-frame filenames."""
    checked={}
    for row in records:
        if row['input_kind']=='color_frames':
            digest=hashlib.sha256()
            for name in row['frames']:
                path=Path(name);digest.update(path.name.encode());digest.update(bytes.fromhex(sha256(path)))
            actual=digest.hexdigest()
        else:
            name=row.get('video',row.get('image'))
            if name not in checked:checked[name]=sha256(name)
            actual=checked[name]
        if actual!=row['source_sha256']:raise ValueError('Source bytes changed after ingestion.')
    return {'samples_rehashed':len(records)}


def avhubert_manifest(records,output):
    """Reuse the verified frozen-encoder head trainer with true word intervals.

    MIRACL's unknown native FPS is not silently replaced with a guessed rate.
    Raw video adapters must supply measured FPS and exact annotation bounds.
    """
    validate(records);verify_sources(records);rows=[]
    for row in records:
        if row['task'] not in WORD_TASKS or row['input_kind']!='raw_video' or not row.get('fps'):
            raise ValueError('AV-HuBERT word export requires aligned raw-video words with measured FPS.')
        rows.append({'id':row['id'],'video':row['video'],'label':row['word_label'],'speaker_id':row['dataset']+':'+row['speaker_id'],
                     'split':row['split'],'data_source':'real_camera','input_kind':'raw_video',
                     'start_frame':row['start_frame'],'end_frame':row['end_frame']})
    output=Path(output)
    if output.exists():raise FileExistsError('Annotation export is protected.')
    output.parent.mkdir(parents=True,exist_ok=True);write_rows(output,rows)
    return {'annotations':len(rows),'sha256':sha256(output),'vocabulary':sorted({r['label'] for r in rows})}


def grid_words(baseline_manifest):
    rows=[]
    for line in Path(baseline_manifest).read_text().splitlines():
        old=json.loads(line)
        if sha256(old['source_video'])!=old['source_sha256'] or sha256(old['annotation'])!=old['annotation_sha256']:raise ValueError('GRID source changed.')
        info=video_info(old['source_video'],500)
        rows.append({'id':'grid:'+old['id'],'dataset':'grid','source':'official_grid_word_alignments','input_kind':'raw_video',
                     'video':old['source_video'],'speaker_id':old['speaker'],'task':'aligned_word','raw_transcript':old['word'],
                     'transcript':normalize(old['word']),'word_label':normalize(old['word']),'alignment_verified':True,
                     'split':old['split'],'start_frame':old['start_frame'],'end_frame':old['end_frame'],'source_frames':info['frames'],
                     'fps':info['fps'],'source_sha256':old['source_sha256'],'annotation_sha256':old['annotation_sha256'],
                     'preprocessing':'raw full-face; model-specific measured mouth alignment required','validation_status':'valid'})
    return rows


def muavic_native(root, tsv, transcripts, speakers, *, allow_mapping_subset=False):
    """Read the publisher's five-column Fairseq TSV and line-paired text.

    Reviewed JSONL speaker map fields: id, speaker_id, split, view,
    provenance_reviewed=true. The TSV audio column is deliberately discarded.
    No sentence-to-word boundaries are inferred from this format.
    """
    root=Path(root).resolve()
    lines=Path(tsv).read_text(encoding='utf-8').splitlines()
    texts=Path(transcripts).read_text(encoding='utf-8').splitlines()
    mapping={}
    for line in Path(speakers).read_text(encoding='utf-8').splitlines():
        item=json.loads(line)
        if item['id'] in mapping:raise ValueError('Duplicate reviewed speaker-map ID.')
        mapping[item['id']]=item
    if not lines or len(lines)-1!=len(texts):raise ValueError('Native TSV/text line counts disagree.')
    # Upstream utils.py writes a root header then [id,video,audio,nframes,nsamples].
    rows=[];seen=set()
    for line,text in zip(lines[1:],texts):
        values=line.split('\t')
        if len(values)!=5:raise ValueError('Expected native MuAViC five-column TSV.')
        sample,video,_ignored_audio,count,_ignored_audio_samples=values
        if sample in seen:raise ValueError('Duplicate native MuAViC ID.')
        seen.add(sample);item=mapping.get(sample,{})
        if item.get('provenance_reviewed') is not True or not item.get('speaker_id') or not item.get('view') or item.get('split') not in {'train','valid','test'}:
            raise ValueError('Every native ID requires a reviewed speaker, view, and split.')
        candidate=Path(video)
        if candidate.is_absolute():path=candidate.resolve()
        else:
            header=Path(lines[0]);base=header if header.is_absolute() else root/header
            path=(base/candidate).resolve()
        if not path.is_relative_to(root):raise ValueError('Native video path escapes authorized corpus root.')
        info=video_info(path)
        if int(count)!=info['frames']:raise ValueError('Native video frame count disagrees with decoding.')
        rows.append({'id':'muavic:'+sample,'dataset':'muavic','source':'official MuAViC native TSV/text with reviewed identities',
                     'input_kind':'raw_video','video':str(path),'speaker_id':item['speaker_id'],'split':item['split'],
                     'view':item['view'],'raw_transcript':text,'transcript':normalize(text),'task':'sentence','word_label':None,
                     'alignment_verified':False,'start_frame':0,'end_frame':info['frames'],'source_frames':info['frames'],
                     'fps':info['fps'],'source_sha256':sha256(path),'validation_status':'valid'})
    if not allow_mapping_subset and set(mapping)!=seen:raise ValueError('Speaker map contains unpaired IDs.')
    return rows


def roboflow_images(root, metadata):
    """Local authenticated classification export: split/phonetic_class/image.

    Object-detection exports are not silently interpreted as classification.
    This auxiliary dataset has unknown speaker identities and cannot be included
    in a speaker-disjoint word benchmark.
    """
    root=Path(root).resolve();meta=json.loads(Path(metadata).read_text(encoding='utf-8'))
    catalog=json.loads((Path(__file__).parent/'catalog.json').read_text())['roboflow']
    if meta.get('version')!=1 or meta.get('license')!='CC BY 4.0' or meta.get('source')!=catalog['source'] or meta.get('authenticated_export') is not True:
        raise ValueError('Reviewed authenticated export version, license, and source required.')
    rows=[]
    for native,split in [('train','train'),('valid','valid'),('test','test')]:
        for path in sorted((root/native).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
            relative=path.relative_to(root/native)
            if len(relative.parts)!=2 or relative.parts[0] not in catalog['classes']:raise ValueError('Expected reviewed phonetic classification folders.')
            image=cv2.imread(str(path))
            if image is None:raise ValueError('Cannot decode auxiliary image.')
            label=relative.parts[0]
            rows.append({'id':'roboflow:'+str(path.relative_to(root)).replace('\\','/'),'dataset':'roboflow',
                         'source':meta['source'],'version':1,'license':meta['license'],'input_kind':'static_image','image':str(path),
                         'speaker_id':None,'split':split,'task':'phonetic_image_auxiliary','raw_transcript':label,'transcript':label,
                         'phonetic_label':label,'word_label':None,'alignment_verified':False,'source_sha256':sha256(path),'validation_status':'valid'})
    return rows


def miracl_splits(speakers):
    groups={prefix:sorted([s for s in speakers if s.startswith(prefix)],key=lambda s:int(s[1:])) for prefix in ('F','M')}
    if len(groups['F'])!=10 or len(groups['M'])!=5 or len(speakers)!=15:raise ValueError('Expected all 15 original MIRACL speaker folders; partial data is not a full benchmark.')
    if len({(s[0],int(s[1:])) for s in speakers})!=15:raise ValueError('Ambiguous speaker aliases; review actual folder identities.')
    assignments={}
    for split,fs,ms in (('train',groups['F'][:6],groups['M'][:3]),('valid',groups['F'][6:8],groups['M'][3:4]),('test',groups['F'][8:],groups['M'][4:])):
        assignments.update({speaker:split for speaker in fs+ms})
    return assignments


def miracl(root,*,include_unresolved=False,pilot_repetitions=None):
    if pilot_repetitions is not None and not 1<=pilot_repetitions<=10:raise ValueError('Pilot repetitions must be 1..10.')
    root=Path(root);groups=defaultdict(list)
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:continue
        parts=path.relative_to(root).parts
        for i,part in enumerate(parts):
            if re.fullmatch(r'[FM]\d{2,3}',part) and len(parts)>i+4 and parts[i+1].casefold() in {'words','phrases'}:
                groups[part,parts[i+1].casefold(),parts[i+2],parts[i+3]].append(path);break
    if not groups:raise FileNotFoundError('No MIRACL speaker/task/class/repetition image sequences found.')
    assignments=miracl_splits({key[0] for key in groups});rows=[];omitted=[]
    for (speaker,kind,class_id,repetition),paths in sorted(groups.items()):
        if not re.fullmatch(r'\d{2}',class_id) or not 1<=int(class_id)<=10 or not re.fullmatch(r'\d{2}',repetition) or not 1<=int(repetition)<=10:raise ValueError('Unknown MIRACL class/repetition folder.')
        if pilot_repetitions is not None and (kind!='words' or int(repetition)>pilot_repetitions or int(class_id) in MIRACL_UNRESOLVED):continue
        # Actual archive names are inspected; color prefixes are mandatory, depth never inferred RGB.
        color=[p for p in paths if re.search(r'(?i)(color|colour|rgb)',p.name)]
        if not color:raise ValueError(f'No explicitly identified color frames: {speaker}/{kind}/{class_id}/{repetition}')
        def ordinal(p):
            numbers=re.findall(r'\d+',p.stem)
            if not numbers:raise ValueError('Color frame order has no numeric identifier.')
            return int(numbers[-1])
        color.sort(key=ordinal);indices=[ordinal(p) for p in color]
        if len(set(indices))!=len(indices) or any(b-a!=1 for a,b in zip(indices,indices[1:])):raise ValueError('Duplicate/missing color frame indices.')
        checksum=hashlib.sha256()
        for path in color:
            image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype!=np.uint8 or image.shape!=(480,640,3):raise ValueError('MIRACL RGB frame geometry/type mismatch.')
            checksum.update(path.name.encode());checksum.update(bytes.fromhex(sha256(path)))
        if len(color)<2 or len(color)>500:raise ValueError('MIRACL word/phrase sequence outside supported frame bounds.')
        index=int(class_id);raw=(MIRACL_WORDS if kind=='words' else MIRACL_PHRASES)[index-1]
        unresolved=kind=='words' and index in MIRACL_UNRESOLVED
        if unresolved and not include_unresolved:
            omitted.append({'id':f'miracl:{speaker}:{kind}:{class_id}:{repetition}','speaker':speaker,'class_id':class_id,'repetition':int(repetition),
                            'source_folder':str(color[0].parent.resolve()),'source_sha256':checksum.hexdigest(),
                            'raw_author_label':raw,'reason':'Author label spelling unresolved'});continue
        task='isolated_word' if kind=='words' else 'phrase'
        rows.append({'id':f'miracl:{speaker}:{kind}:{class_id}:{repetition}','dataset':'miracl','source':'author MIRACL-VC1 RGB frames',
                     'input_kind':'color_frames','frames':[str(p.resolve()) for p in color],'speaker_id':speaker,'task':task,
                     'raw_transcript':raw,'transcript':normalize(raw),'word_label':normalize(raw) if task=='isolated_word' else None,
                     'alignment_verified':not unresolved,'label_status':'unresolved' if unresolved else 'publisher_class_id',
                     'class_id':class_id,'repetition':int(repetition),'split':assignments[speaker],'fps':None,
                     'timestamp_policy':'native FPS not supplied; ordered frame indices only','start_frame':0,'end_frame':len(color),
                     'source_frames':len(color),'source_sha256':checksum.hexdigest(),'validation_status':'valid' if not unresolved else 'unresolved'})
    expected=15*10*10*2
    if len(groups)!=expected:raise ValueError(f'MIRACL instance quota mismatch: {len(groups)} vs {expected}.')
    return rows,omitted


def reviewed_index(dataset,root,index):
    """TCD/MuAViC reviewed sentence index. Explicit word timing is required for crops.

    No native layout/timebase is guessed while the authorized corpus is absent.
    Fields: video, transcript, speaker_id, split, view, plus optional alignment
    and alignment_timebase_hz. Word annotation rows: integer start end token.
    """
    if dataset not in {'tcd_timit','muavic'}:raise ValueError('Use the dataset-specific adapter.')
    root=Path(root).resolve();rows=[]
    for line in Path(index).read_text().splitlines():
        item=json.loads(line);path=(root/item['video']).resolve()
        if not path.is_relative_to(root):raise ValueError('Source path escapes authorized corpus.')
        if not item.get('speaker_id') or not item.get('view') or item.get('provenance_reviewed') is not True:raise ValueError('Reviewed speaker, camera view, and provenance are required.')
        info=video_info(path);text=normalize(item['transcript']);digest=sha256(path)
        base={'id':f'{dataset}:{item["id"]}','dataset':dataset,'source':'authorized reviewed native corpus index',
              'input_kind':'raw_video','video':str(path),'speaker_id':item['speaker_id'],'split':item['split'],'view':item['view'],
              'raw_transcript':item['transcript'],'transcript':text,'word_label':None,'task':'sentence','alignment_verified':False,
              'start_frame':0,'end_frame':info['frames'],'source_frames':info['frames'],'fps':info['fps'],
              'source_sha256':digest,'validation_status':'valid'}
        if not item.get('alignment'):
            rows.append(base);continue
        hz=item.get('alignment_timebase_hz')
        if type(hz) is not int or hz<=0 or item.get('alignment_method') not in {'native_reviewed','forced_alignment_reviewed'}:raise ValueError('Alignment requires reviewed timebase and method; sentence text is insufficient.')
        annotation=(root/item['alignment']).resolve()
        if not annotation.is_relative_to(root):raise ValueError('Annotation path escapes corpus.')
        tokens=[];previous=0
        for number,raw in enumerate(annotation.read_text().splitlines()):
            values=raw.split()
            if len(values)!=3:raise ValueError('Expected start/end/word annotation rows.')
            start,end=int(values[0]),int(values[1]);word=normalize(values[2])
            if start<previous or end<=start:raise ValueError('Nonmonotonic word alignment.')
            previous=end
            if word in {'sil','sp','<sil>'}:continue
            first,last=math.floor(start/hz*info['fps']),math.ceil(end/hz*info['fps'])
            if not 0<=first<last<=info['frames'] or last-first<2:raise ValueError('Implausible/empty word interval.')
            tokens.append((word,first,last))
        if ' '.join(t[0] for t in tokens)!=text:raise ValueError('Word alignment and transcript disagree.')
        for i,(word,first,last) in enumerate(tokens):
            rows.append({**base,'id':base['id']+f':word:{i}','task':'aligned_word','transcript':word,'word_label':word,
                         'alignment_verified':True,'start_frame':first,'end_frame':last,'annotation_sha256':sha256(annotation),
                         'alignment_method':item['alignment_method'],'alignment_timebase_hz':hz})
    return rows


def lrw(dataset_directory):
    from pipeline.lrw.whole_word import load_dataset
    info,old=load_dataset(dataset_directory);rows=[]
    for row in old:
        rows.append({'id':'lrw:'+row['id'],'dataset':'lrw','source':'audited authorized Oxford-BBC LRW',
                     'input_kind':'raw_video','video':row['video'],'speaker_id':None,'task':'isolated_word',
                     'raw_transcript':row['word'],'transcript':normalize(row['word']),'word_label':normalize(row['word']),
                     'alignment_verified':True,'split':row['split'],'fps':row['fps'],'start_frame':0,'end_frame':29,
                     'source_frames':29,'source_sha256':row['source_sha256'],'validation_status':'valid',
                     'split_protocol':info['protocol']})
    return rows
