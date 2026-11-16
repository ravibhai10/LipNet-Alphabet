"""Preflight and dataset operations. Credential values are never printed or persisted."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.word_recognition.common import atomic_json, sha256


def preflight(output):
    import torch
    class Memory(ctypes.Structure):
        _fields_ = [("length",ctypes.c_ulong),("load",ctypes.c_ulong)] + [(name,ctypes.c_ulonglong) for name in
            ("total_physical","available_physical","total_page","available_page","total_virtual","available_virtual","extended")]
    memory = None
    if os.name == "nt":
        state = Memory(); state.length = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            memory = {"physical_gib": state.total_physical/2**30, "available_gib": state.available_physical/2**30}
    catalog = json.loads((ROOT / "pipeline/multidataset/catalog.json").read_text())
    credentials = {"kaggle": bool(os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or (Path.home()/".kaggle/kaggle.json").is_file() or (Path.home()/".kaggle/access_token").is_file()),
                   "roboflow": bool(os.environ.get("ROBOFLOW_API_KEY")),
                   "lrw": bool(os.environ.get("LIPNET_LRW_USERNAME") and os.environ.get("LIPNET_LRW_PASSWORD"))}
    for item in catalog.values():
        path = ROOT/item["local"]
        item["local_exists"] = path.is_dir() and any(path.iterdir())
    result = {"checked_at":datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(),
              "python":sys.executable,"python_version":platform.python_version(),"torch":torch.__version__,
              "cuda_available":torch.cuda.is_available(),"cuda_devices":torch.cuda.device_count(),
              "logical_cpus":os.cpu_count(),"memory":memory,"free_bytes":shutil.disk_usage(ROOT).free,
              "workers":2,"reserve_bytes":20*2**30,"credential_presence_only":credentials,"datasets":catalog,
              "bounded_work": {"reference_sources":"MuAViC source only, no full dataset", "miracl":"Probe official links; maximum 6 GiB archive transfer only after size/access checks", "grid":"Reuse verified original data; bounded pretrained-feature word pilot; no alphabet/letter labels", "sequence":"Inspect actual decoder implementation; unavailable until a video-only fine-tuned checkpoint and evaluation exist"}}
    output=Path(output)
    if output.exists(): raise FileExistsError("Preflight output is protected; choose a new file.")
    atomic_json(output,result)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("preflight");q.add_argument("--output",type=Path,required=True)
    q=sub.add_parser("acquire-miracl");q.add_argument("--root",type=Path,default=ROOT/'data/external/miracl');q.add_argument("--minutes",type=float,default=10)
    q=sub.add_parser("ingest");q.add_argument("--dataset",choices=['grid','miracl','tcd_timit','muavic','lrw'],required=True);q.add_argument("--root",type=Path,required=True);q.add_argument("--index",type=Path);q.add_argument("--output",type=Path,required=True)
    q.add_argument('--pilot-repetitions',type=int)
    q=sub.add_parser('ingest-muavic');q.add_argument('--root',type=Path,required=True);q.add_argument('--tsv',type=Path,nargs='+',required=True);q.add_argument('--transcripts',type=Path,nargs='+',required=True);q.add_argument('--speakers',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('ingest-roboflow');q.add_argument('--root',type=Path,required=True);q.add_argument('--metadata',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('score-sequences');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--predictions',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('validate');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--verify-sources',action='store_true')
    q=sub.add_parser('avhubert-manifest');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=="preflight": preflight(args.output)
    elif args.command=="acquire-miracl":
        if args.minutes<=0:p.error("Positive transfer time budget required.")
        from pipeline.multidataset.acquisition import acquire_miracl
        print(json.dumps(acquire_miracl(args.root,args.minutes),indent=2))
    elif args.command=='ingest-muavic':
        from pipeline.multidataset.adapters import muavic_native,export
        if len(args.tsv)!=len(args.transcripts):p.error('Each native TSV needs its paired transcript file.')
        rows=[]
        for tsv,text in zip(args.tsv,args.transcripts):rows.extend(muavic_native(args.root,tsv,text,args.speakers,allow_mapping_subset=True))
        mapped={json.loads(x)['id'] for x in args.speakers.read_text().splitlines() if x.strip()}
        if {r['id'].removeprefix('muavic:') for r in rows}!=mapped:raise ValueError('Every reviewed speaker-map ID must occur in the native inputs.')
        print(json.dumps(export(rows,args.output),indent=2))
    elif args.command=='ingest-roboflow':
        from pipeline.multidataset.adapters import roboflow_images,export
        print(json.dumps(export(roboflow_images(args.root,args.metadata),args.output,speaker_protocol='unknown'),indent=2))
    elif args.command=='score-sequences':
        from pipeline.multidataset.sequence import score
        print(json.dumps(score(args.manifest,args.predictions,args.output)['metrics'],indent=2))
    elif args.command in ('validate','avhubert-manifest'):
        from pipeline.multidataset import adapters as a
        rows=[json.loads(x) for x in args.manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
        if args.command=='avhubert-manifest':report=a.avhubert_manifest(rows,args.output)
        else:
            inventory=args.manifest.with_suffix('.inventory.json')
            previous=json.loads(inventory.read_text()) if inventory.exists() else {}
            if previous.get('manifest_sha256') not in (None,sha256(args.manifest)):raise ValueError('Manifest fingerprint changed.')
            report=a.validate(rows,speaker_protocol=previous.get('speaker_protocol','disjoint'))
            if args.verify_sources:report.update(a.verify_sources(rows))
        print(json.dumps(report,indent=2))
    elif args.command=="ingest":
        from pipeline.multidataset import adapters as a
        omitted=[]
        if args.dataset=='grid':rows=a.grid_words(args.root)
        elif args.dataset=='miracl':rows,omitted=a.miracl(args.root,pilot_repetitions=args.pilot_repetitions)
        elif args.dataset=='lrw':rows=a.lrw(args.root)
        else:
            if args.index is None:p.error('An authorized reviewed native index is required for TCD/MuAViC.')
            rows=a.reviewed_index(args.dataset,args.root,args.index)
        report=a.export(rows,args.output,speaker_protocol='publisher_dates_unknown_speakers' if args.dataset=='lrw' else 'disjoint')
        if omitted:atomic_json(args.output.with_suffix('.unresolved_labels.json'),omitted)
        print(json.dumps(report,indent=2))


if __name__=="__main__": main()
