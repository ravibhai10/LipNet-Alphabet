"""Inventory current local candidates and verify protected historical artifact hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    result={'scope':'Read-only local artifact audit; historical accuracy/timing values are not new measurements',
            'protected':{'checked':0,'changed':[],'missing':[]},'candidates':[]}
    previous=json.loads((ROOT/'.qa/reliability_20261109/artifact_hashes_before.json').read_text())
    for item in previous:
        path=ROOT/item['path'];result['protected']['checked']+=1
        if not path.is_file():result['protected']['missing'].append(item['path'])
        elif sha(path)!=item['sha256']:result['protected']['changed'].append(item['path'])
    for identifier,checkpoint,source,license_path,scope in [
        ('avhubert-miracl8','runs/word_accuracy/avhubert_miracl8_v1/head.pt','external/av_hubert','external/av_hubert/LICENSE','Eight supervised words; encoder itself has no word vocabulary'),
        ('alphabet-cnn-bilstm','models/alphabet_classifier.pt','models/alphabet_model.py',None,'26 outputs; full A–Z reliability unproven'),
        ('lrw500','models/external/lrw_tcn/resnet18_dctcn_video.pth','external/lrw_tcn_reference','external/lrw_tcn_reference/LICENSE','Publisher LRW contract; separate vocabulary and preprocessing'),
        ('temporal15','runs/reliability_20261109/temporal_v2/bilstm_t15/best.pt','pipeline/word_recognition/temporal_experiment.py',None,'Historical rejected temporal candidate'),
        ('temporal29','runs/reliability_20261109/temporal_v2/bilstm_t29/best.pt','pipeline/word_recognition/temporal_experiment.py',None,'Historical rejected temporal candidate')]:
        path=ROOT/checkpoint;license_file=ROOT/license_path if license_path else None
        result['candidates'].append({'id':identifier,'checkpoint':checkpoint,'present':path.is_file(),
             'bytes':path.stat().st_size if path.is_file() else None,'sha256':sha(path) if path.is_file() else None,
             'source_present':(ROOT/source).exists(),'license_path':license_path,
             'license_sha256':sha(license_file) if license_file and license_file.is_file() else None,'scope':scope})
    result['encoder']={'path':'models/external/avhubert/base_vox_iter5.pt',
                       'bytes':(ROOT/'models/external/avhubert/base_vox_iter5.pt').stat().st_size,
                       'sha256':sha(ROOT/'models/external/avhubert/base_vox_iter5.pt')}
    (args.out/'artifact_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.out/'working_tree.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),encoding='utf-8')
    (args.out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    print(json.dumps(result['protected']))


if __name__=='__main__':main()
