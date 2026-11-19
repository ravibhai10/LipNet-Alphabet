"""Build a new audit-only GRID split; do not edit original manifests."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = tuple(chr(65 + index) for index in range(26))
QUARANTINE = {'grid_T04', 'grid_T05'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_training_gates(protocol, parity):
    if protocol['vocabulary'] != list(LABELS):
        raise ValueError('A-Z order changed')
    if any(protocol['missing_labels_by_split'].values()):
        raise ValueError('Full A-Z training blocked: missing genuine class coverage, including W')
    if not protocol.get('crop_validation_passed') or not protocol.get('speaker_integrity_passed'):
        raise ValueError('Validated crops and speaker protocol required')
    if parity.get('training_inference_compatible') is not True:
        raise ValueError('Preprocessing compatibility gate has not passed')
    return True


def create(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing experiment files are protected')
    output.mkdir(parents=True)
    base = ROOT / 'data/processed/grid_multispeaker_v1'
    before = {name: digest(base / (name + '.jsonl')) for name in ['train', 'valid', 'test']}
    partitions, quarantine, ids, speakers, content = {}, [], set(), {}, defaultdict(set)
    for split in ['train', 'valid', 'test']:
        kept = []
        with (base / (split + '.jsonl')).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['id'] in ids or row['split'] != split or row['label'] not in LABELS:
                    raise ValueError('Duplicate source ID or invalid split/label')
                ids.add(row['id'])
                for key in ['video', 'source_alignment']:
                    path = (base / row[key]).resolve()
                    if not path.is_relative_to((ROOT / 'data/external/grid').resolve()):
                        raise ValueError('Source outside eligible local GRID corpus')
                    row[key] = path.relative_to(ROOT).as_posix()
                row['label_index'] = LABELS.index(row['label'])
                row['path_base'] = 'repository_root'
                row['crop_validation_status'] = 'not_extracted'
                if row['speaker_id'] in QUARANTINE:
                    quarantine.append({**row, 'exclusion_reason': 'historical_partition_conflict'})
                    continue
                speaker = row['speaker_id']
                if not speaker or speaker in speakers and speakers[speaker] != split:
                    raise ValueError('Unknown speaker or speaker leakage')
                speakers[speaker] = split
                content[row['video_sha256']].add(split)
                kept.append(row)
        partitions[split] = kept
    if any(len(splits) > 1 for splits in content.values()):
        raise ValueError('Exact source duplicate crosses splits')
    if len(ids) != 32891 or len(quarantine) != 1999:
        raise ValueError('Unexpected source/quarantine totals')
    for split, group in {**partitions, 'quarantine': quarantine}.items():
        with (output / (split + '.jsonl')).open('x') as stream:
            for row in group:
                stream.write(json.dumps(row) + '\n')
    protocol = {'task':'alphabet_experiment_preparation_only', 'vocabulary':list(LABELS),
        'label_mapping':{label:i for i,label in enumerate(LABELS)}, 'original_manifest_hashes':before,
        'counts':{split:len(group) for split,group in partitions.items()},
        'quarantined':len(quarantine), 'quarantined_speakers':sorted(QUARANTINE),
        'speaker_partitions':{split:sorted(s for s,value in speakers.items() if value == split) for split in partitions},
        'by_label':{split:dict(Counter(r['label'] for r in group)) for split,group in partitions.items()},
        'missing_labels_by_split':{split:[label for label in LABELS if not any(r['label']==label for r in group)] for split,group in partitions.items()},
        'speaker_integrity_passed':True, 'exact_source_hash_cross_split_duplicates':False,
        'near_duplicate_screening_passed':False, 'crop_validation_passed':False, 'full_training_allowed':False,
        'reason':'W is absent; crops/parity are not certified. These split files are proposed raw-source inputs, not training-ready datasets.',
        'original_test_policy':'Historical protected tests unchanged; exposed historical test data is not newly untouched.',
        'source_ids_unique':True, 'allenye66_used':False, 'MIRACL_phrases_used_as_words':False}
    after = {name:digest(base / (name + '.jsonl')) for name in before}
    if before != after:
        raise ValueError('Original manifests changed during preparation')
    protocol['generated_manifest_hashes'] = {p.name:digest(p) for p in output.glob('*.jsonl')}
    with (output / 'protocol.json').open('x') as stream:
        json.dump(protocol, stream, indent=2)
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.output), indent=2))
