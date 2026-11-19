"""Relocate existing repository artifacts without rewriting historical manifests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = {'video', 'file_path', 'roi', 'feature', 'tensor', 'source_video', 'landmarks_path'}


def relocate_existing(value, root=ROOT):
    if not isinstance(value, str) or Path(value).is_file():
        return value
    text = value.replace('\\', '/')
    root = Path(root).resolve()
    candidates = []
    for directory in ('data', 'runs', 'models', 'alphabet_dataset'):
        marker = '/' + directory + '/'
        if marker in text:
            candidates.append(root / directory / text.split(marker, 1)[1])
    if not text.startswith('/') and ':/' not in text:
        candidates.append(root / text)
    for path in candidates:
        candidate = path.resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return str(candidate)
    return value  # Missing bytes remain missing; never invent or rewrite evidence.


def relocate_row(row, root=ROOT):
    if not isinstance(row, dict):
        return row
    result = dict(row)
    for key in PATH_FIELDS:
        if key in result:
            result[key] = relocate_existing(result[key], root)
    if isinstance(result.get('frames'), list):
        result['frames'] = [relocate_existing(p, root) for p in result['frames']]
    return result
