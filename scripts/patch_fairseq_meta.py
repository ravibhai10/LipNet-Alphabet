"""Apply the minimal meta-initialization compatibility guard to vendored Fairseq."""
from pathlib import Path

root=Path(__file__).resolve().parents[1]
path=root/'external/av_hubert/fairseq/fairseq/modules/transformer_sentence_encoder.py'
source=path.read_text(encoding='utf-8')
marker='    def normal_(data):\n'
guard='        if data.is_meta:\n            return  # Parameters are strictly assigned from a checkpoint afterward.\n'
if marker not in source:raise RuntimeError('Unexpected Fairseq revision; inspect before patching.')
if guard not in source:
    source=source.replace(marker,marker+guard,1)
    path.write_text(source,encoding='utf-8')
print('Fairseq meta initialization guard verified; CPU initialization unchanged.')
