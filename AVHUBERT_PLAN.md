# AV-HuBERT Compatibility & Storage Report

**Date:** 2026-10-25
**Environ:** Windows 11, Python 3.11.9 `@ .venv`, CPU-only PyTorch 2.2.2

## Dependency compatibility findings

| Dependency | Current `.venv` | Official AV-HuBERT | Conflict |
|---|---|---|---|
| Python | 3.11.9 | 3.8 (conda create) | `3.8` recommended; current `3.11` may still import but section-table torch/fairseq pairing is safer in 3.8 |
| torch | 2.2.2 (CPU) | fairseq wheels bundle their own torch | **Conflict** — fairseq install needs a torch it compiles against; pip fairseq `requirements.txt` does not require a specific torch, but CPU torch 2.2.2 is accepted |
| opencv | 4.10 (py3.11) | `==4.5.4.60` | Conflict — version pin mismatch |
| scipy | 1.26.4 | `==1.10.0` | Conflict — version pin mismatch |
| python-speech-features | NOT installed | `==0.6` | Missing |
| sentencepiece | NOT installed | `==0.1.96` | Missing |
| editdistance | NOT installed | `==0.6.0` | Missing |
| fairseq | NOT installed | required: `pip install --editable ./` inside `fairseq/` submodule | Missing |

**Conclusion:** The official AV-HuBERT stack cannot install into the existing `.venv` without changing pin versions and adding 4 missing packages + fairseq. Per safety rule #4, I create an **isolated environment `.venv-avhubert`** instead of destabilizing the current one.

## Installation plan for `.venv-avhubert` (verified status 2026-10-25)

```bat
python -m venv .venv-avhubert
.venv-avhubert\Scripts\activate
pip install --upgrade pip
pip install torch==2.2.2+cpu torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install fairseq hydra-core omegaconf cffi cython regex sacrebleu tqdm bitarray torchaudio
pip install scipy>=1.10 opencv-python>=4.5 python-speech-features==0.6 sentencepiece editdistance
```

### Blockers verified on this Windows/Python 3.11 environment

1. **`fairseq` source-install is broken.** The PyPI `fairseq==0.12.2` sdist fails with
   `FileNotFoundError: 'fairseq\version.txt'` (a packaging bug in the sdist). This blocks
   `pip install fairseq` on Python 3.11 without a `version.txt` patch or a wheel (there is
   no fairseq wheel on PyPI for this environment). `fairseq` is REQUIRED for the official
   loading path `fairseq.checkpoint_utils.load_model_ensemble_and_task`.
2. **`sentencepiece` and `editdistance` need C++14 build tools.** Their PyPI source builds
   fail with `Microsoft Visual C++ 14.0 or greater is required` on this Windows machine
   (no C++ Build Tools installed). Newer wheels (0.2.2 / 0.8.1) ARE available for Python 3.11
   and were installed successfully as a workaround.
3. **No real checkpoint present.** `dl.fbaipublicfiles.com` returns `403 Forbidden` from this
   sandbox. The checkpoint (`avhubert_base.pt` ≈ 1.2 GB) must be downloaded manually by the
   user and placed in `models/external/avhubert/`.

### Impact on the AV-HuBERT adapter

The official loading path (`loader.py`) is implemented and imports. Tests run clean as
**interface-level mocked tests** (5/5 pass in `.venv-avhubert`). A **real checkpoint smoke
test is blocked** pending: (a) fairseq loadable on Python 3.11, (b) the user downloads
`avhubert_base.pt`, (c) `AVHubertLoader.load()` → `encode()` returns finite `[T, 768]`
without error. Per policy, this blocker is reported, not worked around.

## Evaluation workstream

`pipeline/eval/` is a placeholder for the speaker-independent split/validation/validation
integrity tooling and WER metric. To be implemented after the isthmus between the corpus
ingestion (Workstream A) and AV-HuBERT adapter (Workstream B) land.


## Storage check (before any download)

- **Total free on C:** ~401 GB
- **AV-HuBERT checkpoint** (Base `avhubert_base.pt` ≈ 1.2 GB / Large ≈ 2.9 GB): fits easily
- **GRID corpus** per talker (50 kHz raw ≈ 300 MB / endpointed ≈ 100 MB / high video ≈ 1.2 GB): full 34-talker corpus ≈ **>100 GB**; download must be per-talk and is sized per license request — see `GRID_ACCESS.md`.

## Checkpoint sources & license

- Official checkpoint page: `http://facebookresearch.github.io/av_hubert` (AV-HuBERT License Agreement, Meta Platforms, non-commercial research only).
- Official repository: `https://github.com/facebookresearch/av_hubert` (archived 2024-09-01, read-only; MIT-like LICENSE).
- Loading procedure (README): `fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])`, then `model = models[0]`.
- **No AV-HuBERT checkpoints currently present in this sandbox** — the `dl.fbaipublicfiles.com` host returns `403 Forbidden` from this environment (direct download blocked). The checkpoint file **must be downloaded manually by a user with internet + permitted access**, then placed at `models/external/avhubert/avhubert_base.pt` (base, LRS3, no finetuning) as the smallest appropriate official pretrained checkpoint for a real inference smoke test.

## Expected smoke-test input

AV-HuBERT features frames at **25 Hz** (label_rate 25). A 2.5 s clip → 62–63 feature frames per video clip. Input video must be a valid file (GRID/webcam) that can be opened with `cv2.VideoCapture`. The `audio-visual` variant consumes audio + video; the `video`-only variant (lip-reading) consumes video frames → AV-HuBERT feature extractor over raw frames. Output is a hidden-unit sequence `[T, D]` (base D=768, large D=1024) or a 20-dim bottleneck. This project's own classifier uses a 192-dim fused representation, so an adapter is required to map AV-HuBERT `[T, D]` → `[T, 192]` before the Phase 4 Conformer.
