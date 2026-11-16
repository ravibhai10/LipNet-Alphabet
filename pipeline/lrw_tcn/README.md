# Publisher-trained LRW 500-word research inference

The model now also has a separate selectable private research API at
`/api/predict/word/lrw500`, with loading/readiness at `/api/word/lrw500/health`.
Start the combined demo from the repository root with `./start_research_demo.ps1`
and open `http://127.0.0.1:8001/demo/live`. The eight-word model stays the default.
See [integration, measured limitations and exact protocol](../../EXPANDED_WORD_RECOGNITION.md).
The seven-source diagnostic has only 1/4 known clips correct across two classes;
it is not a 500-class benchmark and does not establish reliable webcam recognition.

Browser requests use full-frame JPEGs, measured normalized 68-point anatomical
correspondences, face counts and capture timestamps under
`lrw-meanface-mediapipe68-v1`. The backend uses the pinned publisher mean face,
while explicitly marking MediaPipe correspondence differences. The separate
`aligned_mouth` input requires 96-pixel JPEG ROIs and
`caller-aligned-mouth-v1`; alignment is the caller's responsibility.

Offline evaluation accepts an audited JSONL manifest through
`python -m pipeline.lrw_tcn.evaluate --manifest <local-manifest> --output <new-directory>`.
Existing outputs are protected. The complete real local example is
`runs/lrw500/available_samples_v1/manifest.jsonl`; its accompanying protocol
states the limited coverage, unknown native FPS and unverified training-speaker separation.
No local fitting or threshold tuning uses that diagnostic.

Source: [Lipreading using Temporal Convolutional Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks), pinned commit `0cc99fab046bd959578f2baac52e654746da4825`. Vendored model files retain their source hashes in `SOURCE.json`; only import namespaces were adjusted. The full author [LICENSE](LICENSE) is included. It permits the stated research/private-study/comparative-prototype use, not commercial or operational deployment.

The author-linked `resnet18_dctcn_video.pth` is video-only: 3D convolution, ResNet18 and Dense Temporal Convolutional Network, 500 output classes, no supplied word boundaries or audio. Author configuration and ordered labels are stored beside this README. Acquisition stores a provenance receipt and a local SHA-256; no publisher checksum is invented.

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/download_lrw_tcn.py
.\.venv-word-recognition\Scripts\python.exe -m pipeline.lrw_tcn.inference --mouth-npy data/my_aligned_mouth.npy
```

`--mouth-npy` requires real ordered uint8 grayscale ROIs `[T,H,W]`, 8–76 frames, both spatial dimensions at least 88. The caller must establish alignment compatibility. Inference applies center 88-pixel crop, division by 255, mean 0.421/std 0.165, then `[1,1,T,88,88]` input. Weights load strictly on CPU using memory mapping/meta-device assignment, with 500 ordered unique labels and fingerprint checks.

The explicit smoke-only shortcut accepts native standardized LRW videos, exactly 29 frames, 256 × 256, 25 FPS:

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pipeline.lrw_tcn.inference --lrw-centered-video .qa/lrw-public-example/AFTERNOON.mp4
```

This performs a fixed crop and does **not** reproduce publisher mean-face alignment. The actual public AFTERNOON example was correctly predicted, but one clip is not a 500-word benchmark. Confidence/margin rejection is uncalibrated. Full authorized LRW data and publisher-equivalent preprocessing/evaluation remain missing. Public checkpoint access does not authorize corpus acquisition.

This research CLI is separate from the live eight-word AV-HuBERT head and the three legacy local 500-output CNN checkpoints. Neither its confidence nor published results should be presented as local measured accuracy. See [measured results](../../WORD_ACCURACY_IMPROVEMENT.md).
