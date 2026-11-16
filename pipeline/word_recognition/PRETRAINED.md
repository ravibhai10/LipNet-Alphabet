# Frozen AV-HuBERT whole-word workflow

The combined private demo now has a separate 500-word research mode; start it
with `./start_research_demo.ps1` on port 8001. This eight-word head remains its
default fallback. See [current integration and limitations](../../EXPANDED_WORD_RECOGNITION.md).

The completed eight-word experiment and limits are documented in [WORD_ACCURACY_IMPROVEMENT.md](../../WORD_ACCURACY_IMPROVEMENT.md). It recognizes complete words, not letter slots. Do not activate this head through the alphabet backend.

Use the existing `.venv-word-recognition` environment for manifest preparation/head fitting and `.venv-avhubert-cpu` for official encoder extraction/runtime. The stage runner chooses these environments, restricts CPU threads, records native exit codes and reuses completed outputs. Existing source videos and checkpoints are preserved. A new run needs the authentic MIRACL source manifest, landmark model and official AV-HuBERT encoder already present locally.

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/improve_word_accuracy.py --run runs/word_accuracy/another_protocol --through train
# Only after validation selection has been frozen:
.\.venv-word-recognition\Scripts\python.exe scripts/improve_word_accuracy.py --run runs/word_accuracy/another_protocol --through evaluate
```

The current manifest protocol is explicit to MIRACL: eight unambiguous words, speaker-disjoint training/validation/test, quarantined historical test utterances, and separate phrase diagnostics. Changing datasets requires reviewing this protocol rather than substituting arbitrary manifests. Recorded caches bind source bytes, preprocessing, measured landmarks and encoder fingerprints. Failed tracking is explicit; evaluation failures remain errors in the denominator.

For the completed research model:

```powershell
.\start_word_demo.ps1
# Open http://127.0.0.1:8001/demo/live
```

The launcher sets `LIPNET_WORD_BACKEND=avhubert` and the word-head path only for its process, restores shell values afterwards, and does not change `LIPNET_MODEL_BACKEND`. Use another port via `-Port 8002` if necessary. The model is a private research demo, not a production service; webcam accuracy and reliable rejection of unseen isolated words remain unmeasured. A 94.79% offline score on eight words does not apply to arbitrary English.
