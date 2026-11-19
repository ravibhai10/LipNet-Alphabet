# Endpoint inventory

Generated from the current checkout. Request schemas and middleware authorization are in openapi.json; incomplete legacy response types remain an explicit limitation.

| Method | Path | Minimum access |
|---|---|---|
| GET | `/api/recognition/capabilities` | public |
| GET | `/api/word/lrw500/health` | public |
| POST | `/api/predict/word/lrw500` | public |
| GET | `/api/word/expanded/health` | public |
| POST | `/api/predict/word/expanded` | public |
| GET | `/api/studio/config` | reviewer |
| GET | `/api/studio/inventory` | reviewer |
| POST | `/api/studio/bundles` | reviewer |
| GET | `/api/studio/export/{task}` | reviewer |
| GET | `/api/adaptive/config` | collector |
| POST | `/api/adaptive/captures` | collector |
| POST | `/api/adaptive/captures/{identifier}/predict` | collector |
| POST | `/api/adaptive/captures/{identifier}/feedback` | collector |
| POST | `/api/adaptive/captures/{identifier}/withdraw` | collector |
| GET | `/api/adaptive/records` | collector |
| GET | `/api/adaptive/records/{identifier}` | collector |
| GET | `/api/adaptive/records/{identifier}/frames/{index}` | collector |
| GET | `/api/adaptive/records/{identifier}/original-video` | collector |
| GET | `/api/adaptive/records/{identifier}/source` | collector |
| GET | `/api/adaptive/stats` | collector |
| GET | `/api/adaptive/hard-examples` | collector |
| POST | `/api/adaptive/records/{identifier}/review` | reviewer |
| POST | `/api/adaptive/datasets` | reviewer |
| GET | `/api/adaptive/datasets/{identifier}` | reviewer |
| POST | `/api/adaptive/training/jobs` | reviewer |
| GET | `/api/adaptive/training/jobs/{identifier}` | reviewer |
| GET | `/api/adaptive/models` | collector |
| GET | `/api/adaptive/models/{identifier}/comparison` | reviewer |
| POST | `/api/adaptive/models/{identifier}/final-evaluation` | reviewer |
| POST | `/api/adaptive/models/{identifier}/deploy` | reviewer |
| POST | `/api/adaptive/models/{task}/rollback` | reviewer |
| POST | `/api/adaptive/retention/expire` | reviewer |
| POST | `/api/adaptive/participants/verify` | reviewer |
| POST | `/api/adaptive/captures/{identifier}/renew` | collector |
| GET | `/api/adaptive/retention/report` | reviewer |
| POST | `/api/adaptive/retention/backups` | reviewer |
| POST | `/api/adaptive/predict` | collector |
| POST | `/api/adaptive/models/{identifier}/probe` | reviewer |
| GET | `/health` | public |
| GET | `/api/health` | public |
| GET | `/api/model/availability` | public |
| GET | `/api/evidence` | public |
| GET | `/api/dataset/stats` | public |
| GET | `/api/overview` | public |
| GET | `/api/recordings/{letter}` | reviewer |
| GET | `/api/video/{video_path}` | reviewer |
| GET | `/api/recording/{letter}/{speaker}/{session}/{recording_id}` | reviewer |
| POST | `/api/predict` | public |
| GET | `/api/word/health` | public |
| POST | `/api/word/train-personal` | reviewer |
| GET | `/api/word/personal/health` | reviewer |
| POST | `/api/predict/word/personal` | reviewer |
| POST | `/api/predict/word` | public |
| GET | `/api/models` | public |
| GET | `/api/dataset/validate` | public |
| POST | `/api/dataset/build-manifest` | reviewer |
| POST | `/api/dataset/create-splits` | reviewer |
| GET | `/api/training/check` | reviewer |
| POST | `/api/training/start` | reviewer |
| GET | `/api/training/status/{job_id}` | reviewer |
| POST | `/api/training/cancel/{job_id}` | reviewer |
| GET | `/api/training/checkpoints` | reviewer |
| GET | `/api/inference/checkpoints` | public |
| GET | `/api/evaluations` | public |
| GET | `/api/evaluation/{eval_path}` | public |
| GET | `/api/experiments` | public |
| POST | `/api/live/start` | reviewer |
| POST | `/api/live/stop` | reviewer |
| GET | `/api/live/state` | reviewer |
| POST | `/api/live/save_recording` | reviewer |
| GET | `/api/live/cameras` | reviewer |
| GET | `/` | public |

WebSocket `/ws/inference`: bounded lifecycle transport only; no verified recognition decoder.
