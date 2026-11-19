# Hybrid context interpretation benchmark report

Status: preparation complete; model and provider measurements **NOT RUN**. Date: 2026-11-19.

## What was run

- Read-only source/document audit and read-only hardware/tool detection.
- Automated tests use local stub providers only; they make no network request, do not launch FastAPI, do not access SQLite, and do not read credentials.
- Synthetic fixture dataset validation and mock routing tests. These are harness checks, not model evaluation.

## Measurements

| Task | Model/provider | Exact suggestion accuracy | Top-K coverage | Abstention precision/recall | Invalid/out-of-candidate rate | Incorrect confident rate | Median/P95 latency | Fallback success | Failure rate |
|---|---|---|---|---|---|---|---|---|---|
| 8-word (`word`) | Cloud primary | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 8-word (`word`) | Local fallback | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 20-word (`expanded`) | Cloud primary | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 20-word (`expanded`) | Local fallback | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

No accuracy, webcam, latency, cost, or availability claim is made. The runner's fixture elapsed time is explicitly not model latency.

## Candidate state and environment

- OpenRouter free variants: dynamic candidate pool and rate/account limits. Exact model ID, zero price, account restrictions, provider privacy terms: **NOT VERIFIED FOR A RUN**.
- NVIDIA Nemotron 3 Ultra: current NVIDIA model page advertises a free API endpoint; exact account limits, retention and terms: **NOT VERIFIED FOR A RUN**.
- ERNIE 5.5 Flash Preview: API availability on Qianfan is announced; free status and exact account restrictions: **NOT VERIFIED**.
- Gemini API: pricing page describes free access for limited models and different data-use terms; exact eligible model/region/quota and account constraints: **NOT VERIFIED FOR A RUN**.
- Local Ollama Qwen3 4B / optional Qwen3.5 4B: no `ollama` executable or model detected; no NVIDIA GPU or `nvidia-smi`; Intel UHD reported; about 7.7 GiB total RAM and 0.5 GiB free at check time. Inference **NOT RUN**; no model installed/downloaded.
- No real cloud API, OpenAI, OpenRouter, Gemini, Groq, NVIDIA, Baidu or Qianfan request occurred.

## Test results

`python -m pytest engineering/benchmarks/context_interpretation/test_benchmark.py -q`: **41 passed**. Coverage includes all six original benchmark scenarios plus deterministic branches for cloud `ProviderFailure`, malformed JSON, invalid schema/reason, out-of-candidate cloud output, local provider failure, invalid local output, no-retry behavior, unexpected exception propagation, valid local abstention, and visual-prediction preservation. Dataset tests cover strict booleans, string fields, task vocabulary/history, candidate lists/probabilities including extremely large integers, target/abstention consistency, duplicate/blank IDs, allowed difficulty values, and non-empty dataset/context/rationale. These tests do not prove model quality, actual provider behavior, network cancellation, provider privacy, account entitlement, or production deployment performance. No other test suite was run for this correction.

The scoring denominators are documented in `BENCHMARK_PLAN.md`; mock responses are harness correctness fixtures, not model evaluation. Actual cloud/local model evaluation remains **NOT RUN**.

## Limitations and next approval gate

Synthetic labels are illustrative. A realistic evaluation requires a new consented, human-labeled dataset with speaker/session-disjoint splits, approved provider privacy terms, exact pinned models and prompts, and a separately authorized local inference setup. The existing UI notice names OpenAI; a different provider requires notice and consent review. The current provider implementation is OpenAI-specific. No production integration or runtime enablement is authorized by this preparation.
