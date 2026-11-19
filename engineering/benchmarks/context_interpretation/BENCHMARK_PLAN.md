# Benchmark plan

Status: protocol prepared; no model/provider inference executed (2026-11-19).

## Scope and corpus

The initial `dataset.jsonl` contains synthetic cases only. It covers the eight-word `word` mode and twenty-word `expanded` mode separately. Include clear high-confidence predictions, ambiguous pairs, context-resolved cases, misleading/irrelevant context, abstention, answer absent from Top-K, and adversarial untrusted context. Malformed/adversarial provider outputs are tested with fixtures. Do not interpret synthetic fixture scores as product performance, webcam accuracy, or real participant results.

Before any real evaluation, create a separately reviewed, consented, human-labeled contextual dataset with held-out speakers and no overlap with prompt/model selection. Split by speaker and session to avoid leakage. Keep visual outputs frozen; preserve model, preprocessing, vocabulary, thresholds, prompt, sampling and evaluation split. Obtain legal/privacy review and explicit consent for any cloud transmission. Do not use recording consent as contextual-LLM consent.

## Providers and run protocol

Every cloud candidate is a mocked adapter in this preparation. At benchmark time, discover and record exact model IDs and pin immutable variants; do not use an unpinned free-model router for reproducibility. Confirm the actual endpoint charges $0 for the exact variant, account/region eligibility, quotas, rate limits, terms, data retention/training use and provider logging. A provider described as open-weight is not thereby free to call. Current discovery references and unresolved status are in `ARCHITECTURE.md`.

Local candidates are Ollama `qwen3:4b` and optionally Qwen3.5 4B only after confirming the exact Ollama tag/license, installed model digest, available memory, and operator approval. This harness does not invoke Ollama, shell commands, URLs, or provider SDKs. No automatic install/download. A cloud `ProviderFailure` or any invalid cloud response (malformed JSON, invalid schema/semantics, or out-of-candidate word) triggers exactly one local attempt. Neither cloud nor local is retried. Provider adapters must normalize expected provider/network/timeout errors to `ProviderFailure`; unexpected exceptions propagate as harness errors and do not trigger fallback. Invalid local output and local `ProviderFailure` end the case without an actionable suggestion; `safe_prediction` remains the original visual candidate. Empty datasets and unknown difficulty labels are rejected.

For every actual future run, freeze and report: dataset hash/version, task mode, exact provider/model/version or local model digest, prompt template/version/hash, decoding parameters, output schema, timeout, fallback policy, date/region/account tier, free price and quota evidence, consent notice version, cache policy, hardware/runtime, and dependency versions. Record per-case outcomes without context or secrets.

## Metrics, separately by task

- **Exact suggestion accuracy:** correct non-visual contextual corrections / all actionable correction cases. Actionable means `should_abstain=false`, the non-null target is in Top-K, and it differs from Top-1. Every actionable case remains in the denominator even if the provider fails or abstains; those cases count incorrect. Returning Top-1 is never counted as a contextual correction.
- **Top-K candidate coverage:** non-null targets present in Top-K / all cases with a non-null target. A target absent from Top-K is a coverage miss and must abstain.
- **Abstention precision:** valid final outputs labeled abstain and expected abstain / all valid final outputs labeled abstain. **Abstention recall:** valid final outputs labeled abstain and expected abstain / all valid final outputs whose expected label requires abstention. Cases ending in provider/validation failure are excluded from both denominators and counted in failure rate. A zero denominator is `null`.
- **Invalid output rate:** invalid cloud/local responses / all provider response attempts. Denominator includes both cloud and any one local fallback response. **Out-of-candidate output rate:** out-of-candidate responses / all provider response attempts. Report counts as well as rates.
- **Incorrect confident suggestion rate:** valid final `context_fit` suggestions that are wrong or should have abstained / all valid final non-abstaining suggestions. Repeating the visual candidate for a case expecting correction is incorrect; it never earns accuracy.
- **Median and P95 model latency:** **NOT MEASURED by this mock harness.** A future real benchmark must time the complete request from interpreter start through validated cloud or fallback result. Never use mock fixture execution time as model latency.
- **Fallback success rate:** valid schema- and semantics-conforming local responses / all cases where the local fallback was attempted. A valid abstention counts as successful fallback delivery, not as correct interpretation.
- **Failure rate:** cases with no valid final cloud or local output / all cases. Provider failure and invalid output are separately counted; an expected abstention is not a failure.
- Record prompt and model version with every metric group.

Use one run per frozen evaluation dataset. Tune prompts, thresholds and routing only on a separate validation split. Do not use protected test results for tuning. Report confidence intervals for any later human-labeled evaluation; do not promote an automatic correction policy solely on average accuracy. Compare base visual output and suggestion-assisted, user-confirmed outcome, including gains, introduced errors, abstentions and latency.

## Offline execution

From repository root, the prepared runner accepts injected mock adapters in code and has no network-capable implementation:

```powershell
python -m pytest engineering/benchmarks/context_interpretation/test_benchmark.py -q
```

The tests assert strict output semantics, JSON/schema/candidate validation, dataset type/vocabulary/probability/label validation, duplicate and blank IDs, allowed difficulty labels, non-empty dataset/context/rationale, separate task metrics, one-shot fallback for cloud provider and validation failures, unexpected-exception propagation, valid local abstention, invalid local rejection, and preservation of the visual prediction. They never start FastAPI, instantiate the production provider, touch SQLite, or read credentials. The report template deliberately records all cloud/local inference metrics as **NOT RUN**.
