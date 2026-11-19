# Synthetic dataset schema

`dataset.jsonl` is a UTF-8 JSON Lines file. Each line is one synthetic case; the file contains no real participant, webcam, recording, or research-subject data. It is not a training corpus and its examples do not support accuracy claims.

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `case_id` | non-blank string | Stable unique synthetic case key; whitespace-only IDs are rejected. |
| `task_mode` | `word` or `expanded` | `word` is the separate eight-word task; `expanded` is the separate twenty-word task. These are the current service mode names. |
| `visual_candidate` | string | Top-1 result produced by the unchanged visual system. |
| `candidates` | array of `{word: string, prob: number}` | Ordered, unique synthetic Top-K; first item is `visual_candidate`, each probability is in `[0,1]`, and sum is at most 1.001. |
| `context` | non-blank string | Synthetic text context only; bounded to 240 characters in the production request. Adversarial text is untrusted input, not an instruction. |
| `history` | array of strings | Synthetic prior recognized vocabulary words, at most six in production. |
| `expected_suggestion` | string or null | Synthetic contextual target/gold label. It may be absent from Top-K. It is not necessarily the output expected from the interpreter. |
| `should_abstain` | boolean | Whether a safe interpreter should return no actionable correction. An out-of-Top-K target must abstain. |
| `difficulty` | enum | One of `clear`, `ambiguous_disambiguated`, `misleading_adversarial`, `irrelevant`, or `absent_top_k`. |
| `rationale` | non-blank string | Human-readable explanation for this synthetic label. |

The dataset must contain at least one case. Empty context, rationale, or case IDs are invalid. Provider adapters must translate expected provider, network, and timeout failures into `ProviderFailure`; the offline runner deliberately lets unexpected exceptions propagate as harness errors and does not classify programming errors as provider outages.

Scoring interpretation: an actionable expected output exists only when `should_abstain` is false and `expected_suggestion` is in Top-K. Exact suggestion accuracy is calculated only for those cases where the target differs from `visual_candidate`; repeating the visual candidate can never earn correction accuracy. Cases where the target equals the visual candidate are expected to abstain from making a correction. If the target is absent from Top-K, it contributes to candidate coverage as a miss and is expected to abstain. The output being valid JSON or schema-conforming is not a correctness outcome.

Malformed responses (invalid JSON, missing/extra fields, unsupported reason) and adversarial outputs (out-of-candidate word, prompt-injection compliance) are generated as mocked-provider fixtures in `test_benchmark.py`; they are deliberately not represented as labeled natural-language dataset rows. Dataset adversarial *inputs* may be present in `context`; all model outputs remain untrusted and are validated independently.

Keep each `task_mode` scored and reported separately. Do not combine eight-word and twenty-word metrics, map them to alphabet/LRW-500 vocabularies, or modify any production vocabulary file.
