"""Offline-only benchmark helpers. No HTTP, provider SDK, database, or subprocess use."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REASONS = {"context_fit", "insufficient_context", "keep_original"}
PROMPT_VERSION = "context-only-eval-v1"
WORD_VOCAB = {"choose", "connection", "hello", "navigation", "next", "previous", "start", "stop"}
EXPANDED_VOCAB = {
    "a", "bye", "can", "cat", "choose", "connection", "demo", "dog", "hello", "here",
    "is", "lips", "my", "navigation", "next", "previous", "read", "start", "stop", "you",
}
TASK_VOCAB = {"word": WORD_VOCAB, "expanded": EXPANDED_VOCAB}
DIFFICULTIES = {
    "clear", "ambiguous_disambiguated", "misleading_adversarial", "irrelevant", "absent_top_k",
}


class ProviderFailure(Exception):
    """Expected mock provider failure; the harness performs one local fallback only."""


class OutputValidationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Provider(Protocol):
    """Offline provider contract: adapters normalize expected provider/network errors to ProviderFailure.

    Unexpected exceptions intentionally propagate as harness/programming errors; they are not
    silently reclassified as outages and do not trigger fallback.
    """
    def suggest(self, case: dict) -> object: ...


@dataclass(frozen=True)
class ValidatedOutput:
    word: str | None
    reason: str

    @property
    def abstains(self) -> bool:
        return self.reason != "context_fit"


def validate_output(raw: object, candidates: list[dict]) -> ValidatedOutput:
    """Parse/validate a strict response and enforce reason-specific output semantics."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise OutputValidationError("output_schema_invalid") from None
    if not isinstance(raw, dict) or set(raw) != {"word", "reason"}:
        raise OutputValidationError("output_schema_invalid")
    word, reason = raw["word"], raw["reason"]
    if not isinstance(reason, str) or reason not in REASONS:
        raise OutputValidationError("output_reason_invalid")
    if reason == "context_fit":
        if not isinstance(word, str) or not word:
            raise OutputValidationError("output_semantics_invalid")
        if word not in {item["word"] for item in candidates}:
            raise OutputValidationError("output_outside_candidates")
    elif word is not None:
        raise OutputValidationError("output_semantics_invalid")
    return ValidatedOutput(word, reason)


def _valid_dataset_row(row: object, required: set[str], seen: set[str]) -> None:
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("dataset_schema_invalid")
    case_id = row["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("dataset_case_id_invalid")
    if case_id in seen:
        raise ValueError("duplicate_case_id")
    seen.add(case_id)

    task = row["task_mode"]
    if not isinstance(task, str) or task not in TASK_VOCAB:
        raise ValueError("dataset_task_mode_invalid")
    vocabulary = TASK_VOCAB[task]
    context, rationale = row["context"], row["rationale"]
    if not isinstance(context, str) or not context.strip() or len(context) > 240:
        raise ValueError("dataset_context_invalid")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("dataset_rationale_invalid")
    history = row["history"]
    if (not isinstance(history, list) or len(history) > 6
            or any(not isinstance(word, str) or word not in vocabulary for word in history)):
        raise ValueError("dataset_history_invalid")
    should_abstain = row["should_abstain"]
    if type(should_abstain) is not bool:
        raise ValueError("dataset_abstention_invalid")
    difficulty = row["difficulty"]
    if not isinstance(difficulty, str) or difficulty not in DIFFICULTIES:
        raise ValueError("dataset_difficulty_invalid")

    candidates = row["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("dataset_candidates_invalid")
    words: list[str] = []
    probabilities: list[float] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"word", "prob"}:
            raise ValueError("dataset_candidates_invalid")
        word, probability = candidate["word"], candidate["prob"]
        if not isinstance(word, str) or word not in vocabulary:
            raise ValueError("dataset_task_vocabulary_invalid")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("dataset_probabilities_invalid")
        # Range comparisons reject arbitrarily large integers before math.isfinite()
        # attempts a potentially overflowing int-to-float conversion.
        if probability < 0 or probability > 1:
            raise ValueError("dataset_probabilities_invalid")
        if not math.isfinite(probability):
            raise ValueError("dataset_probabilities_invalid")
        words.append(word)
        probabilities.append(probability)
    if len(set(words)) != len(words):
        raise ValueError("dataset_candidates_invalid")
    if probabilities != sorted(probabilities, reverse=True) or sum(probabilities) > 1.001:
        raise ValueError("dataset_probabilities_invalid")

    visual = row["visual_candidate"]
    if not isinstance(visual, str) or visual != words[0]:
        raise ValueError("dataset_visual_candidate_invalid")
    expected = row["expected_suggestion"]
    if expected is not None and (not isinstance(expected, str) or expected not in vocabulary):
        raise ValueError("dataset_expected_suggestion_invalid")
    target_is_candidate = expected is not None and expected in words
    # A correction target must be an allowed non-visual candidate. Abstention is
    # required when the target is absent, unknown, or already equals Top-1.
    if expected is None or expected == visual or not target_is_candidate:
        if not should_abstain:
            raise ValueError("dataset_expected_abstention_mismatch")
    elif should_abstain:
        raise ValueError("dataset_expected_abstention_mismatch")


def load_dataset(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return validate_dataset(rows)


def validate_dataset(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset_empty")
    required = {"case_id", "task_mode", "visual_candidate", "candidates", "context", "history",
                "expected_suggestion", "should_abstain", "difficulty", "rationale"}
    seen: set[str] = set()
    for row in rows:
        _valid_dataset_row(row, required, seen)
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _call(provider: Provider, case: dict) -> ValidatedOutput:
    return validate_output(provider.suggest(case), case["candidates"])


def run_benchmark(cases: list[dict], cloud: Provider, local: Provider) -> dict:
    """Run a cloud mock, then at most one local mock if cloud fails or output is invalid."""
    validate_dataset(cases)
    outcomes = []
    for case in cases:
        attempts = []
        fallback_attempted = False
        output = None
        final_status = "cloud_valid"
        try:
            output = _call(cloud, case)
        except ProviderFailure:
            attempts.append({"valid": False, "invalid": False, "out_of_candidate": False})
            fallback_attempted = True
            final_status = "cloud_provider_failure"
        except OutputValidationError as exc:
            attempts.append({"valid": False, "invalid": True,
                             "out_of_candidate": exc.code == "output_outside_candidates"})
            fallback_attempted = True
            final_status = "cloud_invalid_output"
        else:
            attempts.append({"valid": True, "invalid": False, "out_of_candidate": False})

        if fallback_attempted:
            try:
                output = _call(local, case)  # Exactly one attempt. Never retry either provider.
            except ProviderFailure:
                attempts.append({"valid": False, "invalid": False, "out_of_candidate": False})
                output = None
                final_status = "local_provider_failure"
            except OutputValidationError as exc:
                attempts.append({"valid": False, "invalid": True,
                                 "out_of_candidate": exc.code == "output_outside_candidates"})
                output = None
                final_status = "local_invalid_output"
            else:
                attempts.append({"valid": True, "invalid": False, "out_of_candidate": False})
                final_status = "local_valid"

        outcomes.append({
            "case": case,
            "output": output,
            "final_status": final_status,
            "fallback_attempted": fallback_attempted,
            "attempts": attempts,
            # On every failure path, retain the original visual candidate as the
            # safe prediction. A contextual suggestion is a separate value.
            "safe_prediction": case["visual_candidate"],
            "suggestion": output.word if output is not None and not output.abstains else None,
        })

    report = {"prompt_version": PROMPT_VERSION, "model_latency": "NOT RUN — mock timing is not measured",
              "fixture_metrics_are_not_model_accuracy": True, "by_task": {}}
    for task in ("word", "expanded"):
        group = [item for item in outcomes if item["case"]["task_mode"] == task]
        actionable_total = exact = 0
        target_total = target_covered = 0
        abstention_tp = abstention_fp = abstention_fn = 0
        invalid_responses = out_of_candidate_responses = response_attempts = 0
        fallback_attempts = fallback_successes = 0
        nonabstaining = incorrect_confident = 0
        failed_cases = 0

        for item in group:
            case, output = item["case"], item["output"]
            expected = case["expected_suggestion"]
            candidate_words = {candidate["word"] for candidate in case["candidates"]}
            if expected is not None:
                target_total += 1
                target_covered += int(expected in candidate_words)

            response_attempts += len(item["attempts"])
            invalid_responses += sum(int(attempt["invalid"]) for attempt in item["attempts"])
            out_of_candidate_responses += sum(int(attempt["out_of_candidate"]) for attempt in item["attempts"])
            fallback_attempts += int(item["fallback_attempted"])
            fallback_successes += int(item["fallback_attempted"] and item["final_status"] == "local_valid")
            failed_cases += int(output is None)

            actionable = (not case["should_abstain"] and expected is not None
                          and expected in candidate_words and expected != case["visual_candidate"])
            if actionable:
                actionable_total += 1
                exact += int(output is not None and not output.abstains and output.word == expected)

            # Abstention precision/recall exclude cases with no valid final
            # output; provider failures are measured separately as failures.
            if output is not None:
                predicted_abstain = output.abstains
                expected_abstain = case["should_abstain"]
                abstention_tp += int(predicted_abstain and expected_abstain)
                abstention_fp += int(predicted_abstain and not expected_abstain)
                abstention_fn += int(not predicted_abstain and expected_abstain)
                if not predicted_abstain:
                    nonabstaining += 1
                    incorrect_confident += int(
                        case["should_abstain"] or expected is None or output.word != expected
                    )

        report["by_task"][task] = {
            "case_count": len(group),
            "exact_suggestion_accuracy": _ratio(exact, actionable_total),
            "exact_suggestion_correct_count": exact,
            "exact_suggestion_denominator_actionable_corrections": actionable_total,
            "top_k_candidate_coverage": _ratio(target_covered, target_total),
            "top_k_covered_target_count": target_covered,
            "top_k_non_null_target_denominator": target_total,
            "abstention_precision": _ratio(abstention_tp, abstention_tp + abstention_fp),
            "abstention_precision_denominator_valid_final_abstain_predictions": abstention_tp + abstention_fp,
            "abstention_recall": _ratio(abstention_tp, abstention_tp + abstention_fn),
            "abstention_recall_denominator_valid_final_expected_abstentions": abstention_tp + abstention_fn,
            "invalid_output_rate": _ratio(invalid_responses, response_attempts),
            "invalid_response_count": invalid_responses,
            "invalid_output_count": invalid_responses,
            "invalid_output_rate_denominator_provider_response_attempts": response_attempts,
            "out_of_candidate_output_rate": _ratio(out_of_candidate_responses, response_attempts),
            "out_of_candidate_response_count": out_of_candidate_responses,
            "out_of_candidate_rate_denominator_provider_response_attempts": response_attempts,
            "incorrect_confident_suggestion_rate": _ratio(incorrect_confident, nonabstaining),
            "incorrect_confident_denominator_valid_final_nonabstaining_suggestions": nonabstaining,
            "fallback_success_rate": _ratio(fallback_successes, fallback_attempts),
            "fallback_success_denominator_local_fallback_attempts": fallback_attempts,
            "fallback_attempt_count": fallback_attempts,
            "fallback_success_count": fallback_successes,
            "failure_rate": _ratio(failed_cases, len(group)),
            "failure_denominator_cases_with_no_valid_final_output": len(group),
            "failure_count": failed_cases,
            "provider_response_attempt_count": response_attempts,
        }

    # Safe, non-contextual case evidence for tests and offline review only.
    report["case_results"] = [
        {"case_id": item["case"]["case_id"], "safe_prediction": item["safe_prediction"],
         "suggestion": item["suggestion"], "status": item["final_status"],
         "fallback_attempted": item["fallback_attempted"]}
        for item in outcomes
    ]
    return report
