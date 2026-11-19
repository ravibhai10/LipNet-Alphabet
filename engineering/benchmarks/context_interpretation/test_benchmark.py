from copy import deepcopy
from pathlib import Path

import pytest

from benchmark_runner import ProviderFailure, load_dataset, run_benchmark, validate_dataset, validate_output

HERE = Path(__file__).parent


class StubProvider:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def suggest(self, case):
        self.calls.append(case["case_id"])
        result = self.results[case["case_id"]]
        if isinstance(result, Exception):
            raise result
        return result


def test_dataset_has_separate_synthetic_tasks_and_required_scenarios():
    cases = load_dataset(HERE / "dataset.jsonl")
    assert {case["task_mode"] for case in cases} == {"word", "expanded"}
    assert {case["difficulty"] for case in cases} >= {
        "clear", "ambiguous_disambiguated", "misleading_adversarial", "irrelevant", "absent_top_k"
    }
    assert all(case["case_id"].startswith(("w-", "x-")) for case in cases)


def test_output_validation_rejects_malformed_and_out_of_candidate_values():
    candidates = [{"word": "start", "prob": .6}, {"word": "stop", "prob": .3}]
    assert validate_output({"word": "stop", "reason": "context_fit"}, candidates).word == "stop"
    with pytest.raises(ValueError, match="output_schema_invalid"):
        validate_output({"word": "stop", "reason": "context_fit", "debug": "leak"}, candidates)
    with pytest.raises(ValueError, match="output_outside_candidates"):
        validate_output({"word": "hello", "reason": "context_fit"}, candidates)
    with pytest.raises(ValueError, match="output_reason_invalid"):
        validate_output({"word": None, "reason": "made_up"}, candidates)
    with pytest.raises(ValueError, match="output_schema_invalid"):
        validate_output("not-json-object", candidates)


@pytest.mark.parametrize("output", [
    {"word": None, "reason": "context_fit"},
    {"word": "stop", "reason": "keep_original"},
    {"word": "start", "reason": "insufficient_context"},
])
def test_output_reason_semantics_reject_inconsistent_word_values(output):
    candidates = [{"word": "start", "prob": .6}, {"word": "stop", "prob": .3}]
    with pytest.raises(ValueError, match="output_semantics_invalid"):
        validate_output(output, candidates)


def test_valid_output_reason_combinations_are_accepted():
    candidates = [{"word": "start", "prob": .6}, {"word": "stop", "prob": .3}]
    assert validate_output('{"word":"stop","reason":"context_fit"}', candidates).word == "stop"
    assert validate_output({"word": None, "reason": "keep_original"}, candidates).abstains
    assert validate_output({"word": None, "reason": "insufficient_context"}, candidates).abstains


def test_abstention_and_visual_repeat_never_score_as_correction():
    cases = load_dataset(HERE / "dataset.jsonl")[:6]
    outputs = {
        "w-clear-high-confidence": {"word": None, "reason": "keep_original"},
        "w-ambiguous-begin": {"word": "start", "reason": "context_fit"},
        "w-context-previous": {"word": "next", "reason": "context_fit"},
        "w-misleading-context": {"word": None, "reason": "keep_original"},
        "w-irrelevant-context": {"word": None, "reason": "insufficient_context"},
        "w-answer-absent-topk": {"word": None, "reason": "insufficient_context"},
    }
    cloud = StubProvider(outputs)
    local = StubProvider({})
    result = run_benchmark(cases, cloud, local)["by_task"]["word"]
    assert result["exact_suggestion_accuracy"] == .5
    assert result["exact_suggestion_denominator_actionable_corrections"] == 2
    assert result["top_k_candidate_coverage"] == .8333
    assert result["invalid_output_count"] == 0
    assert result["incorrect_confident_suggestion_rate"] == .5
    assert not local.calls


def test_primary_failure_routes_once_to_local_and_provider_failures_are_counted():
    cases = load_dataset(HERE / "dataset.jsonl")[:2]
    cloud = StubProvider({
        "w-clear-high-confidence": ProviderFailure("mock cloud down"),
        "w-ambiguous-begin": ProviderFailure("mock cloud down"),
    })
    local = StubProvider({
        "w-clear-high-confidence": {"word": None, "reason": "keep_original"},
        "w-ambiguous-begin": ProviderFailure("mock local down"),
    })
    result = run_benchmark(cases, cloud, local)["by_task"]["word"]
    assert cloud.calls == ["w-clear-high-confidence", "w-ambiguous-begin"]
    assert local.calls == ["w-clear-high-confidence", "w-ambiguous-begin"]
    assert result["fallback_attempt_count"] == 2
    assert result["fallback_success_rate"] == .5
    assert result["failure_count"] == 1
    assert result["failure_rate"] == .5
    failed_case = run_benchmark(cases[1:], StubProvider({
        "w-ambiguous-begin": ProviderFailure("mock cloud down")}), StubProvider({
        "w-ambiguous-begin": ProviderFailure("mock local down")}))["case_results"][0]
    assert failed_case["safe_prediction"] == "stop"
    assert failed_case["suggestion"] is None
    assert failed_case["status"] == "local_provider_failure"


def test_out_of_candidate_primary_response_falls_back_once_without_cloud_retry():
    cases = load_dataset(HERE / "dataset.jsonl")[:1]
    cloud = StubProvider({"w-clear-high-confidence": {"word": "outside", "reason": "context_fit"}})
    local = StubProvider({"w-clear-high-confidence": {"word": None, "reason": "keep_original"}})
    report = run_benchmark(cases, cloud, local)
    result = report["by_task"]["word"]
    assert len(cloud.calls) == 1
    assert local.calls == ["w-clear-high-confidence"]
    assert result["invalid_response_count"] == 1
    assert result["out_of_candidate_output_rate"] == .5
    assert result["failure_count"] == 0
    assert report["case_results"][0]["safe_prediction"] == "hello"


@pytest.mark.parametrize("cloud_result,expected_status", [
    (ProviderFailure("cloud unavailable"), "local_valid"),
    ("not-json", "local_valid"),
    ({}, "local_valid"),
    ({"word": "hello", "reason": "bad-reason"}, "local_valid"),
    ({"word": "outside", "reason": "context_fit"}, "local_valid"),
])
def test_each_cloud_failure_or_invalid_output_routes_once_to_local(cloud_result, expected_status):
    case = load_dataset(HERE / "dataset.jsonl")[1]
    cloud = StubProvider({case["case_id"]: cloud_result})
    local = StubProvider({case["case_id"]: {"word": "start", "reason": "context_fit"}})
    report = run_benchmark([case], cloud, local)
    assert cloud.calls == [case["case_id"]]
    assert local.calls == [case["case_id"]]
    assert report["case_results"][0]["status"] == expected_status
    assert report["case_results"][0]["suggestion"] == "start"


def test_invalid_local_output_is_rejected_without_retry_and_keeps_visual_result():
    case = load_dataset(HERE / "dataset.jsonl")[1]
    cloud = StubProvider({case["case_id"]: ProviderFailure("cloud unavailable")})
    local = StubProvider({case["case_id"]: {"word": "outside", "reason": "context_fit"}})
    report = run_benchmark([case], cloud, local)
    assert cloud.calls == [case["case_id"]]
    assert local.calls == [case["case_id"]]
    result = report["case_results"][0]
    assert result["status"] == "local_invalid_output"
    assert result["safe_prediction"] == case["visual_candidate"]
    assert result["suggestion"] is None
    assert report["by_task"]["word"]["failure_count"] == 1


def test_successful_cloud_response_does_not_call_local():
    case = load_dataset(HERE / "dataset.jsonl")[1]
    cloud = StubProvider({case["case_id"]: {"word": "start", "reason": "context_fit"}})
    local = StubProvider({})
    report = run_benchmark([case], cloud, local)
    assert cloud.calls == [case["case_id"]]
    assert local.calls == []
    assert report["case_results"][0]["status"] == "cloud_valid"


def test_unexpected_provider_exception_is_harness_error_and_does_not_fallback():
    case = load_dataset(HERE / "dataset.jsonl")[1]
    cloud = StubProvider({case["case_id"]: TimeoutError("unexpected adapter error")})
    local = StubProvider({})
    with pytest.raises(TimeoutError, match="unexpected adapter error"):
        run_benchmark([case], cloud, local)
    assert cloud.calls == [case["case_id"]]
    assert local.calls == []


@pytest.mark.parametrize("mutate,error", [
    (lambda row: row.update(should_abstain=1), "dataset_abstention_invalid"),
    (lambda row: row.update(context=17), "dataset_context_invalid"),
    (lambda row: row.update(context="   "), "dataset_context_invalid"),
    (lambda row: row.update(rationale=None), "dataset_rationale_invalid"),
    (lambda row: row.update(rationale=""), "dataset_rationale_invalid"),
    (lambda row: row.update(difficulty="unknown"), "dataset_difficulty_invalid"),
    (lambda row: row.update(difficulty=""), "dataset_difficulty_invalid"),
    (lambda row: row.update(case_id=""), "dataset_case_id_invalid"),
    (lambda row: row.update(case_id="   "), "dataset_case_id_invalid"),
    (lambda row: row.update(history="hello"), "dataset_history_invalid"),
    (lambda row: row.update(history=["cat"]), "dataset_history_invalid"),
    (lambda row: row.update(task_mode="word", candidates=[{"word": "dog", "prob": .6}]),
     "dataset_task_vocabulary_invalid"),
    (lambda row: row.update(candidates=[]), "dataset_candidates_invalid"),
    (lambda row: row.update(candidates=[{"word": "start", "prob": .6}, {"word": "start", "prob": .3}]),
     "dataset_candidates_invalid"),
    (lambda row: row.update(candidates=[{"word": "start", "prob": .3}, {"word": "stop", "prob": .6}]),
     "dataset_probabilities_invalid"),
    (lambda row: row.update(candidates=[{"word": "start", "prob": float("nan")}]),
     "dataset_probabilities_invalid"),
    (lambda row: row.update(candidates=[{"word": "start", "prob": 10**10000}]),
     "dataset_probabilities_invalid"),
    (lambda row: row.update(expected_suggestion=None, should_abstain=False),
     "dataset_expected_abstention_mismatch"),
    (lambda row: row.update(expected_suggestion="hello", should_abstain=False),
     "dataset_expected_abstention_mismatch"),
    (lambda row: row.update(should_abstain=True), "dataset_expected_abstention_mismatch"),
])
def test_dataset_validation_rejects_invalid_types_vocab_probabilities_and_labels(mutate, error):
    row = deepcopy(load_dataset(HERE / "dataset.jsonl")[1])
    mutate(row)
    with pytest.raises(ValueError, match=error):
        run_benchmark([row], StubProvider({}), StubProvider({}))


def test_dataset_validation_rejects_duplicate_case_ids():
    rows = load_dataset(HERE / "dataset.jsonl")[:2]
    duplicate = deepcopy(rows[1])
    duplicate["case_id"] = rows[0]["case_id"]
    from benchmark_runner import validate_dataset
    with pytest.raises(ValueError, match="duplicate_case_id"):
        validate_dataset([rows[0], duplicate])


def test_empty_dataset_is_rejected_by_validator_and_benchmark_entry_point():
    with pytest.raises(ValueError, match="dataset_empty"):
        validate_dataset([])
    with pytest.raises(ValueError, match="dataset_empty"):
        run_benchmark([], StubProvider({}), StubProvider({}))


def test_cloud_failure_then_valid_local_abstention_is_delivery_not_correction():
    case = load_dataset(HERE / "dataset.jsonl")[1]
    cloud = StubProvider({case["case_id"]: ProviderFailure("mock cloud unavailable")})
    local = StubProvider({case["case_id"]: {"word": None, "reason": "keep_original"}})
    report = run_benchmark([case], cloud, local)
    result = report["by_task"]["word"]
    assert report["case_results"][0]["status"] == "local_valid"
    assert report["case_results"][0]["suggestion"] is None
    assert result["fallback_success_count"] == 1
    assert result["exact_suggestion_correct_count"] == 0
    assert result["exact_suggestion_denominator_actionable_corrections"] == 1
    assert result["failure_count"] == 0


def test_adversarial_but_allowlisted_confident_output_is_not_validated_as_correct():
    case = next(c for c in load_dataset(HERE / "dataset.jsonl")
                if c["case_id"] == "w-misleading-context")
    cloud = StubProvider({case["case_id"]: {"word": "start", "reason": "context_fit"}})
    result = run_benchmark([case], cloud, StubProvider({}))["by_task"]["word"]
    assert result["invalid_response_count"] == 0  # structurally valid and inside Top-K
    assert result["incorrect_confident_suggestion_rate"] == 1.0
    assert result["abstention_recall"] == 0.0
