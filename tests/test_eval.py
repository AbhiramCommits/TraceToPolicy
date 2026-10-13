import numpy as np
import pytest

from tracetopolicy.eval import (
    aggregate,
    bootstrap_ci,
    evaluate_variant,
    paired_bootstrap_ci,
)


def test_bootstrap_ci_constant_values():
    ci = bootstrap_ci([1.0] * 50, n=1000, seed=1)
    assert ci["mean"] == pytest.approx(1.0)
    assert ci["lo"] == pytest.approx(1.0)
    assert ci["hi"] == pytest.approx(1.0)


def test_bootstrap_ci_known_bernoulli_contains_mean():
    values = np.array([1.0] * 250 + [0.0] * 250)  # mean exactly 0.5
    ci = bootstrap_ci(values, n=2000, seed=2)
    assert ci["mean"] == pytest.approx(0.5)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["lo"] < 0.5 < ci["hi"]


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], n=100)
    assert ci["n"] == 0


def test_paired_bootstrap_zero_when_identical():
    a = [1, 0, 1, 1, 0]
    ci = paired_bootstrap_ci(a, list(a), n=1000, seed=3)
    assert ci["mean"] == pytest.approx(0.0)
    assert ci["lo"] == pytest.approx(0.0)
    assert ci["hi"] == pytest.approx(0.0)


def test_paired_bootstrap_positive_delta():
    a = [1, 1, 1, 1, 1]
    b = [0, 0, 0, 0, 0]
    ci = paired_bootstrap_ci(a, b, n=1000, seed=4)
    assert ci["mean"] == pytest.approx(1.0)
    assert ci["lo"] > 0.5


def _task(acc1, accf, turns=1, violations=0, attempts=1, tokens=100, latency=10):
    return {
        "task_id": "t",
        "turn1_correct": acc1,
        "final_correct": accf,
        "turns_to_success": turns if accf else 0,
        "turns_used": attempts,
        "violations": violations,
        "attempts": attempts,
        "prompt_tokens": tokens // 2,
        "completion_tokens": tokens // 2,
        "tokens": tokens,
        "latency_ms": latency,
        "final_failure_mode": "correct" if accf else "wrong_filter",
    }


def test_aggregate_metrics():
    per_task = [
        _task(True, True),
        _task(False, True, turns=2),
        _task(False, False, violations=1, attempts=2),
    ]
    agg = aggregate(per_task)
    assert agg["n_tasks"] == 3
    assert agg["accuracy_turn1"] == pytest.approx(1 / 3)
    assert agg["accuracy_final"] == pytest.approx(2 / 3)
    assert agg["turns_to_success"] == pytest.approx(1.5)
    assert agg["guardrail_violation_rate"] == pytest.approx(1 / 4)
    assert agg["tool_call_accuracy"] == pytest.approx(0.75)


def test_evaluate_variant_win_loss():
    baseline = [_task(False, False), _task(True, True), _task(False, False)]
    candidate = [_task(True, True), _task(True, True), _task(False, False)]
    out = evaluate_variant(candidate, n_boot=500, seed=0, baseline=baseline)
    wtl = out["win_tie_loss_vs_baseline"]
    assert wtl == {"win": 1, "tie": 2, "loss": 0}
    assert out["win_rate_vs_baseline"] == pytest.approx(1 / 3)
    assert out["ci_accuracy_final_delta_vs_baseline"]["mean"] > 0
