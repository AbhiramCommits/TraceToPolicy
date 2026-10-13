from tracetopolicy import monitoring


def _report():
    return {
        "baseline": {
            "metrics": {
                "accuracy_final": 0.25,
                "accuracy_turn1": 0.25,
                "tool_call_accuracy": 0.9,
                "turns_to_success": 1.0,
            }
        },
        "sft": {
            "metrics": {
                "accuracy_final": 0.25,
                "accuracy_turn1": 0.25,
                "tool_call_accuracy": 0.8,
                "turns_to_success": 1.0,
            }
        },
        "dpo": {
            "metrics": {
                "accuracy_final": 0.25,
                "accuracy_turn1": 0.2,
                "tool_call_accuracy": 0.84,
                "turns_to_success": 1.2,
            },
            "win_rate_vs_baseline": 0.1,
        },
    }


def test_update_from_eval_sets_gauges():
    monitoring.update_from_eval(_report())
    changed_before = monitoring.WIN_RATE._value.get()
    monitoring.update_from_eval(_report())
    assert monitoring.WIN_RATE._value.get() == changed_before


def test_update_drift_and_observe_tasks():
    monitoring.update_drift(0.42)
    monitoring.observe_tasks(
        [{"latency_ms": 12.0, "tokens": 100}, {"latency_ms": 500.0, "tokens": 5000}]
    )
    assert monitoring.DRIFT_SCORE._value.get() == 0.42


def test_metrics_text_has_expected_series():
    t = monitoring.metrics_text()
    for name in [
        "tracetopolicy_win_rate",
        "tracetopolicy_final_accuracy",
        "tracetopolicy_tool_call_accuracy",
        "tracetopolicy_drift_score",
    ]:
        assert name in t
