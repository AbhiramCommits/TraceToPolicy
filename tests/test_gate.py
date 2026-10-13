from tracetopolicy.loop import DEFAULT_THRESHOLDS, gate


def metrics(acc, modes=None, cost=100.0):
    return {
        "accuracy_final": acc,
        "cost_per_task": cost,
        "failure_modes": modes or {"correct": 5, "wrong_filter": 5},
    }


def test_gate_promotes_on_improvement():
    prev = metrics(0.5)
    new = metrics(0.7)
    promote, reasons, details = gate(prev, new, DEFAULT_THRESHOLDS)
    assert promote
    assert details["accuracy_delta"] > 0
    assert any("improved" in r for r in reasons)


def test_gate_rolls_back_on_no_improvement():
    prev = metrics(0.7)
    new = metrics(0.7)
    promote, reasons, _ = gate(prev, new, DEFAULT_THRESHOLDS)
    assert not promote
    assert any("did not improve" in r for r in reasons)


def test_gate_rolls_back_on_mode_regression():
    prev = metrics(0.5, modes={"correct": 5, "wrong_filter": 2, "syntax_error": 1})
    new = metrics(0.8, modes={"correct": 8, "wrong_filter": 8, "syntax_error": 1})
    promote, reasons, details = gate(prev, new, DEFAULT_THRESHOLDS)
    assert not promote
    assert details["mode_regressions"]
    assert any("failure-mode regressions" in r for r in reasons)


def test_gate_rolls_back_on_cost_blowup():
    prev = metrics(0.5, cost=100.0)
    new = metrics(0.7, cost=200.0)
    promote, reasons, details = gate(prev, new, DEFAULT_THRESHOLDS)
    assert not promote
    assert details["cost_rise_pct"] > 25
    assert any("cost per task rose" in r for r in reasons)


def test_gate_allows_small_cost_rise():
    prev = metrics(0.5, cost=100.0)
    new = metrics(0.7, cost=110.0)
    promote, _, details = gate(prev, new, DEFAULT_THRESHOLDS)
    assert promote
    assert details["cost_rise_pct"] == 10.0
