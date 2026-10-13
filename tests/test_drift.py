from tracetopolicy.drift import compute_drift, kl_divergence, psi


def test_psi_zero_for_identical():
    counts = [10, 20, 30, 40]
    assert psi(counts, counts) < 1e-6


def test_psi_monotonic_with_shift():
    base = [25, 25, 25, 25]
    small = [28, 24, 24, 24]
    large = [40, 20, 20, 20]
    assert psi(base, small) < psi(base, large)


def test_kl_zero_for_identical_and_positive_otherwise():
    counts = [5, 5, 5]
    assert kl_divergence(counts, counts) < 1e-6
    assert kl_divergence([9, 1], [1, 9]) > 0


def _recs(category, mode, sql_len, tokens):
    return {
        "category": category,
        "failure_mode": mode,
        "generated_sql": "x" * sql_len,
        "prompt_tokens": tokens // 2,
        "completion_tokens": tokens // 2,
    }


def test_compute_drift_identical_no_drift():
    ref = [_recs("join", "correct", 50, 200) for _ in range(50)]
    report = compute_drift(ref, list(ref), threshold=0.2)
    assert report["drift_detected"] is False
    assert report["max_psi"] < 1e-6


def test_compute_drift_detects_shift_and_is_monotonic():
    ref = [_recs("join", "correct", 50, 200) for _ in range(50)] + [
        _recs("filter", "wrong_filter", 20, 100) for _ in range(50)
    ]
    near = [_recs("join", "correct", 50, 200) for _ in range(95)] + [
        _recs("filter", "wrong_filter", 20, 100) for _ in range(5)
    ]
    far = [_recs("window", "syntax_error", 500, 900) for _ in range(100)]
    near_report = compute_drift(ref, near, threshold=0.2)
    far_report = compute_drift(ref, far, threshold=0.2)
    assert far_report["max_psi"] > near_report["max_psi"]
    assert far_report["drift_detected"] is True
