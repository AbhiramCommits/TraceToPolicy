import pytest

from tracetopolicy.labeling import REWARD_WEIGHTS, score_trace


def rec(**kw):
    base = {"correct": False, "turn": 0, "failure_mode": "wrong_filter", "guardrail_blocked": False, "execution_status": "ok"}
    base.update(kw)
    return base


def test_reward_correct_first_turn():
    assert score_trace(rec(correct=True, failure_mode="correct")) == pytest.approx(1.0)


def test_reward_correct_after_repair():
    assert score_trace(rec(correct=True, turn=2, failure_mode="correct")) == pytest.approx(0.8)


def test_reward_wrong_answer():
    assert score_trace(rec(correct=False, turn=0)) == pytest.approx(0.0)


def test_reward_guardrail_penalty():
    r = score_trace(rec(correct=False, failure_mode="guardrail_block", guardrail_blocked=True))
    assert r == pytest.approx(-REWARD_WEIGHTS["guardrail_penalty"])


def test_reward_timeout_penalty():
    r = score_trace(rec(correct=False, failure_mode="timeout", execution_status="timeout"))
    assert r == pytest.approx(-REWARD_WEIGHTS["timeout_penalty"])


def test_reward_empty_penalty():
    r = score_trace(rec(correct=False, failure_mode="empty_result"))
    assert r == pytest.approx(-REWARD_WEIGHTS["empty_penalty"])


def test_reward_guardrail_plus_turn_penalty():
    r = score_trace(rec(correct=False, turn=2, failure_mode="guardrail_block", guardrail_blocked=True))
    assert r == pytest.approx(-0.5 - 0.2)
