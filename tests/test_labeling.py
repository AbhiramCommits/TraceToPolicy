from tracetopolicy.labeling import (
    build_dpo_dataset,
    build_sft_dataset,
    normalize_sql,
)

INIT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "question"},
]


def rec(task, turn, sql, correct, mode="wrong_filter"):
    return {
        "task_id": task,
        "turn": turn,
        "generated_sql": sql,
        "correct": correct,
        "initial_prompt": INIT,
        "failure_mode": "correct" if correct else mode,
        "guardrail_blocked": mode == "guardrail_block",
        "execution_status": "ok",
        "prompt": INIT,
    }


def test_normalize_sql_collapses_whitespace_and_case():
    assert normalize_sql("SELECT  Name\nFROM customers") == normalize_sql("select name from customers")


def test_sft_uses_final_sql_conditioned_on_original_prompt():
    traces = [
        rec("t1", 0, "SELECT bad FROM x", False, "wrong_table"),
        rec("t1", 1, "SELECT name FROM customers", True),
    ]
    sft = build_sft_dataset(traces)
    assert len(sft) == 1
    assert sft[0]["messages"][:2] == INIT
    assert sft[0]["messages"][-1]["content"] == "SELECT name FROM customers"
    assert sft[0]["repair_derived"] is True


def test_sft_prefers_shortest_correct_per_task():
    traces = [
        rec("t2", 0, "SELECT a, b, c, d FROM t WHERE x = 1 AND y = 2", True),
        rec("t2", 1, "SELECT a FROM t", True),
    ]
    sft = build_sft_dataset(traces)
    assert sft[0]["messages"][-1]["content"] == "SELECT a FROM t"


def test_dpo_pairs_require_correct_and_incorrect():
    traces = [
        rec("t3", 0, "SELECT name FROM customers", True),
        rec("t3", 0, "SELECT bad", False, "wrong_table"),
        rec("t3", 0, "SELECT name FROM customers WHERE 1=1", False, "wrong_filter"),
    ]
    # no incorrect attempts -> no pair
    only_correct = [rec("t4", 0, "SELECT 1", True)]
    assert build_dpo_dataset(only_correct) == []
    pairs = build_dpo_dataset(traces)
    assert len(pairs) >= 1
    assert pairs[0]["chosen"][0]["content"] == "SELECT name FROM customers"
    assert pairs[0]["rejected"][0]["content"] != pairs[0]["chosen"][0]["content"]


def test_dpo_dedup_near_identical_rejected():
    traces = [
        rec("t5", 0, "SELECT name FROM customers", True),
        rec("t5", 0, "select name from customers", False, "wrong_filter"),
        rec("t5", 0, "SELECT   name   FROM customers", False, "empty_result"),
    ]
    # both rejected normalize to the chosen SQL -> no pairs
    pairs = build_dpo_dataset(traces)
    assert pairs == []


def test_dpo_caps_pairs_per_task():
    traces = [rec("t6", 0, "SELECT ok FROM t", True)]
    for i in range(10):
        traces.append(rec("t6", 0, f"SELECT bad_{i} FROM t", False, "wrong_filter"))
    pairs = build_dpo_dataset(traces, max_pairs_per_task=3)
    assert len(pairs) == 3


def test_dpo_stratifies_rejected_by_failure_mode():
    traces = [
        rec("t7", 0, "SELECT ok FROM t", True),
        rec("t7", 0, "SELECT a FROM missing", False, "wrong_table"),
        rec("t7", 0, "SELECT a FROM t WHERE", False, "syntax_error"),
        rec("t7", 0, "SELECT a FROM t WHERE 1=2", False, "empty_result"),
    ]
    pairs = build_dpo_dataset(traces, max_pairs_per_task=3)
    modes = {p["rejected_failure_mode"] for p in pairs}
    assert len(modes) == len(pairs)  # each pair a distinct mode
