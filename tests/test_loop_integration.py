"""Integration test for the continuous-learning loop with stubbed training/eval.

Exercises ``run_cycles`` end to end (harvest -> label -> train -> gate) without
loading any model weights, and asserts a genuine promote followed by a genuine
rollback on the degraded cycle.
"""

from tests.fakes import gold_generator
from tracetopolicy import inference, loop, train_dpo, train_sft
from tracetopolicy.splits import get_splits, load_tasks
from tracetopolicy.warehouse import build_warehouse


def test_run_cycles_promote_then_rollback(tmp_path, monkeypatch):
    wh = str(tmp_path / "w.duckdb")
    build_warehouse(wh)
    monkeypatch.setenv("TRACETOPOLICY_WAREHOUSE", wh)
    monkeypatch.setenv("TRACETOPOLICY_DEVICE", "cpu")
    monkeypatch.setattr(loop, "REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setattr(loop, "write_json", lambda *a, **k: None)

    # stub training so no weights are produced
    monkeypatch.setattr(train_sft, "main", lambda cfg: {"stage": "sft", "wall_clock_minutes": 0.0})
    monkeypatch.setattr(train_dpo, "main", lambda cfg: {"stage": "dpo", "wall_clock_minutes": 0.0})

    def fake_eval(base, adapter, con, tasks, cfg, seed):
        if adapter is None:
            m = {
                "accuracy_final": 0.5,
                "failure_modes": {"correct": 10, "wrong_filter": 5},
                "cost_per_task": 100.0,
            }
        elif "cycle2" in adapter:
            m = {
                "accuracy_final": 0.2,
                "failure_modes": {"correct": 4, "missing_join": 8},
                "cost_per_task": 160.0,
            }
        else:
            m = {
                "accuracy_final": 0.9,
                "failure_modes": {"correct": 18, "wrong_filter": 2},
                "cost_per_task": 60.0,
            }
        return {"metrics": m, "per_task": []}

    monkeypatch.setattr(loop, "_eval_adapter", fake_eval)

    tasks_all = load_tasks()
    train_ids = set(get_splits()["train"])
    live = [t for t in tasks_all if t["task_id"] in train_ids][:3]
    monkeypatch.setattr(
        inference, "get_generator_with_fallback", lambda *a, **k: (gold_generator(live), [])
    )

    out = loop.run_cycles(cycles=2, config_path=None)
    decisions = [c["decision"] for c in out["cycles"]]
    assert decisions == ["promote", "rollback"]
    assert out["final_current"] == "v1"
    assert "improved" in out["cycles"][0]["reasons"][0]
    assert any("did not improve" in r for r in out["cycles"][1]["reasons"])
