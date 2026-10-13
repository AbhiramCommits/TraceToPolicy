"""End-to-end smoke test on a tiny warehouse with a scripted (no-weights) generator."""


def test_end_to_end_smoke(tmp_path, monkeypatch):
    from tracetopolicy import eval as ev
    from tracetopolicy import warehouse as wh
    from tracetopolicy.agent import NL2SQLAgent
    from tracetopolicy.labeling import label_all
    from tracetopolicy.splits import get_splits, load_tasks
    from tests.fakes import gold_generator

    path = str(tmp_path / "wh.duckdb")
    wh.build_warehouse(path)
    con = wh.connect(path)

    train_ids = set(get_splits()["train"])
    tasks = [t for t in load_tasks() if t["task_id"] in train_ids][:4]

    gen = gold_generator(tasks)
    agent = NL2SQLAgent(gen, con, max_turns=2)
    traces = tmp_path / "traces.jsonl"
    summary = agent.harvest(tasks, "train", str(traces), samples_per_task=1,
                            temperature=0.0, seed=0, progress=False)
    assert summary["task_solve_rate"] == 1.0

    stats = label_all(str(traces), str(tmp_path / "datasets"), stats_path=str(tmp_path / "stats.json"))
    assert stats["sft_examples"] == len(tasks)

    monkeypatch.setenv("TRACETOPOLICY_WAREHOUSE", path)
    monkeypatch.setattr(ev, "RESULTS", str(tmp_path / "results"))
    report = ev.run_evaluation("configs/eval.yaml", dry_run=True)
    assert set(["baseline", "sft", "dpo"]).issubset(report.keys())
    assert (tmp_path / "results" / "eval_report.json").exists()
