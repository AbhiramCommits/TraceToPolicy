"""CI smoke: build warehouse -> harvest (fake generator) -> label -> eval dry-run.

Never downloads model weights and never touches the committed results/ artifacts.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tracetopolicy import eval as ev  # noqa: E402
from tracetopolicy import warehouse as wh  # noqa: E402
from tracetopolicy.agent import NL2SQLAgent  # noqa: E402
from tracetopolicy.labeling import label_all  # noqa: E402
from tracetopolicy.splits import get_splits, load_tasks  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from tests.fakes import gold_generator  # noqa: E402


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="ttp_smoke_")
    wh_path = os.path.join(tmp, "wh.duckdb")
    wh.build_warehouse(wh_path)
    con = wh.connect(wh_path)

    train_ids = set(get_splits()["train"])
    tasks = [t for t in load_tasks() if t["task_id"] in train_ids][:6]
    gen = gold_generator(tasks)
    agent = NL2SQLAgent(gen, con, max_turns=2)
    traces = os.path.join(tmp, "traces.jsonl")
    summary = agent.harvest(tasks, "train", traces, samples_per_task=1, temperature=0.0, seed=0, progress=False)
    print("harvest:", summary["records"], "solve_rate:", summary["task_solve_rate"])

    stats = label_all(traces, os.path.join(tmp, "datasets"), stats_path=os.path.join(tmp, "stats.json"))
    print("label: sft=", stats["sft_examples"], "dpo=", stats["dpo_pairs"])

    os.environ["TRACETOPOLICY_WAREHOUSE"] = wh_path
    ev.RESULTS = os.path.join(tmp, "results")
    report = ev.run_evaluation("configs/eval.yaml", dry_run=True)
    print("eval dry-run ok; variants:", report["variants"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
