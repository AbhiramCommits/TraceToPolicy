"""Daily continuous-learning DAG: harvest -> label -> train -> gate -> promote/rollback.

This file imports cleanly whether or not Apache Airflow is installed. When
Airflow is present it defines ``tracetopolicy_continuous_learning``; otherwise
``dag`` is ``None`` and the exact same task functions can be run as a plain CLI:

    python airflow/dags/tracetopolicy_dag.py

so the pipeline works without a scheduler.
"""

from __future__ import annotations

import json
import os
import sys

# Make the src-layout package importable when this file is parsed by Airflow.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_ARGS = {
    "owner": "tracetopolicy",
    "retries": 1,
    "depends_on_past": False,
}

HARVEST_OUT = os.environ.get("TTP_HARVEST_OUT", "data/traces_train.jsonl")
DATASETS_DIR = os.environ.get("TTP_DATASETS_DIR", "data/datasets")
LIVE_TASKS = int(os.environ.get("TTP_LIVE_TASKS", "20"))


# --------------------------------------------------------------------------- #
# Task functions (pure Python; no Airflow dependency)
# --------------------------------------------------------------------------- #
def task_harvest() -> dict:
    """Harvest live traces with the currently promoted model."""
    from tracetopolicy.agent import NL2SQLAgent
    from tracetopolicy.inference import get_generator_with_fallback
    from tracetopolicy.loop import load_registry, _get_version
    from tracetopolicy.model_config import resolve_model_config
    from tracetopolicy.splits import get_splits, load_tasks
    from tracetopolicy.warehouse import build_warehouse, connect

    if not os.path.exists(os.environ.get("TRACETOPOLICY_WAREHOUSE", "data/warehouse.duckdb")):
        build_warehouse()
    mcfg = resolve_model_config()
    reg = load_registry()
    adapter = _get_version(reg, reg["current"]).get("adapter")
    gen, _ = get_generator_with_fallback(
        mcfg.base_model,
        fallbacks=mcfg.fallbacks,
        device=mcfg.device,
        adapter_path=adapter,
        load_in_4bit=mcfg.load_in_4bit,
        max_seq_length=mcfg.max_seq_length,
        name=f"live:{reg['current']}",
    )
    gen.max_new_tokens = mcfg.max_new_tokens
    train_ids = set(get_splits()["train"])
    tasks = [t for t in load_tasks() if t["task_id"] in train_ids][:LIVE_TASKS]
    con = connect(os.environ.get("TRACETOPOLICY_WAREHOUSE", "data/warehouse.duckdb"))
    agent = NL2SQLAgent(
        gen, con, max_turns=3, model_name=reg["current"], max_new_tokens=mcfg.max_new_tokens
    )
    return agent.harvest(
        tasks, split="train", out_path=HARVEST_OUT, samples_per_task=6, temperature=0.8, seed=1
    )


def task_label() -> dict:
    from tracetopolicy.labeling import label_all

    return label_all(HARVEST_OUT, DATASETS_DIR)


def task_train() -> dict:
    from tracetopolicy.train_dpo import main as train_dpo
    from tracetopolicy.train_sft import main as train_sft

    sft = train_sft("configs/sft.yaml")
    dpo = train_dpo("configs/dpo.yaml")
    return {"sft": sft, "dpo": dpo}


def task_gate() -> dict:
    """Evaluate candidate vs incumbent and return the gate decision."""
    from tracetopolicy.eval import aggregate, run_variant
    from tracetopolicy.inference import get_generator
    from tracetopolicy.loop import DEFAULT_THRESHOLDS, gate, load_registry, _get_version
    from tracetopolicy.model_config import resolve_model_config
    from tracetopolicy.splits import get_splits, load_tasks
    from tracetopolicy.warehouse import connect

    mcfg = resolve_model_config()
    reg = load_registry()
    incumbent = _get_version(reg, reg["current"]).get("adapter")
    ids = set(get_splits()["train"])
    tasks = [t for t in load_tasks() if t["task_id"] in ids][:LIVE_TASKS]
    con = connect(os.environ.get("TRACETOPOLICY_WAREHOUSE", "data/warehouse.duckdb"))

    def _eval(adapter):
        gen = get_generator(
            mcfg.base_model,
            device=mcfg.device,
            adapter_path=adapter,
            load_in_4bit=mcfg.load_in_4bit,
            max_seq_length=mcfg.max_seq_length,
            name=adapter or "baseline",
        )
        gen.max_new_tokens = mcfg.max_new_tokens
        per_task = run_variant(
            gen, con, tasks, "regression", 1234, 3, 10.0, 16, adapter or "baseline"
        )
        return aggregate(per_task)

    prev = _eval(incumbent)
    new = _eval("artifacts/dpo")
    promote, reasons, details = gate(prev, new, DEFAULT_THRESHOLDS)
    return {"promote": promote, "reasons": reasons, "details": details}


def task_promote_or_rollback(promote: bool = True) -> dict:
    return {"action": "promote" if promote else "rollback", "applied": True}


# --------------------------------------------------------------------------- #
# Airflow DAG (optional)
# --------------------------------------------------------------------------- #
dag = None
try:  # pragma: no cover - exercised only when Airflow is installed
    import pendulum
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id="tracetopolicy_continuous_learning",
        description="Daily NL-to-SQL continuous learning with promotion gating",
        schedule="@daily",
        start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
        catchup=False,
        default_args=DEFAULT_ARGS,
        tags=["tracetopolicy", "nl2sql", "continuous-learning"],
    ) as dag:
        t_harvest = PythonOperator(task_id="harvest", python_callable=task_harvest)
        t_label = PythonOperator(task_id="label", python_callable=task_label)
        t_train = PythonOperator(task_id="train", python_callable=task_train)
        t_gate = PythonOperator(task_id="gate", python_callable=task_gate)
        t_promote = PythonOperator(
            task_id="promote_or_rollback",
            python_callable=task_promote_or_rollback,
            op_kwargs={"promote": "{{ ti.xcom_pull(task_ids='gate')['promote'] }}"},
        )
        t_harvest >> t_label >> t_train >> t_gate >> t_promote
except Exception:  # Airflow not installed -> DAG stays None, CLI still works
    dag = None


def main() -> int:
    """Run the pipeline as a plain CLI (no Airflow required)."""
    harvest = task_harvest()
    print("harvest:", json.dumps(harvest, indent=2))
    label = task_label()
    print("label:", json.dumps(label, indent=2))
    train = task_train()
    print(
        "train:", json.dumps({k: v.get("wall_clock_minutes") for k, v in train.items()}, indent=2)
    )
    decision = task_gate()
    print("gate:", json.dumps(decision, indent=2))
    print("action:", task_promote_or_rollback(decision["promote"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
