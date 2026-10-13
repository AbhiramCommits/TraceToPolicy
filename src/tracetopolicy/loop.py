"""Continuous-learning loop: harvest -> label -> train -> gate -> promote/rollback.

Each cycle:
  1. harvest fresh traces with the *currently promoted* model on live traffic,
  2. auto-label them into SFT / DPO datasets,
  3. run a small incremental SFT + DPO,
  4. evaluate the candidate and the incumbent on the regression suite,
  5. promote only if final accuracy improves, no failure mode regresses by more
     than ``mode_regression_abs`` tasks, and cost/task does not rise by more than
     ``cost_tolerance_pct`` percent; otherwise roll back with a recorded reason.

The registry at ``artifacts/registry.json`` records every version, its parent,
gate decision, reasons and metrics.
"""

from __future__ import annotations

import json
import os
import time

from .drift import compute_drift, load_traces
from .labeling import label_all
from .model_config import resolve_model_config
from .training_utils import load_yaml, write_json

REGISTRY_PATH = "artifacts/registry.json"
SEED_TRACES = "data/traces_train.jsonl"
DEFAULT_THRESHOLDS = {"cost_tolerance_pct": 25.0, "mode_regression_abs": 2}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def load_registry(path: str = REGISTRY_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"current": "v0", "versions": []}


def save_registry(reg: dict, path: str = REGISTRY_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(reg, fh, indent=2)


def _get_version(reg: dict, version: str) -> dict:
    for v in reg["versions"]:
        if v["version"] == version:
            return v
    return {"version": "v0", "adapter": None, "metrics": None}


# --------------------------------------------------------------------------- #
# Promotion gate
# --------------------------------------------------------------------------- #
def gate(prev_metrics: dict, new_metrics: dict, thresholds: dict) -> tuple[bool, list[str], dict]:
    """Return ``(promote, reasons, details)``.

    Promote iff final accuracy strictly improves AND no failure mode regresses
    by more than ``mode_regression_abs`` tasks AND cost/task rises by at most
    ``cost_tolerance_pct`` percent.
    """
    reasons: list[str] = []
    prev_acc = float(prev_metrics.get("accuracy_final", 0.0))
    new_acc = float(new_metrics.get("accuracy_final", 0.0))
    improves = new_acc > prev_acc
    if not improves:
        reasons.append(f"final accuracy did not improve ({prev_acc:.3f} -> {new_acc:.3f})")

    prev_modes = prev_metrics.get("failure_modes", {})
    new_modes = new_metrics.get("failure_modes", {})
    regressions = {}
    for mode, prev_count in prev_modes.items():
        delta = new_modes.get(mode, 0) - prev_count
        threshold = thresholds.get("mode_regression_abs", 2)
        if mode == "correct":
            continue
        if delta > threshold:
            regressions[mode] = delta
    if regressions:
        reasons.append(f"failure-mode regressions exceed threshold: {regressions}")

    prev_cost = float(prev_metrics.get("cost_per_task", 0.0))
    new_cost = float(new_metrics.get("cost_per_task", 0.0))
    cost_rise_pct = ((new_cost - prev_cost) / prev_cost * 100.0) if prev_cost else 0.0
    if cost_rise_pct > thresholds.get("cost_tolerance_pct", 25.0):
        reasons.append(
            f"cost per task rose {cost_rise_pct:.1f}% (limit "
            f"{thresholds.get('cost_tolerance_pct', 25.0)}%)"
        )

    promote = improves and not regressions and (
        cost_rise_pct <= thresholds.get("cost_tolerance_pct", 25.0)
    )
    if promote:
        reasons = [
            f"final accuracy improved {prev_acc:.3f} -> {new_acc:.3f}",
            f"no failure-mode regression above {thresholds.get('mode_regression_abs', 2)}",
            f"cost per task change {cost_rise_pct:+.1f}%",
        ]
    details = {
        "prev_accuracy_final": prev_acc,
        "new_accuracy_final": new_acc,
        "accuracy_delta": new_acc - prev_acc,
        "cost_rise_pct": cost_rise_pct,
        "mode_regressions": regressions,
    }
    return promote, reasons, details


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _eval_adapter(base_model, adapter, con, tasks, cfg, seed):
    from .eval import aggregate, run_variant
    from .inference import get_generator

    mcfg = resolve_model_config()
    batch_size = int(cfg.get("batch_size", 16))
    gen = get_generator(
        base_model,
        device=mcfg.device,
        adapter_path=adapter,
        load_in_4bit=mcfg.load_in_4bit,
        max_seq_length=mcfg.max_seq_length,
        batch_size=batch_size,
        name=adapter or "baseline",
    )
    gen.max_new_tokens = mcfg.max_new_tokens
    per_task = run_variant(
        gen, con, tasks, "regression", seed, int(cfg.get("max_turns", 3)),
        float(cfg.get("timeout_s", 10.0)), batch_size, adapter or "baseline",
    )
    del gen
    return {"metrics": aggregate(per_task), "per_task": per_task}


def _degrade_datasets(sft_path: str, dpo_path: str, cycle_dir: str):
    """Return paths to deliberately corrupted datasets (to force a rollback)."""
    os.makedirs(cycle_dir, exist_ok=True)
    bad_sft = os.path.join(cycle_dir, "sft_train_degraded.jsonl")
    bad_dpo = os.path.join(cycle_dir, "dpo_train_degraded.jsonl")

    with open(sft_path) as fh:
        sft = [json.loads(line) for line in fh if line.strip()]
    for row in sft:
        row["messages"] = list(row["messages"])
        row["messages"][-1] = {"role": "assistant", "content": "SELECT 1"}
    _write_jsonl(bad_sft, sft)

    with open(dpo_path) as fh:
        dpo = [json.loads(line) for line in fh if line.strip()]
    for row in dpo:
        row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
    _write_jsonl(bad_dpo, dpo)
    return bad_sft, bad_dpo


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
# One cycle
# --------------------------------------------------------------------------- #
def run_cycle(cycle_idx: int, reg: dict, live_tasks: list[dict], degrade: bool,
              cfg: dict) -> dict:
    from .agent import NL2SQLAgent
    from .inference import get_generator_with_fallback
    from .warehouse import connect

    mcfg = resolve_model_config()
    thresholds = cfg.get("thresholds", DEFAULT_THRESHOLDS)
    cycle_dir = os.path.join("data", "loop", f"cycle{cycle_idx}")
    os.makedirs(cycle_dir, exist_ok=True)
    con = connect(os.environ.get("TRACETOPOLICY_WAREHOUSE", "data/warehouse.duckdb"))

    current_version = reg["current"]
    current_entry = _get_version(reg, current_version)
    current_adapter = current_entry.get("adapter")

    # 1. harvest live traces with the incumbent
    gen, _ = get_generator_with_fallback(
        mcfg.base_model,
        fallbacks=mcfg.fallbacks,
        device=mcfg.device,
        adapter_path=current_adapter,
        load_in_4bit=mcfg.load_in_4bit,
        max_seq_length=mcfg.max_seq_length,
        batch_size=int(cfg.get("batch_size", 16)),
        name=f"live:{current_version}",
    )
    gen.max_new_tokens = mcfg.max_new_tokens
    agent = NL2SQLAgent(
        gen, con, max_turns=int(cfg.get("max_turns", 3)),
        timeout_s=float(cfg.get("timeout_s", 10.0)),
        model_name=current_version, max_new_tokens=mcfg.max_new_tokens,
    )
    traces_path = os.path.join(cycle_dir, "live_traces.jsonl")
    harvest_summary = agent.harvest(
        live_tasks, split="train", out_path=traces_path,
        samples_per_task=int(cfg.get("live_samples", 6)), temperature=0.8,
        seed=cycle_idx * 7 + 1, batch_size=int(cfg.get("batch_size", 16)),
        progress=False,
    )
    del gen, agent

    # 2. label
    stats = label_all(traces_path, os.path.join(cycle_dir, "datasets"))
    sft_path, dpo_path = stats["sft_path"], stats["dpo_path"]
    if degrade:
        sft_path, dpo_path = _degrade_datasets(sft_path, dpo_path, cycle_dir)

    # 3. incremental SFT + DPO
    from .train_dpo import main as train_dpo
    from .train_sft import main as train_sft

    sft_out = os.path.join("artifacts", "loop", f"cycle{cycle_idx}", "sft")
    dpo_out = os.path.join("artifacts", "loop", f"cycle{cycle_idx}", "dpo")
    sft_cfg = {
        "seed": 42 + cycle_idx, "dataset": sft_path, "output_dir": sft_out,
        "max_steps": int(cfg.get("sft_steps", 50)),
        "learning_rate": float(cfg.get("sft_learning_rate", 5e-4)),
        "num_train_epochs": 8,
        "log_path": os.path.join("results", "loop", f"cycle{cycle_idx}_sft_log.json"),
        "run_name": f"cycle{cycle_idx}_sft", "wandb_project": cfg.get("wandb_project", "tracetopolicy"),
    }
    dpo_cfg = {
        "seed": 42 + cycle_idx, "dataset": dpo_path, "output_dir": dpo_out,
        "sft_adapter": sft_out, "max_steps": int(cfg.get("dpo_steps", 12)),
        "learning_rate": float(cfg.get("dpo_learning_rate", 1e-5)),
        "beta": float(cfg.get("dpo_beta", 0.5)),
        "use_weighting": False,
        "log_path": os.path.join("results", "loop", f"cycle{cycle_idx}_dpo_log.json"),
        "run_name": f"cycle{cycle_idx}_dpo", "wandb_project": cfg.get("wandb_project", "tracetopolicy"),
    }
    sft_yaml = os.path.join(cycle_dir, "sft.yaml")
    dpo_yaml = os.path.join(cycle_dir, "dpo.yaml")
    _wj_yaml(sft_yaml, sft_cfg)
    _wj_yaml(dpo_yaml, dpo_cfg)
    sft_card = train_sft(sft_yaml)
    dpo_card = train_dpo(dpo_yaml)

    # 4. regression-suite eval: candidate vs incumbent
    prev_eval = _eval_adapter(mcfg.base_model, current_adapter, con, live_tasks, cfg, seed=1234)
    new_eval = _eval_adapter(mcfg.base_model, dpo_out, con, live_tasks, cfg, seed=1234)

    # 5. gate
    promote, reasons, details = gate(prev_eval["metrics"], new_eval["metrics"], thresholds)

    # drift between seed (training) traces and this cycle's live traces
    drift = compute_drift(load_traces(SEED_TRACES), load_traces(traces_path),
                          threshold=cfg.get("drift_threshold", 0.2))
    write_json("results/drift.json", drift)

    # 6. registry
    version = f"v{len(reg['versions'])}"
    entry = {
        "version": version,
        "parent": current_version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate_adapter": dpo_out,
        "promoted_adapter": dpo_out if promote else current_adapter,
        "decision": "promote" if promote else "rollback",
        "reasons": reasons,
        "gate_details": details,
        "metrics": new_eval["metrics"],
        "incumbent_metrics": prev_eval["metrics"],
        "harvest": harvest_summary,
        "dataset_stats": stats,
        "degraded_cycle": bool(degrade),
        "drift": {"max_psi": drift["max_psi"], "drift_detected": drift["drift_detected"]},
        "sft_card": sft_card,
        "dpo_card": dpo_card,
    }
    reg["versions"].append(entry)
    if promote:
        reg["current"] = version
    save_registry(reg)

    # monitoring
    try:
        from .monitoring import DRIFT_SCORE, PROMOTIONS, ROLLBACKS, SFT_RUNS, DPO_RUNS

        SFT_RUNS.inc()
        DPO_RUNS.inc()
        (PROMOTIONS if promote else ROLLBACKS).inc()
        DRIFT_SCORE.set(drift["max_psi"])
    except Exception:
        pass

    entry["drift_detail"] = drift
    print(json.dumps({k: entry[k] for k in ["version", "decision", "reasons", "gate_details"]}, indent=2))
    return entry


def _wj_yaml(path: str, obj: dict) -> None:
    import yaml

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh)


# --------------------------------------------------------------------------- #
# Cycles
# --------------------------------------------------------------------------- #
def run_cycles(cycles: int = 2, config_path: str | None = None) -> dict:
    cfg = load_yaml(config_path) if config_path and os.path.exists(config_path) else {}
    from .splits import get_splits, load_tasks

    tasks_all = load_tasks()
    train_ids = set(get_splits()["train"])
    train_tasks = [t for t in tasks_all if t["task_id"] in train_ids]
    n_live = int(cfg.get("live_tasks", 20))
    live_tasks = train_tasks[:n_live]

    reg = load_registry()
    if not reg["versions"]:
        reg["versions"].append(
            {
                "version": "v0",
                "parent": None,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "adapter": None,
                "decision": "initial",
                "reasons": ["untrained base model"],
                "metrics": None,
                "promoted_adapter": None,
            }
        )
        save_registry(reg)

    history = []
    for i in range(1, cycles + 1):
        # alternate: odd cycles are healthy, even cycles are degraded to force a rollback
        degrade = bool(cfg.get("degrade_even_cycles", True)) and (i % 2 == 0)
        print(f"\n[loop] === cycle {i} (degrade={degrade}) ===", flush=True)
        entry = run_cycle(i, reg, live_tasks, degrade=degrade, cfg=cfg)
        history.append(entry)

    out = {"cycles": history, "final_current": reg["current"], "registry": "artifacts/registry.json"}
    write_json("results/loop_history.json", out)
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    print(json.dumps(run_cycles(a.cycles, a.config), indent=2, default=str))
