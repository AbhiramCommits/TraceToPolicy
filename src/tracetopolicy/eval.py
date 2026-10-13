"""Held-out evaluation: baseline vs SFT vs DPO through the same agent loop.

Every variant runs the identical multi-turn agent (greedy decoding, fixed seed)
on the test split. We report execution accuracy at turn 1 and at the end,
turns-to-success, guardrail-violation rate, token/latency cost, pairwise
win/tie/loss versus baseline, per-failure-mode breakdowns, and 10k-sample
bootstrap 95% CIs on every accuracy metric and on the paired win-rate delta.
"""

from __future__ import annotations

import json
import math
import os
import random

import numpy as np

from .agent import FAILURE_MODES, NL2SQLAgent
from .model_config import resolve_model_config
from .training_utils import load_yaml, write_json

RESULTS = "results"


# --------------------------------------------------------------------------- #
# Bootstrap statistics
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def paired_bootstrap_ci(a, b, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap CI for ``mean(a) - mean(b)`` on aligned samples."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or aa.size != bb.size:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": int(aa.size)}
    diff = aa - bb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(diff.size)}


# --------------------------------------------------------------------------- #
# Running a single variant
# --------------------------------------------------------------------------- #
def _per_task_metrics(records: list[dict], task_id: str) -> dict:
    recs = [r for r in records if r["task_id"] == task_id]
    recs.sort(key=lambda r: r["turn"])
    turn1_correct = bool(recs[0]["correct"]) if recs else False
    final_correct = any(r["correct"] for r in recs)
    turns_to_success = 0
    for r in recs:
        if r["correct"]:
            turns_to_success = r["turn"] + 1
            break
    violations = sum(1 for r in recs if r["guardrail_blocked"])
    prompt_tokens = sum(r["prompt_tokens"] for r in recs)
    completion_tokens = sum(r["completion_tokens"] for r in recs)
    latency = sum(r["latency_ms"] for r in recs)
    final_mode = recs[-1]["failure_mode"] if recs else "empty_result"
    return {
        "task_id": task_id,
        "turn1_correct": turn1_correct,
        "final_correct": final_correct,
        "turns_to_success": turns_to_success,
        "turns_used": len(recs),
        "violations": violations,
        "attempts": len(recs),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens": prompt_tokens + completion_tokens,
        "latency_ms": latency,
        "final_failure_mode": "correct" if final_correct else final_mode,
    }


def run_variant(generator, conn, tasks: list[dict], split: str, seed: int,
                max_turns: int, timeout_s: float, batch_size: int,
                model_name: str) -> list[dict]:
    from .warehouse import run_sql

    gold_lookup = {t["task_id"]: run_sql(conn, t["gold_sql"], timeout_s) for t in tasks}
    agent = NL2SQLAgent(generator, conn, max_turns=max_turns, timeout_s=timeout_s,
                        model_name=model_name, max_new_tokens=generator_max_new_tokens(generator))
    records = agent.run_tasks(
        tasks, gold_lookup, samples_per_task=1, temperature=0.0, seed=seed,
        split=split, batch_size=batch_size,
    )
    return [_per_task_metrics([r.to_json() for r in records], t["task_id"]) for t in tasks]


def generator_max_new_tokens(generator) -> int:
    return getattr(generator, "max_new_tokens", 96)


def aggregate(per_task: list[dict]) -> dict:
    n = len(per_task)
    acc1 = [1.0 if r["turn1_correct"] else 0.0 for r in per_task]
    accf = [1.0 if r["final_correct"] else 0.0 for r in per_task]
    solved = [r["turns_to_success"] for r in per_task if r["final_correct"]]
    violations = sum(r["violations"] for r in per_task)
    attempts = sum(r["attempts"] for r in per_task)
    tokens = [r["tokens"] for r in per_task]
    latency = [r["latency_ms"] for r in per_task]
    modes = {m: 0 for m in FAILURE_MODES}
    for r in per_task:
        modes[r["final_failure_mode"]] = modes.get(r["final_failure_mode"], 0) + 1
    return {
        "n_tasks": n,
        "accuracy_turn1": float(np.mean(acc1)) if n else 0.0,
        "accuracy_final": float(np.mean(accf)) if n else 0.0,
        "turns_to_success": (sum(solved) / len(solved)) if solved else 0.0,
        "num_solved": len(solved),
        "guardrail_violation_rate": (violations / attempts) if attempts else 0.0,
        "tool_call_accuracy": 1.0 - (violations / attempts) if attempts else 1.0,
        "mean_tokens_per_task": float(np.mean(tokens)) if n else 0.0,
        "mean_latency_ms_per_task": float(np.mean(latency)) if n else 0.0,
        "cost_per_task": float(np.mean(tokens)) if n else 0.0,
        "failure_modes": modes,
    }


def evaluate_variant(per_task: list[dict], n_boot: int, seed: int, baseline=None) -> dict:
    acc1 = [1.0 if r["turn1_correct"] else 0.0 for r in per_task]
    accf = [1.0 if r["final_correct"] else 0.0 for r in per_task]
    agg = aggregate(per_task)
    out = {
        "metrics": agg,
        "ci_accuracy_turn1": bootstrap_ci(acc1, n=n_boot, seed=seed),
        "ci_accuracy_final": bootstrap_ci(accf, n=n_boot, seed=seed),
    }
    if baseline is not None:
        base_accf = [1.0 if r["final_correct"] else 0.0 for r in baseline]
        wins = ties = losses = 0
        for a, b in zip(per_task, baseline, strict=True):
            if a["final_correct"] and not b["final_correct"]:
                wins += 1
            elif b["final_correct"] and not a["final_correct"]:
                losses += 1
            else:
                ties += 1
        out["win_tie_loss_vs_baseline"] = {"win": wins, "tie": ties, "loss": losses}
        out["win_rate_vs_baseline"] = wins / len(per_task) if per_task else 0.0
        out["ci_accuracy_final_delta_vs_baseline"] = paired_bootstrap_ci(accf, base_accf, n=n_boot, seed=seed)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _figure_paths():
    os.makedirs(RESULTS, exist_ok=True)
    return (
        os.path.join(RESULTS, "fig_accuracy_ci.png"),
        os.path.join(RESULTS, "fig_failure_modes.png"),
        os.path.join(RESULTS, "fig_cost_delta.png"),
    )


def _make_figures(report: dict, variant_order: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f_acc, f_modes, f_cost = _figure_paths()

    # accuracy with CI
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(variant_order))
    width = 0.35
    for i, (metric, label) in enumerate([("ci_accuracy_turn1", "turn-1"), ("ci_accuracy_final", "final")]):
        means = [report[v]["metrics"].get(f"accuracy_{'turn1' if i == 0 else 'final'}") for v in variant_order]
        lo = [report[v][metric]["lo"] for v in variant_order]
        hi = [report[v][metric]["hi"] for v in variant_order]
        err = [[m - l for m, l in zip(means, lo, strict=True)], [h - m for h, m in zip(means, hi, strict=True)]]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=err, capsize=4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_order)
    ax.set_ylabel("execution accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Execution accuracy with 95% bootstrap CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f_acc, dpi=130)
    plt.close(fig)

    # failure modes grouped bars
    fig, ax = plt.subplots(figsize=(9, 4.5))
    modes = FAILURE_MODES
    x = np.arange(len(modes))
    width = 0.8 / max(len(variant_order), 1)
    for i, v in enumerate(variant_order):
        counts = [report[v]["metrics"]["failure_modes"].get(m, 0) for m in modes]
        ax.bar(x + i * width, counts, width, label=v)
    ax.set_xticks(x + width * (len(variant_order) - 1) / 2)
    ax.set_xticklabels(modes, rotation=40, ha="right")
    ax.set_ylabel("tasks")
    ax.set_title("Final failure mode by variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f_modes, dpi=130)
    plt.close(fig)

    # cost delta vs baseline
    base = variant_order[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    deltas_tok = [report[v]["metrics"]["mean_tokens_per_task"] - report[base]["metrics"]["mean_tokens_per_task"] for v in variant_order]
    deltas_lat = [report[v]["metrics"]["mean_latency_ms_per_task"] - report[base]["metrics"]["mean_latency_ms_per_task"] for v in variant_order]
    x = np.arange(len(variant_order))
    ax.bar(x - 0.2, deltas_tok, 0.4, label="tokens/task")
    ax.bar(x + 0.2, deltas_lat, 0.4, label="latency ms/task")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_order)
    ax.set_title(f"Cost delta vs {base}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f_cost, dpi=130)
    plt.close(fig)


def _markdown_table(report: dict, order: list[str]) -> str:
    lines = ["| variant | acc@1 | acc@1 95% CI | final acc | final 95% CI | turns-to-success | guardrail-viol rate | tok/task | ms/task | win/tie/loss vs baseline | win rate |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in order:
        m = report[v]["metrics"]
        c1 = report[v]["ci_accuracy_turn1"]
        cf = report[v]["ci_accuracy_final"]
        wtl = report[v].get("win_tie_loss_vs_baseline")
        wtl_s = f"{wtl['win']}/{wtl['tie']}/{wtl['loss']}" if wtl else "-"
        wr = report[v].get("win_rate_vs_baseline")
        wr_s = f"{wr:.3f}" if wr is not None else "-"
        lines.append(
            f"| {v} | {m['accuracy_turn1']:.3f} | [{c1['lo']:.3f}, {c1['hi']:.3f}] "
            f"| {m['accuracy_final']:.3f} | [{cf['lo']:.3f}, {cf['hi']:.3f}] "
            f"| {m['turns_to_success']:.2f} | {m['guardrail_violation_rate']:.3f} "
            f"| {m['mean_tokens_per_task']:.0f} | {m['mean_latency_ms_per_task']:.0f} | {wtl_s} | {wr_s} |"
        )
    return "\n".join(lines)


def run_evaluation(config_path: str = "configs/eval.yaml", dry_run: bool = False) -> dict:
    cfg = load_yaml(config_path) if os.path.exists(config_path) else {}
    mcfg = resolve_model_config()
    if cfg.get("base_model"):
        mcfg.base_model = cfg["base_model"]
    seed = int(cfg.get("seed", 42))
    split = cfg.get("split", "test")
    max_turns = int(cfg.get("max_turns", 3))
    timeout_s = float(cfg.get("timeout_s", 10.0))
    batch_size = int(cfg.get("batch_size", 16))
    n_boot = int(cfg.get("bootstrap_samples", 10000))

    from .splits import get_splits, load_tasks
    from .warehouse import connect

    tasks_all = load_tasks()
    ids = set(get_splits()[split])
    tasks = [t for t in tasks_all if t["task_id"] in ids]

    variants = ["baseline", "sft", "dpo"]
    adapters = {
        "baseline": None,
        "sft": cfg.get("sft_adapter", "artifacts/sft"),
        "dpo": cfg.get("dpo_adapter", "artifacts/dpo"),
    }

    con = connect(os.environ.get("TRACETOPOLICY_WAREHOUSE", "data/warehouse.duckdb"))

    report: dict = {
        "config": cfg,
        "model": mcfg.base_model,
        "tier": mcfg.tier,
        "device": mcfg.device,
        "split": split,
        "seed": seed,
        "bootstrap_samples": n_boot,
        "n_tasks": len(tasks),
        "variants": variants,
    }

    if dry_run:
        from .inference import Generation, Generator

        class ScriptedGenerator(Generator):
            backend = "scripted"

            def __init__(self):
                self.name = "scripted-dry-run"
                self.max_new_tokens = 96

            def render(self, messages):
                return messages[-1]["content"]

            def generate(self, prompts, max_new_tokens=96, temperature=0.0, top_p=1.0, seed=None):
                out = []
                for i, p in enumerate(prompts):
                    task = next((t for t in tasks if t["question"] in p), tasks[0])
                    correct = (hash((task["task_id"], len(prompts))) % 2) == 0
                    sql = task["gold_sql"] if correct else "SELECT 1"
                    out.append(Generation(text=sql, prompt_tokens=len(p.split()), completion_tokens=5, latency_ms=1.0))
                return out

        generator = ScriptedGenerator()
        per_variant = {}
        for v in variants:
            per_variant[v] = run_variant(generator, con, tasks, split, seed, max_turns, timeout_s, batch_size, v)
        report["dry_run"] = True
    else:
        from .inference import get_generator

        per_variant = {}
        for v in variants:
            gen = get_generator(
                mcfg.base_model, device=mcfg.device, adapter_path=adapters[v],
                load_in_4bit=mcfg.load_in_4bit, max_seq_length=mcfg.max_seq_length,
                batch_size=batch_size, name=f"{v}:{mcfg.base_model}",
            )
            gen.max_new_tokens = mcfg.max_new_tokens
            print(f"[eval] running {v} ...", flush=True)
            per_variant[v] = run_variant(gen, con, tasks, split, seed, max_turns, timeout_s, batch_size, v)
            del gen

    # per-variant results
    for v in variants:
        base = per_variant["baseline"] if v != "baseline" else None
        report[v] = evaluate_variant(per_variant[v], n_boot, seed, baseline=base)
        report[v]["per_task"] = per_variant[v]

    # per-failure-mode delta table (dpo - baseline, sft - baseline)
    mode_deltas = {}
    for v in variants:
        if v == "baseline":
            continue
        mode_deltas[v] = {
            m: report[v]["metrics"]["failure_modes"].get(m, 0) - report["baseline"]["metrics"]["failure_modes"].get(m, 0)
            for m in FAILURE_MODES
        }
    report["failure_mode_deltas"] = mode_deltas

    report["headline"] = {
        "baseline": report["baseline"]["metrics"],
        "sft": report["sft"]["metrics"],
        "dpo": report["dpo"]["metrics"],
        "dpo_win_rate_vs_baseline": report["dpo"].get("win_rate_vs_baseline"),
    }

    write_json(os.path.join(RESULTS, "eval_report.json"), report)
    md = "# Baseline vs SFT vs DPO (held-out test split)\n\n"
    md += f"Base model: `{mcfg.base_model}` (tier `{mcfg.tier}`, device `{mcfg.device}`), "
    md += f"{len(tasks)} test tasks, {n_boot} bootstrap samples.\n\n"
    md += _markdown_table(report, variants) + "\n\n"
    md += "## Per-failure-mode delta vs baseline\n\n"
    md += "| variant | " + " | ".join(FAILURE_MODES) + " |\n"
    md += "|---|" + "---|" * len(FAILURE_MODES) + "\n"
    for v, deltas in mode_deltas.items():
        md += f"| {v} | " + " | ".join(str(deltas[m]) for m in FAILURE_MODES) + " |\n"
    with open(os.path.join(RESULTS, "eval_table.md"), "w") as fh:
        fh.write(md)

    _make_figures(report, variants)
    print(f"[eval] wrote {RESULTS}/eval_report.json, eval_table.md, and 3 figures")
    return report


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_evaluation(a.config, a.dry_run)
