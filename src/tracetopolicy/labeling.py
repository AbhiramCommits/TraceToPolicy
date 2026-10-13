"""Turn raw agent traces into (a) an SFT dataset and (b) DPO preference pairs.

The reward is *differential execution correctness*: an attempt is rewarded when
its result set matches the gold query's result set (order-insensitive,
type-tolerant). We never ask a human or a model to grade SQL — the warehouse is
the judge.

SFT targets are built so the model internalizes repair: for a task the agent
only solved after a failed first attempt, the training example conditions on the
*original* first-turn prompt and targets the *final correct* SQL.

DPO pairs use the same first-turn prompt for ``chosen``/``rejected`` so the two
completions are directly comparable. Rejected samples are stratified across
failure modes and near-duplicate SQL is collapsed.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import sqlglot

from .splits import assert_no_leakage, get_splits

# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
REWARD_WEIGHTS = {
    "correct": 1.0,
    "per_turn_penalty": 0.1,
    "guardrail_penalty": 0.5,
    "timeout_penalty": 0.3,
    "empty_penalty": 0.1,
}


def score_trace(record: dict) -> float:
    """Reward for a single trace record.

    Formula (documented exactly as implemented):

        reward = (1.0 if correct else 0.0)
                 - 0.1 * turn_index
                 - 0.5 if guardrail_blocked
                 - 0.3 if failure_mode == "timeout"
                 - 0.1 if failure_mode == "empty_result"

    The turn penalty rewards solving a task on the first attempt rather than
    after a long repair chain. Guardrail blocks (attempts to write / run DDL) and
    timeouts are penalized hardest because they are operational failures, not
    merely wrong answers. The result is not clipped; a guardrail block on turn 2
    can score as low as ``-0.7``.
    """
    reward = 1.0 if record.get("correct") else 0.0
    reward -= REWARD_WEIGHTS["per_turn_penalty"] * int(record.get("turn", 0))
    if record.get("guardrail_blocked") or record.get("failure_mode") == "guardrail_block":
        reward -= REWARD_WEIGHTS["guardrail_penalty"]
    if record.get("failure_mode") == "timeout" or record.get("execution_status") == "timeout":
        reward -= REWARD_WEIGHTS["timeout_penalty"]
    if record.get("failure_mode") == "empty_result":
        reward -= REWARD_WEIGHTS["empty_penalty"]
    return reward


# --------------------------------------------------------------------------- #
# SQL normalization / dedup
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    """Canonicalize SQL for dedup: parse+normalize with sqlglot, else whitespace/case."""
    if not sql:
        return ""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
        return tree.sql(dialect="duckdb", normalize=True, pretty=False).strip().lower()
    except Exception:
        return _WS.sub(" ", sql).strip().rstrip(";").lower()


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _correct_sqls_by_task(traces: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in traces:
        if rec.get("correct") and rec.get("generated_sql"):
            out[rec["task_id"]].append(rec)
    return out


def build_sft_dataset(traces: list[dict]) -> list[dict]:
    """One chat example per task: original first prompt -> shortest correct SQL."""
    by_task = _correct_sqls_by_task(traces)
    examples: list[dict] = []
    for task_id, recs in by_task.items():
        # shortest correct completion wins; prefer earlier turns to break ties
        best = min(recs, key=lambda r: (len(r["generated_sql"]), int(r.get("turn", 0))))
        messages = list(best["initial_prompt"]) + [
            {"role": "assistant", "content": best["generated_sql"]}
        ]
        examples.append(
            {
                "task_id": task_id,
                "messages": messages,
                "repair_derived": int(best.get("turn", 0)) > 0,
                "source_turn": int(best.get("turn", 0)),
            }
        )
    examples.sort(key=lambda e: e["task_id"])
    return examples


def build_dpo_dataset(traces: list[dict], max_pairs_per_task: int = 3) -> list[dict]:
    """Preference pairs: original first prompt + chosen SQL vs a stratified rejected SQL."""
    correct = _correct_sqls_by_task(traces)

    # group incorrect attempts per task, keyed by normalized SQL + failure mode
    incorrect: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in traces:
        if rec.get("correct") or not rec.get("generated_sql"):
            continue
        key = normalize_sql(rec["generated_sql"])
        if not key:
            continue
        prev = incorrect[rec["task_id"]].get(key)
        # keep the highest-reward (usually earliest/last-chance) representative
        if prev is None or score_trace(rec) > score_trace(prev):
            incorrect[rec["task_id"]][key] = rec

    pairs: list[dict] = []
    for task_id, recs in correct.items():
        chosen_rec = min(recs, key=lambda r: (len(r["generated_sql"]), int(r.get("turn", 0))))
        chosen_norm = normalize_sql(chosen_rec["generated_sql"])
        prompt = list(chosen_rec["initial_prompt"])

        # stratify rejected across failure modes: one per distinct mode, then fill
        by_mode: dict[str, list[dict]] = defaultdict(list)
        for rec in incorrect.get(task_id, {}).values():
            if normalize_sql(rec["generated_sql"]) == chosen_norm:
                continue
            by_mode[rec["failure_mode"]].append(rec)

        ordered: list[dict] = []
        modes_sorted = sorted(by_mode, key=lambda m: -len(by_mode[m]))
        for mode in modes_sorted:
            recs_sorted = sorted(by_mode[mode], key=lambda r: -score_trace(r))
            ordered.append(recs_sorted[0])
        # fill remaining slots with the next-best rejected samples
        leftovers = [
            r
            for mode in modes_sorted
            for r in sorted(by_mode[mode], key=lambda x: -score_trace(x))[1:]
        ]
        ordered.extend(sorted(leftovers, key=lambda r: -score_trace(r)))

        seen = {chosen_norm}
        for rec in ordered:
            norm = normalize_sql(rec["generated_sql"])
            if norm in seen:
                continue
            seen.add(norm)
            pairs.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "chosen": [{"role": "assistant", "content": chosen_rec["generated_sql"]}],
                    "rejected": [{"role": "assistant", "content": rec["generated_sql"]}],
                    "rejected_failure_mode": rec["failure_mode"],
                }
            )
            if len(seen) - 1 >= max_pairs_per_task:
                break
    pairs.sort(key=lambda p: p["task_id"])
    return pairs


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_traces(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def label_all(
    traces_path: str, outdir: str = "data/datasets", stats_path: str = "results/dataset_stats.json"
) -> dict:
    """Build SFT + DPO datasets, enforce the leakage guard, write artifacts."""
    traces = load_traces(traces_path)
    sft = build_sft_dataset(traces)
    dpo = build_dpo_dataset(traces)

    # ---- leakage guard (hard failure) ----
    splits = get_splits()
    eval_ids = set(splits["dev"]) | set(splits["test"])
    used = {e["task_id"] for e in sft} | {p["task_id"] for p in dpo}
    # any eval task present at all is a bug in the harvest split wiring
    leaked = used & eval_ids
    if leaked:
        raise AssertionError(f"LEAKAGE: eval tasks present in train datasets: {sorted(leaked)}")
    assert_no_leakage([e["task_id"] for e in sft], sorted(eval_ids), "eval")

    _write_jsonl(os.path.join(outdir, "sft_train.jsonl"), sft)
    _write_jsonl(os.path.join(outdir, "dpo_train.jsonl"), dpo)

    rejected_modes: dict[str, int] = defaultdict(int)
    for p in dpo:
        rejected_modes[p["rejected_failure_mode"]] += 1

    stats = {
        "traces": len(traces),
        "train_tasks_with_correct": len(sft),
        "sft_examples": len(sft),
        "repair_derived_examples": sum(1 for e in sft if e["repair_derived"]),
        "dpo_pairs": len(dpo),
        "dpo_tasks": len({p["task_id"] for p in dpo}),
        "max_pairs_per_task": 3,
        "rejected_failure_modes": dict(sorted(rejected_modes.items(), key=lambda kv: -kv[1])),
        "leakage_check": "passed",
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "sft_path": os.path.join(outdir, "sft_train.jsonl"),
        "dpo_path": os.path.join(outdir, "dpo_train.jsonl"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)) or ".", exist_ok=True)
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    return stats


if __name__ == "__main__":  # pragma: no cover
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/traces_train.jsonl"
    print(json.dumps(label_all(path), indent=2))
