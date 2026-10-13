"""Deterministic train / dev / test splits over ``data/tasks.jsonl``.

The split is a pure function of the task ids and a fixed seed, so every module
(harvesting, labeling, eval) agrees on which ``task_id`` belongs where. The
leakage assertion here is the single source of truth used as a hard failure by
the labeling stage.
"""

from __future__ import annotations

import json
import random

DEFAULT_TASKS = "data/tasks.jsonl"
SEED = 13
RATIOS = (0.70, 0.15, 0.15)


def load_tasks(path: str = DEFAULT_TASKS) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_ids(
    task_ids: list[str], seed: int = SEED, ratios: tuple = RATIOS
) -> dict[str, list[str]]:
    ids = sorted(task_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_dev = int(round(n * ratios[1]))
    train = ids[:n_train]
    dev = ids[n_train : n_train + n_dev]
    test = ids[n_train + n_dev :]
    return {"train": sorted(train), "dev": sorted(dev), "test": sorted(test)}


def get_splits(path: str = DEFAULT_TASKS) -> dict[str, list[str]]:
    tasks = load_tasks(path)
    return split_ids([t["task_id"] for t in tasks])


def task_lookup(path: str = DEFAULT_TASKS) -> dict[str, dict]:
    return {t["task_id"]: t for t in load_tasks(path)}


def assert_no_leakage(train_ids, eval_ids, name: str = "eval") -> None:
    """Hard-fail if any eval task id appears in the training set."""
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise AssertionError(
            f"LEAKAGE: {len(overlap)} {name} task id(s) present in training set: "
            f"{sorted(overlap)[:10]}"
        )


def split_summary(path: str = DEFAULT_TASKS) -> dict:
    tasks = load_tasks(path)
    splits = split_ids([t["task_id"] for t in tasks])
    return {k: len(v) for k, v in splits.items()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(split_summary(), indent=2))
