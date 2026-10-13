"""Shared helpers for the SFT / DPO training entry points."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager

import yaml


def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def pick(value, fallback):
    """YAML ``null`` means 'use the hardware tier default'."""
    return fallback if value is None else value


@contextmanager
def timer():
    """Yield a callable that returns elapsed minutes."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) / 60.0


def setup_wandb(project: str = "tracetopolicy", run_name: str | None = None) -> str:
    """Force offline W&B so a run never blocks on the network. Returns report_to list."""
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ.setdefault("WANDB_NAME", run_name)
    try:
        import wandb  # noqa: F401

        return "wandb"
    except Exception:
        return "none"


def make_dataset(path: str, keep_columns: list[str]):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    drop = [c for c in ds.column_names if c not in keep_columns]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def save_model_card(
    out_dir: str,
    *,
    stage: str,
    model_cfg,
    lora_cfg: dict,
    steps: int,
    seed: int,
    dataset_path: str,
    wall_minutes: float,
    extra: dict | None = None,
) -> dict:
    card = {
        "stage": stage,
        "base_model": model_cfg.base_model,
        "tier": model_cfg.tier,
        "device": model_cfg.device,
        "load_in_4bit": model_cfg.load_in_4bit,
        "lora_rank": lora_cfg.get("r"),
        "lora_alpha": lora_cfg.get("lora_alpha"),
        "lora_dropout": lora_cfg.get("lora_dropout"),
        "lora_target_modules": lora_cfg.get("target_modules"),
        "steps": steps,
        "seed": seed,
        "dataset": dataset_path,
        "dataset_hash": file_hash(dataset_path) if os.path.exists(dataset_path) else None,
        "wall_clock_minutes": round(wall_minutes, 2),
        "notes": model_cfg.notes,
    }
    if extra:
        card.update(extra)
    write_json(os.path.join(out_dir, "model_card.json"), card)
    return card
