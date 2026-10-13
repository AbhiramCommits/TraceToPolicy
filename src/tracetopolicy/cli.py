"""Command line entry point: ``python -m tracetopolicy.cli <command>``.

Commands: ``warehouse``, ``harvest``, ``label``, ``eval``, ``loop``,
``dashboard``, ``info``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_WAREHOUSE = "data/warehouse.duckdb"
DEFAULT_TASKS = "data/tasks.jsonl"


def cmd_warehouse(args: argparse.Namespace) -> int:
    from .warehouse import build_warehouse, warehouse_summary

    path = build_warehouse(args.path, seed=args.seed)
    summary = warehouse_summary(path)
    print(f"warehouse: {path}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .model_config import hardware_summary, resolve_model_config

    print(
        json.dumps(
            {"config": resolve_model_config().to_dict(), "hardware": hardware_summary()}, indent=2
        )
    )
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    from .agent import NL2SQLAgent
    from .inference import get_generator_with_fallback
    from .model_config import hardware_summary, resolve_model_config
    from .splits import get_splits, load_tasks
    from .warehouse import connect

    cfg = resolve_model_config()
    model_name = args.model or cfg.base_model
    os.environ.setdefault("WANDB_MODE", "offline")

    gen, notes = get_generator_with_fallback(
        model_name,
        fallbacks=cfg.fallbacks,
        device=cfg.device,
        backend=args.backend,
        load_in_4bit=cfg.load_in_4bit,
        max_seq_length=cfg.max_seq_length,
        batch_size=args.batch_size,
        name=model_name,
    )
    for n in notes:
        print(f"[inference] {n}", file=sys.stderr)

    con = connect(args.warehouse)
    tasks = load_tasks(args.tasks)
    if args.split != "all":
        ids = set(get_splits(args.tasks)[args.split])
        tasks = [t for t in tasks if t["task_id"] in ids]
    if args.limit:
        tasks = tasks[: args.limit]

    agent = NL2SQLAgent(
        gen,
        con,
        max_turns=args.max_turns,
        timeout_s=args.timeout_s,
        model_name=model_name,
        max_new_tokens=cfg.max_new_tokens,
    )
    summary = agent.harvest(
        tasks,
        split=args.split,
        out_path=args.out,
        samples_per_task=args.samples_per_task,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    summary["model"] = model_name
    summary["tier"] = cfg.tier
    summary["hardware"] = hardware_summary()
    summary["out"] = args.out
    print(json.dumps(summary, indent=2))
    print("\nfailure-mode histogram:")
    for mode, count in sorted(summary["failure_mode_histogram"].items(), key=lambda kv: -kv[1]):
        print(f"  {mode:16s} {count}")
    print("\nNote:", cfg.notes)
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    from .labeling import label_all

    stats = label_all(args.traces, args.outdir)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import run_evaluation

    report = run_evaluation(config_path=args.config, dry_run=args.dry_run)
    print(json.dumps(report.get("headline", report), indent=2))
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    from .loop import run_cycles

    history = run_cycles(cycles=args.cycles, config_path=args.config)
    print(json.dumps(history, indent=2))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import render_dashboard

    path = render_dashboard()
    print(f"dashboard: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tracetopolicy", description="TraceToPolicy pipeline CLI")
    p.add_argument("--warehouse", default=DEFAULT_WAREHOUSE, help="path to the DuckDB warehouse")
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("warehouse", help="build the DuckDB warehouse")
    w.add_argument("--path", default=DEFAULT_WAREHOUSE)
    w.add_argument("--seed", type=int, default=42)
    w.set_defaults(func=cmd_warehouse)

    s = sub.add_parser("info", help="print hardware + model tier")
    s.set_defaults(func=cmd_info)

    h = sub.add_parser("harvest", help="harvest agent traces")
    h.add_argument("--split", default="train", choices=["train", "dev", "test", "all"])
    h.add_argument("--samples-per-task", type=int, default=4)
    h.add_argument("--out", default="data/traces_train.jsonl")
    h.add_argument("--tasks", default=DEFAULT_TASKS)
    h.add_argument("--model", default=None)
    h.add_argument("--backend", default="auto", choices=["auto", "transformers", "vllm"])
    h.add_argument("--temperature", type=float, default=0.8)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--max-turns", type=int, default=3)
    h.add_argument("--timeout-s", type=float, default=10.0)
    h.add_argument("--batch-size", type=int, default=8)
    h.add_argument("--limit", type=int, default=None)
    h.add_argument("--resume", action="store_true", help="skip tasks already present in --out")
    h.set_defaults(func=cmd_harvest)

    lb = sub.add_parser("label", help="build SFT/DPO datasets from traces")
    lb.add_argument("--traces", default="data/traces_train.jsonl")
    lb.add_argument("--outdir", default="data/datasets")
    lb.set_defaults(func=cmd_label)

    e = sub.add_parser("eval", help="run baseline vs SFT vs DPO evaluation")
    e.add_argument("--config", default="configs/eval.yaml")
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(func=cmd_eval)

    lp = sub.add_parser("loop", help="run continuous-learning cycles")
    lp.add_argument("--cycles", type=int, default=2)
    lp.add_argument("--config", default=None)
    lp.set_defaults(func=cmd_loop)

    d = sub.add_parser("dashboard", help="render the static dashboard")
    d.set_defaults(func=cmd_dashboard)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
