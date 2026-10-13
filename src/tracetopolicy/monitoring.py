"""Prometheus instrumentation for the NL-to-SQL agent.

Exposes gauges for win rate, tool-call accuracy, turns-to-success and drift
score, counters for pipeline events, and histograms for latency / tokens. The
``/metrics`` endpoint is served by :func:`serve`. All registration is guarded so
re-importing the module (e.g. under pytest) never raises a duplicate-time-series
error.
"""

from __future__ import annotations

import os

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

REGISTRY = CollectorRegistry(auto_describe=True)

WIN_RATE = Gauge("tracetopolicy_win_rate", "DPO/SFT win rate vs baseline", registry=REGISTRY)
FINAL_ACCURACY = Gauge("tracetopolicy_final_accuracy", "final execution accuracy", ["variant"], registry=REGISTRY)
TURN1_ACCURACY = Gauge("tracetopolicy_turn1_accuracy", "turn-1 execution accuracy", ["variant"], registry=REGISTRY)
TOOL_CALL_ACCURACY = Gauge("tracetopolicy_tool_call_accuracy", "fraction of attempts passing the guardrail", ["variant"], registry=REGISTRY)
TURNS_TO_SUCCESS = Gauge("tracetopolicy_turns_to_success", "mean turns to success", ["variant"], registry=REGISTRY)
DRIFT_SCORE = Gauge("tracetopolicy_drift_score", "max PSI between train and live traces", registry=REGISTRY)

HARVEST_RECORDS = Counter("tracetopolicy_harvest_records_total", "trace records harvested", registry=REGISTRY)
SFT_RUNS = Counter("tracetopolicy_sft_runs_total", "SFT runs", registry=REGISTRY)
DPO_RUNS = Counter("tracetopolicy_dpo_runs_total", "DPO runs", registry=REGISTRY)
PROMOTIONS = Counter("tracetopolicy_promotions_total", "model promotions", registry=REGISTRY)
ROLLBACKS = Counter("tracetopolicy_rollbacks_total", "model rollbacks", registry=REGISTRY)

LATENCY = Histogram(
    "tracetopolicy_task_latency_ms",
    "per-task latency in ms",
    buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
    registry=REGISTRY,
)
TOKENS = Histogram(
    "tracetopolicy_tokens_per_task",
    "prompt+completion tokens per task",
    buckets=(50, 100, 200, 400, 800, 1600, 3200, 6400),
    registry=REGISTRY,
)


def update_from_eval(report: dict) -> None:
    """Push an eval report's headline numbers into the gauges."""
    for variant in ["baseline", "sft", "dpo"]:
        if variant not in report or "metrics" not in report[variant]:
            continue
        m = report[variant]["metrics"]
        FINAL_ACCURACY.labels(variant=variant).set(m.get("accuracy_final", 0.0))
        TURN1_ACCURACY.labels(variant=variant).set(m.get("accuracy_turn1", 0.0))
        TOOL_CALL_ACCURACY.labels(variant=variant).set(m.get("tool_call_accuracy", 1.0))
        TURNS_TO_SUCCESS.labels(variant=variant).set(m.get("turns_to_success", 0.0))
    wr = report.get("dpo", {}).get("win_rate_vs_baseline")
    if wr is not None:
        WIN_RATE.set(wr)


def update_drift(score: float) -> None:
    DRIFT_SCORE.set(score)


def observe_tasks(per_task: list[dict]) -> None:
    for r in per_task:
        LATENCY.observe(r.get("latency_ms", 0.0))
        TOKENS.observe(r.get("tokens", 0))


def metrics_text() -> str:
    return generate_latest(REGISTRY).decode("utf-8")


def serve(port: int = 8000) -> None:
    """Start the Prometheus ``/metrics`` HTTP endpoint."""
    start_http_server(port, registry=REGISTRY)
    print(f"[monitoring] serving /metrics on :{port}")


if __name__ == "__main__":  # pragma: no cover
    serve(int(os.environ.get("METRICS_PORT", "8000")))
    import time

    while True:
        time.sleep(3600)
