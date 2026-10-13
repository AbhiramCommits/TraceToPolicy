"""Render a static, self-contained HTML dashboard from the registry + eval history."""

from __future__ import annotations

import json
import os
import time

RESULTS = "results"
REGISTRY = "artifacts/registry.json"


def _load(path: str, default=None):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return default
    return default


def _bar(value: float, width: int = 220, color: str = "#3b82f6") -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (
        f'<div style="background:#e5e7eb;border-radius:6px;height:14px;width:{width}px;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{pct:.1f}%;height:14px;border-radius:6px"></div></div>'
    )


def render_dashboard(out_path: str = "results/dashboard.html") -> str:
    reg = _load(REGISTRY, {"current": "n/a", "versions": []})
    report = _load(os.path.join(RESULTS, "eval_report.json"), {})
    loop = _load(os.path.join(RESULTS, "loop_history.json"), {})
    datasets = _load(os.path.join(RESULTS, "dataset_stats.json"), {})
    drift = _load(os.path.join(RESULTS, "drift.json"), {})

    versions = reg.get("versions", [])
    current = reg.get("current", "n/a")
    cur_entry = next((v for v in versions if v.get("version") == current), {})
    dpo_metrics = report.get("dpo", {}).get("metrics", {})
    win_rate = report.get("dpo", {}).get("win_rate_vs_baseline", 0.0) or 0.0
    tool_acc = dpo_metrics.get("tool_call_accuracy", 0.0)
    turns = dpo_metrics.get("turns_to_success", 0.0)
    drift_score = drift.get("max_psi", cur_entry.get("drift", {}).get("max_psi", 0.0))

    rows_variants = ""
    for v in ["baseline", "sft", "dpo"]:
        m = report.get(v, {}).get("metrics", {})
        if not m:
            continue
        rows_variants += (
            f"<tr><td>{v}</td><td>{m.get('accuracy_turn1', 0):.3f}</td>"
            f"<td>{m.get('accuracy_final', 0):.3f} {_bar(m.get('accuracy_final', 0))}</td>"
            f"<td>{m.get('turns_to_success', 0):.2f}</td>"
            f"<td>{m.get('tool_call_accuracy', 0):.3f}</td>"
            f"<td>{m.get('mean_tokens_per_task', 0):.0f}</td>"
            f"<td>{m.get('mean_latency_ms_per_task', 0):.0f}</td></tr>"
        )

    rows_registry = ""
    for v in versions:
        reason = "; ".join(v.get("reasons", [])[:2])
        rows_registry += (
            f"<tr><td>{v.get('version')}</td><td>{v.get('parent')}</td>"
            f"<td>{v.get('decision')}</td><td>{v.get('created', '')}</td>"
            f"<td>{reason}</td></tr>"
        )

    rows_loop = ""
    for c in loop.get("cycles", []):
        gd = c.get("gate_details", {})
        rows_loop += (
            f"<tr><td>{c.get('version')}</td><td>{c.get('decision')}</td>"
            f"<td>{gd.get('prev_accuracy_final', 0):.3f} &rarr; {gd.get('new_accuracy_final', 0):.3f}</td>"
            f"<td>{gd.get('cost_rise_pct', 0):+.1f}%</td>"
            f"<td>{c.get('drift', {}).get('max_psi', 0):.3f}</td>"
            f"<td>{'; '.join(c.get('reasons', []))}</td></tr>"
        )

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TraceToPolicy dashboard</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color:#111827; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
 table {{ border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
 th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }}
 th {{ background:#f9fafb; }}
 .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:12px; }}
 .card {{ border:1px solid #e5e7eb; border-radius:10px; padding:12px 16px; min-width:170px; }}
 .card .v {{ font-size: 24px; font-weight:600; }}
 .muted {{ color:#6b7280; font-size:12px; }}
</style></head><body>
<h1>TraceToPolicy &mdash; NL-to-SQL continuous learning</h1>
<p class="muted">generated {generated} &middot; current promoted version: <b>{current}</b>
 &middot; adapter: {cur_entry.get('promoted_adapter') or 'base model'}</p>
<div class="cards">
  <div class="card"><div class="muted">DPO win rate vs baseline</div><div class="v">{win_rate:.3f}</div></div>
  <div class="card"><div class="muted">tool-call accuracy (DPO)</div><div class="v">{tool_acc:.3f}</div></div>
  <div class="card"><div class="muted">turns-to-success (DPO)</div><div class="v">{turns:.2f}</div></div>
  <div class="card"><div class="muted">train/live drift (max PSI)</div><div class="v">{drift_score:.3f}</div></div>
</div>

<h2>Held-out evaluation</h2>
<table><tr><th>variant</th><th>acc@1</th><th>final acc</th><th>turns-to-success</th>
<th>tool-call acc</th><th>tokens/task</th><th>ms/task</th></tr>{rows_variants}</table>

<h2>Model registry</h2>
<table><tr><th>version</th><th>parent</th><th>decision</th><th>created</th><th>reason</th></tr>{rows_registry}</table>

<h2>Continuous-learning cycles</h2>
<table><tr><th>version</th><th>decision</th><th>accuracy (inc&rarr;cand)</th><th>cost delta</th><th>drift PSI</th><th>reasons</th></tr>{rows_loop}</table>

<h2>Datasets</h2>
<p class="muted">traces={datasets.get('traces', 'n/a')} &middot; SFT examples={datasets.get('sft_examples', 'n/a')}
 &middot; DPO pairs={datasets.get('dpo_pairs', 'n/a')} &middot; repair-derived={datasets.get('repair_derived_examples', 'n/a')}</p>
</body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


if __name__ == "__main__":  # pragma: no cover
    print(render_dashboard())
