"""Train/live distribution drift detection.

Compares the distribution of the *training* traces against freshly harvested
*live* traces over four features: task type (category), failure mode, generated
SQL length, and total token count. Categorical features use the Population
Stability Index (PSI); numeric features are bucketed on the reference quantiles
and scored with both PSI and KL divergence. A feature whose PSI exceeds
``threshold`` raises the drift flag.
"""

from __future__ import annotations

import json
import math

import numpy as np

CATEGORICAL_FEATURES = ["category", "failure_mode"]
NUMERIC_FEATURES = ["sql_length", "token_count"]
DEFAULT_THRESHOLD = 0.2
_EPS = 1e-6


def _proportions(counts, eps: float = _EPS):
    total = float(sum(counts)) or 1.0
    return [(c / total) + eps for c in counts]


def psi(reference_counts, actual_counts, eps: float = _EPS) -> float:
    """Population Stability Index between two count vectors."""
    e = _proportions(reference_counts, eps)
    a = _proportions(actual_counts, eps)
    return float(sum((ai - ei) * math.log(ai / ei) for ai, ei in zip(a, e, strict=True)))


def kl_divergence(p_counts, q_counts, eps: float = _EPS) -> float:
    """KL(p || q) between two count vectors (nats)."""
    p = _proportions(p_counts, eps)
    q = _proportions(q_counts, eps)
    return float(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True)))


def _categorical_counts(records, feature):
    values = [str(r.get(feature, "unknown")) for r in records]
    return values


def _numeric_values(records, feature):
    if feature == "sql_length":
        return [len(r.get("generated_sql") or "") for r in records]
    if feature == "token_count":
        return [
            int(r.get("prompt_tokens", 0)) + int(r.get("completion_tokens", 0)) for r in records
        ]
    return [0 for _ in records]


def _bucket_edges(ref_values, n_bins: int = 10):
    if not ref_values:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(ref_values, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([edges[0] - 1, edges[0] + 1])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _hist(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    return counts.tolist()


def compute_drift(
    reference: list[dict], current: list[dict], threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """PSI/KL per feature plus an overall drift decision."""
    result: dict = {
        "threshold": threshold,
        "n_reference": len(reference),
        "n_current": len(current),
        "features": {},
    }

    for feat in CATEGORICAL_FEATURES:
        ref_vals = _categorical_counts(reference, feat)
        cur_vals = _categorical_counts(current, feat)
        cats = sorted(set(ref_vals) | set(cur_vals))
        ref_counts = [ref_vals.count(c) for c in cats]
        cur_counts = [cur_vals.count(c) for c in cats]
        p = psi(ref_counts, cur_counts)
        result["features"][feat] = {
            "type": "categorical",
            "psi": p,
            "kl": kl_divergence(ref_counts, cur_counts),
            "drift": p > threshold,
            "categories": cats,
            "reference_counts": ref_counts,
            "current_counts": cur_counts,
        }

    for feat in NUMERIC_FEATURES:
        ref_vals = _numeric_values(reference, feat)
        cur_vals = _numeric_values(current, feat)
        edges = _bucket_edges(ref_vals)
        p = psi(_hist(ref_vals, edges), _hist(cur_vals, edges))
        result["features"][feat] = {
            "type": "numeric",
            "psi": p,
            "kl": kl_divergence(_hist(ref_vals, edges), _hist(cur_vals, edges)),
            "drift": p > threshold,
            "edges": [float(e) if math.isfinite(e) else str(e) for e in edges],
        }

    result["max_psi"] = max((f["psi"] for f in result["features"].values()), default=0.0)
    result["drift_detected"] = any(f["drift"] for f in result["features"].values())
    return result


def load_traces(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":  # pragma: no cover
    import sys

    ref = load_traces(sys.argv[1])
    cur = load_traces(sys.argv[2])
    print(json.dumps(compute_drift(ref, cur), indent=2))
