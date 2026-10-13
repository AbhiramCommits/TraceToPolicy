"""DuckDB warehouse: a deterministic 6-table analytics schema plus the SQL
execution utilities used as the reward signal for the agent.

Tables (≈50k rows, fixed seed):

* ``customers``      — 2,000 rows
* ``products``       — 300 rows
* ``orders``         — 12,000 rows
* ``order_items``    — ~30,000 rows
* ``events``         — 8,000 rows
* ``subscriptions``  — 2,000 rows

``run_sql`` executes a statement with a wall-clock timeout and returns a
normalized :class:`QueryResult`. ``results_equal`` performs the *differential
execution* comparison that turns "did the generated SQL return the same rows as
the gold SQL?" into a reward.
"""

from __future__ import annotations

import datetime as dt
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np
import pandas as pd

SEED = 42

_TABLES = ["customers", "products", "orders", "order_items", "events", "subscriptions"]


@dataclass
class QueryResult:
    """Normalized outcome of a single SQL execution."""

    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    latency_ms: float = 0.0
    timed_out: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def _random_dates(rng: np.random.Generator, n: int, start: dt.date, span_days: int) -> list[dt.date]:
    offsets = rng.integers(0, span_days, size=n)
    return [start + dt.timedelta(days=int(o)) for o in offsets]


def _generate_frames(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    n_customers, n_products, n_orders = 2000, 300, 12000

    countries = ["US", "CA", "GB", "DE", "FR", "IN", "BR", "AU", "JP", "NL"]
    segments = ["consumer", "smb", "enterprise"]
    channels = ["web", "mobile_app", "partner", "sales_rep"]
    statuses = ["completed", "completed", "completed", "pending", "cancelled", "refunded"]
    categories = ["electronics", "home", "apparel", "grocery", "sports", "toys", "beauty"]
    brands = ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli", "Stark"]
    devices = ["ios", "android", "desktop", "tablet"]
    event_types = ["page_view", "add_to_cart", "checkout_start", "purchase", "search", "login"]
    plans = ["basic", "pro", "team", "enterprise"]
    sub_status = ["active", "active", "churned", "trialing"]

    customer_id = np.arange(1, n_customers + 1)
    customers = pd.DataFrame(
        {
            "customer_id": customer_id,
            "name": [f"Customer_{i:04d}" for i in customer_id],
            "email": [f"user{i:04d}@example.com" for i in customer_id],
            "country": rng.choice(countries, n_customers, p=[0.4, 0.1, 0.12, 0.08, 0.06, 0.1, 0.05, 0.04, 0.03, 0.02]),
            "signup_date": _random_dates(rng, n_customers, dt.date(2022, 1, 1), 900),
            "segment": rng.choice(segments, n_customers, p=[0.6, 0.3, 0.1]),
        }
    )

    product_id = np.arange(1, n_products + 1)
    products = pd.DataFrame(
        {
            "product_id": product_id,
            "name": [f"Product_{i:03d}" for i in product_id],
            "category": rng.choice(categories, n_products, p=[0.25, 0.18, 0.15, 0.12, 0.1, 0.1, 0.1]),
            "price": np.round(rng.uniform(5, 500, n_products), 2),
            "cost": np.round(rng.uniform(2, 300, n_products), 2),
            "brand": rng.choice(brands, n_products),
        }
    )

    order_id = np.arange(1, n_orders + 1)
    order_customer = rng.integers(1, n_customers + 1, n_orders)
    orders = pd.DataFrame(
        {
            "order_id": order_id,
            "customer_id": order_customer,
            "order_date": _random_dates(rng, n_orders, dt.date(2023, 1, 1), 720),
            "status": rng.choice(statuses, n_orders),
            "channel": rng.choice(channels, n_orders, p=[0.5, 0.3, 0.12, 0.08]),
        }
    )

    # order_items: 1-4 line items per order
    items_per_order = rng.integers(1, 5, n_orders)
    total_items = int(items_per_order.sum())
    order_ids_expanded = np.repeat(order_id, items_per_order)
    item_product = rng.integers(1, n_products + 1, total_items)
    price_lookup = products.set_index("product_id")["price"]
    unit_price = price_lookup.reindex(item_product).to_numpy()
    discount = np.round(rng.choice([0.0, 0.05, 0.1, 0.15, 0.2], total_items, p=[0.5, 0.15, 0.15, 0.1, 0.1]), 2)
    quantity = rng.integers(1, 6, total_items)
    order_items = pd.DataFrame(
        {
            "item_id": np.arange(1, total_items + 1),
            "order_id": order_ids_expanded,
            "product_id": item_product,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
        }
    )

    # orders.total_amount derived from line items so joins are consistent
    line_total = order_items["quantity"] * order_items["unit_price"] * (1 - order_items["discount"])
    totals = line_total.groupby(order_items["order_id"]).sum().round(2)
    orders["total_amount"] = orders["order_id"].map(totals).fillna(0.0).round(2)
    # cancelled / refunded orders get zeroed usage for realism
    mask = orders["status"].isin(["cancelled", "refunded"])
    orders.loc[mask, "total_amount"] = 0.0

    n_events = 8000
    events = pd.DataFrame(
        {
            "event_id": np.arange(1, n_events + 1),
            "customer_id": rng.integers(1, n_customers + 1, n_events),
            "event_type": rng.choice(event_types, n_events, p=[0.4, 0.18, 0.1, 0.12, 0.12, 0.08]),
            "event_date": _random_dates(rng, n_events, dt.date(2023, 6, 1), 400),
            "session_id": [f"sess_{i:06d}" for i in rng.integers(1, 40000, n_events)],
            "device": rng.choice(devices, n_events, p=[0.35, 0.35, 0.2, 0.1]),
        }
    )

    subs_customer = rng.choice(customer_id, 2000, replace=False)
    plan_arr = rng.choice(plans, 2000, p=[0.4, 0.3, 0.2, 0.1])
    mrr_map = {"basic": 29.0, "pro": 99.0, "team": 299.0, "enterprise": 999.0}
    start = _random_dates(rng, 2000, dt.date(2022, 6, 1), 700)
    status_arr = rng.choice(sub_status, 2000, p=[0.55, 0.1, 0.25, 0.1])
    end = [
        (s + dt.timedelta(days=int(rng.integers(30, 500)))) if st == "churned" else None
        for s, st in zip(start, status_arr, strict=True)
    ]
    subscriptions = pd.DataFrame(
        {
            "subscription_id": np.arange(1, 2001),
            "customer_id": subs_customer,
            "plan": plan_arr,
            "start_date": start,
            "end_date": end,
            "mrr": [mrr_map[p] for p in plan_arr],
            "status": status_arr,
        }
    )

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "events": events,
        "subscriptions": subscriptions,
    }


def build_warehouse(path: str = "data/warehouse.duckdb", seed: int = SEED) -> str:
    """Create the DuckDB warehouse file at ``path`` and seed it. Idempotent."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    frames = _generate_frames(seed)

    con = duckdb.connect(path)
    try:
        con.execute("DROP TABLE IF EXISTS subscriptions")
        con.execute("DROP TABLE IF EXISTS events")
        con.execute("DROP TABLE IF EXISTS order_items")
        con.execute("DROP TABLE IF EXISTS orders")
        con.execute("DROP TABLE IF EXISTS products")
        con.execute("DROP TABLE IF EXISTS customers")

        con.register("df_customers", frames["customers"])
        con.register("df_products", frames["products"])
        con.register("df_orders", frames["orders"])
        con.register("df_order_items", frames["order_items"])
        con.register("df_events", frames["events"])
        con.register("df_subscriptions", frames["subscriptions"])

        con.execute(
            "CREATE TABLE customers AS SELECT * FROM df_customers"
        )
        con.execute("ALTER TABLE customers ADD PRIMARY KEY (customer_id)")
        con.execute("CREATE TABLE products AS SELECT * FROM df_products")
        con.execute("ALTER TABLE products ADD PRIMARY KEY (product_id)")
        con.execute("CREATE TABLE orders AS SELECT * FROM df_orders")
        con.execute("ALTER TABLE orders ADD PRIMARY KEY (order_id)")
        con.execute("CREATE TABLE order_items AS SELECT * FROM df_order_items")
        con.execute("ALTER TABLE order_items ADD PRIMARY KEY (item_id)")
        con.execute("CREATE TABLE events AS SELECT * FROM df_events")
        con.execute("ALTER TABLE events ADD PRIMARY KEY (event_id)")
        con.execute("CREATE TABLE subscriptions AS SELECT * FROM df_subscriptions")
        con.execute("ALTER TABLE subscriptions ADD PRIMARY KEY (subscription_id)")
    finally:
        con.close()
    return path


def connect(path: str = "data/warehouse.duckdb") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    return con


def warehouse_summary(path: str = "data/warehouse.duckdb") -> dict:
    con = duckdb.connect(path, read_only=True)
    try:
        counts = {}
        for t in _TABLES:
            counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        counts["total"] = sum(counts.values())
    finally:
        con.close()
    return counts


# --------------------------------------------------------------------------- #
# SQL execution + differential comparison
# --------------------------------------------------------------------------- #
def _to_python(value: Any) -> Any:
    """Normalize a DuckDB cell to a comparable Python value (type tolerant)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return round(float(value), 6)
    if isinstance(value, (dt.datetime,)):
        return value.isoformat()
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _norm_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if math.isnan(f) else round(f, 6)
    return value


def run_sql(conn: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = 10.0) -> QueryResult:
    """Execute ``sql`` and return a normalized :class:`QueryResult`.

    A watchdog thread calls ``conn.interrupt()`` if the query exceeds
    ``timeout_s`` seconds. The connection is shared with the caller, so callers
    that need isolation should pass a dedicated connection.
    """
    start = time.perf_counter()
    timed_out = {"v": False}

    def _interrupt() -> None:
        timed_out["v"] = True
        try:
            conn.interrupt()
        except Exception:
            pass

    timer = threading.Timer(timeout_s, _interrupt)
    timer.daemon = True
    timer.start()
    try:
        cur = conn.execute(sql)
        raw = cur.fetchall()
        columns = [d[0] for d in (cur.description or [])]
        rows = [tuple(_to_python(v) for v in row) for row in raw]
        latency = (time.perf_counter() - start) * 1000.0
        if timed_out["v"]:
            return QueryResult(
                ok=False,
                columns=columns,
                rows=rows,
                error=f"query exceeded {timeout_s}s timeout",
                error_type="timeout",
                latency_ms=latency,
                timed_out=True,
            )
        return QueryResult(ok=True, columns=columns, rows=rows, latency_ms=latency)
    except Exception as exc:  # noqa: BLE001 - we classify every failure mode
        latency = (time.perf_counter() - start) * 1000.0
        etype = type(exc).__name__
        if timed_out["v"] or "interrupt" in str(exc).lower():
            etype = "timeout"
        return QueryResult(
            ok=False,
            error=str(exc),
            error_type=etype,
            latency_ms=latency,
            timed_out=timed_out["v"],
        )
    finally:
        timer.cancel()


def _split(obj: Any) -> tuple[list[str], list[tuple]]:
    """Accept a QueryResult, ``(columns, rows)``, a dict, or bare rows."""
    if isinstance(obj, QueryResult):
        return list(obj.columns), [tuple(r) for r in obj.rows]
    if isinstance(obj, dict):
        return list(obj.get("columns", [])), [tuple(r) for r in obj.get("rows", [])]
    if (
        isinstance(obj, (tuple, list))
        and len(obj) == 2
        and isinstance(obj[0], (list, tuple))
        and (len(obj[0]) == 0 or isinstance(obj[0][0], str))
        and isinstance(obj[1], (list, tuple))
        and (len(obj[1]) == 0 or isinstance(obj[1][0], (tuple, list)))
    ):
        return list(obj[0]), [tuple(r) for r in obj[1]]
    return [], [tuple(r) for r in obj]


def _row_counter(rows: list[tuple]) -> dict:
    counter: dict[tuple, int] = {}
    for row in rows:
        key = tuple(_norm_number(v) for v in row)
        counter[key] = counter.get(key, 0) + 1
    return counter


def results_equal(a: Any, b: Any) -> bool:
    """Order-insensitive, type-tolerant comparison of two result sets.

    * Column *order* and column *names* do not matter (positional fallback plus
      a by-name alignment pass).
    * Row order does not matter (multiset comparison).
    * ``NULL``/NaN compare equal; ints and floats compare equal after rounding
      to 6 decimal places.
    """
    ca, ra = _split(a)
    cb, rb = _split(b)
    if len(ra) != len(rb):
        return False
    if not ra and not rb:
        return True
    if len(ra[0]) != len(rb[0]):
        return False

    if _row_counter(ra) == _row_counter(rb):
        return True

    # Try aligning by column name (handles differing column order / aliases).
    if ca and cb and len(ca) == len(cb) and set(ca) == set(cb):
        order = [cb.index(c) for c in ca]
        rb_aligned = [tuple(_norm_number(row[i]) for i in order) for row in rb]
        if _row_counter(ra) == _row_counter(rb_aligned):
            return True
    return False


if __name__ == "__main__":  # pragma: no cover
    p = build_warehouse()
    print(f"built {p}: {warehouse_summary(p)}")
