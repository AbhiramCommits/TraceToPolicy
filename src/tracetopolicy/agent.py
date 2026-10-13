"""The trace-emitting NL-to-SQL agent.

A multi-turn loop: propose SQL -> execute against DuckDB -> on a guardrail
block, execution error, empty result, or wrong result, feed feedback back and
repair, up to ``max_turns`` attempts. Every attempt (not just the final one) is
appended as a JSONL trace record carrying the prompt, SQL, execution status,
error text, correctness versus the gold query, latency, token counts and a
``failure_mode`` from a fixed taxonomy.

Harvesting is batched by *round*: all still-active samples across all tasks are
generated together for a turn, which keeps the small CPU model busy.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import sqlglot
from sqlglot import exp

from .inference import Generator
from .warehouse import QueryResult, results_equal, run_sql

FAILURE_MODES = [
    "correct",
    "syntax_error",
    "wrong_table",
    "missing_join",
    "bad_aggregation",
    "wrong_filter",
    "empty_result",
    "timeout",
    "guardrail_block",
]

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|"
    r"truncate|grant|revoke|pragma|vacuum|use|call|merge|replace)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are an expert analytics engineer writing DuckDB SQL. "
    "Reply with exactly one read-only SELECT query. "
    "Output ONLY the SQL — no explanation, no markdown, no backticks. "
    "Use the real table and column names from the schema. Never modify data."
)


def schema_text(conn) -> str:
    """Compact schema string for prompting."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    lines = []
    for (t,) in tables:
        cols = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{t}' ORDER BY ordinal_position"
        ).fetchall()
        lines.append(f"{t}(" + ", ".join(f"{c} {d}" for c, d in cols) + ")")
    return "\n".join(lines)


def check_select_only(sql: str) -> tuple[bool, str | None]:
    """Return ``(allowed, reason)``. Only single read-only SELECT statements pass."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, "empty SQL"
    # crude multi-statement check (ignores ';' inside string literals for safety)
    parts = [p for p in re.split(r";\s*", re.sub(r"'[^']*'", "''", stripped)) if p.strip()]
    if len(parts) > 1:
        return False, "multiple statements are not allowed"
    lowered = stripped.lower().lstrip()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "only SELECT / WITH queries are allowed"
    if _FORBIDDEN.search(re.sub(r"'[^']*'", "''", lowered)):
        return False, "write / DDL keywords are not allowed"
    try:
        parsed = sqlglot.parse(stripped, read="duckdb")
    except Exception as exc:  # noqa: BLE001
        return False, f"could not parse SQL as DuckDB: {exc}"
    if len(parsed) != 1:
        return False, "multiple statements are not allowed"
    root = parsed[0]
    if root is None:
        return False, "empty statement"
    allowed_types = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)
    if not isinstance(root, allowed_types):
        return False, f"statement type {type(root).__name__} is not allowed"
    return True, None


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model completion."""
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()
    # drop common preambles
    for start in re.finditer(r"\b(select|with)\b", text, re.IGNORECASE):
        text = text[start.start() :]
        break
    text = text.split(";")[0].strip()
    return text


def _has_clause(sql: str, kind: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return False
    if kind == "join":
        return any(isinstance(n, exp.Join) for n in tree.walk()) or "join" in sql.lower()
    if kind == "aggregation":
        return (
            tree.find(exp.Group) is not None
            or tree.find(exp.AggFunc) is not None
            or any(isinstance(n, exp.AggFunc) for n in tree.walk())
        )
    return False


def classify_failure(
    sql: str,
    result: QueryResult,
    gold_sql: str,
    guardrail_blocked: bool,
) -> str:
    """Assign one of :data:`FAILURE_MODES` to an attempt."""
    if guardrail_blocked:
        return "guardrail_block"
    if result.timed_out or result.error_type == "timeout":
        return "timeout"
    if not result.ok:
        err = (result.error or "").lower()
        if "table with name" in err or "does not exist" in err:
            return "wrong_table"
        if "parser" in err or "syntax" in err or "unexpected" in err:
            return "syntax_error"
        if "column" in err or "binder" in err:
            return "wrong_filter"
        return "syntax_error"
    if result.row_count == 0:
        return "empty_result"
    if _has_clause(gold_sql, "join") and not _has_clause(sql, "join"):
        return "missing_join"
    if _has_clause(gold_sql, "aggregation") and not _has_clause(sql, "aggregation"):
        return "bad_aggregation"
    if len(sql) < 0.5 * len(gold_sql):
        return "wrong_filter"
    return "wrong_filter"


@dataclass
class TraceRecord:
    task_id: str
    category: str
    split: str
    sample_index: int
    turn: int
    model: str
    temperature: float
    initial_prompt: list[dict]
    prompt: list[dict]
    raw_completion: str
    generated_sql: str
    execution_status: str
    error_text: str | None
    row_count: int
    correct: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    failure_mode: str
    guardrail_blocked: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return asdict(self)


class NL2SQLAgent:
    """Runs the propose/execute/repair loop and emits trace records."""

    def __init__(
        self,
        generator: Generator,
        conn,
        max_turns: int = 3,
        timeout_s: float = 10.0,
        model_name: str | None = None,
        max_new_tokens: int = 192,
    ):
        self.generator = generator
        self.conn = conn
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.model_name = model_name or getattr(generator, "name", "model")
        self.max_new_tokens = max_new_tokens
        self.schema = schema_text(conn)

    # -- prompt construction -------------------------------------------------
    def _initial_messages(self, question: str) -> list[dict]:
        user = (
            f"Database schema (DuckDB):\n{self.schema}\n\n"
            f"Question: {question}\n\n"
            "Write a single read-only SELECT query that answers the question."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _feedback(sql: str, status: str, error: str | None, rows: int) -> str:
        if status == "guardrail_block":
            return (
                f"The query was rejected by the read-only guardrail: {error}. "
                "Return a single SELECT statement only."
            )
        if status == "timeout":
            return "The query timed out. Rewrite it to be more efficient."
        if status == "error":
            return f"The query failed with this DuckDB error:\n{error}\nFix it and return a new query."
        if rows == 0:
            return "The query ran but returned no rows. Reconsider joins, filters and date ranges."
        return (
            "The query ran but its result did not match the expected answer. "
            "Reconsider the joins, filters, groupings and aggregation."
        )

    # -- core loop -----------------------------------------------------------
    def run_tasks(
        self,
        tasks: list[dict],
        gold_lookup: dict[str, QueryResult],
        samples_per_task: int = 1,
        temperature: float = 0.0,
        seed: int = 0,
        split: str = "train",
        batch_size: int = 8,
    ) -> list[TraceRecord]:
        """Run every (task, sample) pair through the multi-turn loop."""
        # state per active sample
        states = []
        for task in tasks:
            gold = gold_lookup[task["task_id"]]
            for s in range(samples_per_task):
                states.append(
                    {
                        "task": task,
                        "gold": gold,
                        "sample": s,
                        "turn": 0,
                        "messages": self._initial_messages(task["question"]),
                        "done": False,
                        "records": [],
                    }
                )

        records: list[TraceRecord] = []
        for turn in range(self.max_turns):
            active = [st for st in states if not st["done"]]
            if not active:
                break
            convs = [st["messages"] for st in active]
            gens = self.generator.chat(
                convs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                seed=seed + turn * 997,
            )
            for st, gen in zip(active, gens, strict=True):
                task = st["task"]
                sql = extract_sql(gen.text or "")
                allowed, reason = check_select_only(sql) if sql else (False, "no SQL produced")
                if sql and allowed:
                    result = run_sql(self.conn, sql, timeout_s=self.timeout_s)
                    status = (
                        "timeout"
                        if result.timed_out
                        else ("ok" if result.ok else "error")
                    )
                    correct = result.ok and results_equal(result, st["gold"])
                    failure_mode = classify_failure(
                        sql, result, task["gold_sql"], guardrail_blocked=False
                    )
                    if correct:
                        failure_mode = "correct"
                    row_count = result.row_count if result.ok else 0
                    error_text = result.error
                else:
                    result = QueryResult(ok=False, error=reason, error_type="guardrail")
                    status = "guardrail_block"
                    correct = False
                    failure_mode = "guardrail_block"
                    row_count = 0
                    error_text = reason

                rec = TraceRecord(
                    task_id=task["task_id"],
                    category=task.get("category", "unknown"),
                    split=split,
                    sample_index=st["sample"],
                    turn=turn,
                    model=self.model_name,
                    temperature=temperature,
                    initial_prompt=st["messages"][:2],
                    prompt=list(st["messages"]),
                    raw_completion=gen.text or "",
                    generated_sql=sql,
                    execution_status=status,
                    error_text=error_text,
                    row_count=row_count,
                    correct=correct,
                    latency_ms=result.latency_ms,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    failure_mode=failure_mode,
                    guardrail_blocked=(status == "guardrail_block"),
                )
                records.append(rec)
                st["records"].append(rec)
                st["turn"] = turn + 1

                if correct or turn == self.max_turns - 1:
                    st["done"] = True
                else:
                    st["messages"] = st["messages"] + [
                        {"role": "assistant", "content": sql or gen.text or ""},
                        {
                            "role": "user",
                            "content": self._feedback(sql, status, error_text, row_count),
                        },
                    ]
        return records

    def harvest(
        self,
        tasks: list[dict],
        split: str,
        out_path: str,
        samples_per_task: int = 4,
        temperature: float = 0.8,
        seed: int = 0,
        batch_size: int = 8,
        chunk_tasks: int = 16,
        progress: bool = True,
        resume: bool = False,
    ) -> dict:
        """Run the loop and write trace records to ``out_path`` (JSONL).

        Tasks are processed in chunks and every record is flushed to disk as it
        is produced. With ``resume=True`` a task already present in ``out_path``
        is skipped and new records are appended, so a long harvest never loses
        completed work.
        """
        gold_lookup = {t["task_id"]: run_sql(self.conn, t["gold_sql"], self.timeout_s) for t in tasks}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        mode = "w"
        all_records: list[dict] = []
        if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path) as fh:
                existing = [json.loads(line) for line in fh if line.strip()]
            done_ids = {r["task_id"] for r in existing}
            all_records.extend(existing)
            tasks = [t for t in tasks if t["task_id"] not in done_ids]
            mode = "a"
            if progress:
                print(f"[harvest] resuming: {len(done_ids)} tasks / {len(all_records)} records already done", flush=True)
        with open(out_path, mode) as fh:
            for start in range(0, len(tasks), chunk_tasks):
                chunk = tasks[start : start + chunk_tasks]
                recs = self.run_tasks(
                    chunk,
                    gold_lookup,
                    samples_per_task=samples_per_task,
                    temperature=temperature,
                    seed=seed,
                    split=split,
                    batch_size=batch_size,
                )
                for rec in recs:
                    fh.write(json.dumps(rec.to_json()) + "\n")
                fh.flush()
                all_records.extend(r.to_json() for r in recs)
                if progress:
                    print(
                        f"[harvest] {len(all_records)} records | "
                        f"{min(start + chunk_tasks, len(tasks))}/{len(tasks)} tasks",
                        flush=True,
                    )
        return summarize_traces(all_records)


def summarize_traces(records: list[dict]) -> dict:
    """Histogram of failure modes and headline counts."""
    hist: dict[str, int] = {m: 0 for m in FAILURE_MODES}
    tasks = set()
    correct_tasks = set()
    turns = 0
    for r in records:
        hist[r["failure_mode"]] = hist.get(r["failure_mode"], 0) + 1
        tasks.add(r["task_id"])
        turns += 1
        if r.get("correct"):
            correct_tasks.add(r["task_id"])
    return {
        "records": len(records),
        "tasks": len(tasks),
        "task_solve_rate": (len(correct_tasks) / len(tasks)) if tasks else 0.0,
        "failure_mode_histogram": hist,
    }


def load_traces(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":  # pragma: no cover
    from .warehouse import build_warehouse, connect

    build_warehouse()
    con = connect()
    print(schema_text(con))
