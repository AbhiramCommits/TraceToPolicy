import duckdb
from tests.fakes import ScriptedGenerator

from tracetopolicy.agent import NL2SQLAgent
from tracetopolicy.warehouse import run_sql


def _conn():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE customers AS SELECT * FROM (VALUES (1,'a','US'),(2,'b','CA')) t(customer_id,name,country)"
    )
    return con


TASK = {
    "task_id": "t1",
    "question": "List all customer names.",
    "gold_sql": "SELECT name FROM customers",
    "category": "filter",
}


def test_agent_single_turn_correct():
    con = _conn()
    gen = ScriptedGenerator(["SELECT name FROM customers"])
    agent = NL2SQLAgent(gen, con, max_turns=3)
    gold = {"t1": run_sql(con, TASK["gold_sql"])}
    recs = agent.run_tasks([TASK], gold, samples_per_task=1, temperature=0.0, seed=0)
    assert len(recs) == 1
    assert recs[0].correct is True
    assert recs[0].failure_mode == "correct"
    assert recs[0].turn == 0


def test_agent_repair_loop():
    con = _conn()
    # first attempt references a missing table, second is correct
    gen = ScriptedGenerator(["SELECT * FROM nope", "SELECT name FROM customers"])
    agent = NL2SQLAgent(gen, con, max_turns=3)
    gold = {"t1": run_sql(con, TASK["gold_sql"])}
    recs = agent.run_tasks([TASK], gold, samples_per_task=1, temperature=0.0, seed=0)
    assert len(recs) == 2
    assert recs[0].correct is False
    assert recs[0].failure_mode == "wrong_table"
    assert recs[1].correct is True
    assert recs[1].turn == 1


def test_agent_guardrail_block_recorded():
    con = _conn()
    gen = ScriptedGenerator(["DROP TABLE customers"] * 3)
    agent = NL2SQLAgent(gen, con, max_turns=3)
    gold = {"t1": run_sql(con, TASK["gold_sql"])}
    recs = agent.run_tasks([TASK], gold, samples_per_task=1, temperature=0.0, seed=0)
    assert all(r.guardrail_blocked for r in recs)
    assert all(r.failure_mode == "guardrail_block" for r in recs)
    assert len(recs) == 3


def test_agent_harvest_writes_jsonl(tmp_path):
    con = _conn()
    gen = ScriptedGenerator(["SELECT name FROM customers"])
    agent = NL2SQLAgent(gen, con, max_turns=3)
    out = tmp_path / "traces.jsonl"
    summary = agent.harvest(
        [TASK],
        split="train",
        out_path=str(out),
        samples_per_task=2,
        temperature=0.0,
        seed=0,
        progress=False,
    )
    assert out.exists()
    assert summary["records"] == 2
    assert summary["task_solve_rate"] == 1.0
