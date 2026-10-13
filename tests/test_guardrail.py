import pytest

from tracetopolicy.agent import check_select_only, classify_failure, extract_sql
from tracetopolicy.warehouse import QueryResult


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "select count(*) from customers",
        "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
        "SELECT o.order_id FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
    ],
)
def test_select_only_accepts_read_queries(sql):
    ok, reason = check_select_only(sql)
    assert ok, reason


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE customers",
        "DELETE FROM orders",
        "UPDATE orders SET total_amount = 0",
        "INSERT INTO orders VALUES (1)",
        "CREATE TABLE x (a INT)",
        "ALTER TABLE orders ADD COLUMN z INT",
        "ATTACH 'other.db'",
        "PRAGMA database_list",
        "SELECT 1; DROP TABLE customers",
        "SELECT * FROM orders; SELECT 1",
        "COPY orders TO 'x.csv'",
    ],
)
def test_select_only_rejects_writes(sql):
    ok, reason = check_select_only(sql)
    assert not ok
    assert reason


def test_select_only_rejects_empty():
    assert not check_select_only("   ")[0]


def test_extract_sql_strips_fences_and_preamble():
    text = "Sure! Here you go:\n```sql\nSELECT name FROM customers;\n```"
    assert extract_sql(text) == "SELECT name FROM customers"


def test_extract_sql_takes_first_statement():
    assert extract_sql("SELECT a FROM t; SELECT b FROM t") == "SELECT a FROM t"


def test_classify_failure_wrong_table():
    res = QueryResult(ok=False, error="Catalog Error: Table with name nope does not exist!")
    assert classify_failure("SELECT * FROM nope", res, "SELECT * FROM orders", False) == "wrong_table"


def test_classify_failure_guardrail_and_timeout():
    res = QueryResult(ok=False, error="blocked")
    assert classify_failure("", res, "SELECT 1", True) == "guardrail_block"
    res2 = QueryResult(ok=False, timed_out=True, error_type="timeout")
    assert classify_failure("SELECT 1", res2, "SELECT 1", False) == "timeout"


def test_classify_failure_empty_and_missing_join():
    empty = QueryResult(ok=True, rows=[])
    assert classify_failure("SELECT 1", empty, "SELECT 1", False) == "empty_result"
    wrong = QueryResult(ok=True, rows=[(1,)])
    gold = "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id"
    assert classify_failure("SELECT name FROM customers", wrong, gold, False) == "missing_join"
