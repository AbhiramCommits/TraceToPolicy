from tracetopolicy.warehouse import QueryResult, results_equal, run_sql


def test_results_equal_identical():
    a = [("US", 10), ("CA", 5)]
    assert results_equal(a, list(a))


def test_results_equal_row_order_insensitive():
    assert results_equal([("US", 10), ("CA", 5)], [("CA", 5), ("US", 10)])


def test_results_equal_column_order_and_alias():
    a = (["country", "n"], [("US", 10), ("CA", 5)])
    b = (["n", "country"], [(5, "CA"), (10, "US")])
    assert results_equal(a, b)


def test_results_equal_null_handling():
    assert results_equal([("US", None)], [("US", None)])
    assert not results_equal([("US", None)], [("US", 0)])
    # NaN normalized to None
    assert results_equal([("US", float("nan"))], [("US", None)])


def test_results_equal_float_tolerance():
    assert results_equal([(0.30000001,)], [(0.3,)])
    assert not results_equal([(0.3,)], [(0.4,)])


def test_results_equal_int_vs_float():
    assert results_equal([(5,)], [(5.0,)])


def test_results_equal_different_lengths():
    assert not results_equal([(1,), (2,)], [(1,)])
    assert not results_equal([], [(1,)])


def test_results_equal_query_result_object():
    a = QueryResult(ok=True, columns=["x"], rows=[(1,)])
    b = QueryResult(ok=True, columns=["x"], rows=[(1,)])
    assert results_equal(a, b)


def test_run_sql_ok_and_error():
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS x, 'a' AS y")
    r = run_sql(con, "SELECT x, y FROM t")
    assert r.ok and r.row_count == 1 and r.columns == ["x", "y"]
    bad = run_sql(con, "SELECT * FROM missing_table")
    assert not bad.ok and bad.error_type
