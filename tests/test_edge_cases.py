from __future__ import annotations
from tests.conftest import *

def test_empty_table():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    r = e.execute("SELECT * FROM t;")
    assert r.row_count == 0

def test_empty_table_count():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    r = e.execute("SELECT COUNT(*) FROM t;")
    assert_value(r, 0)

def test_single_row():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (42);")
    r = e.execute("SELECT MIN(a), MAX(a), AVG(a), SUM(a) FROM t;")
    assert r.rows[0] == [42, 42, 42.0, 42]

def test_all_nulls():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (NULL),(NULL),(NULL);")
    r = e.execute("SELECT COUNT(*), COUNT(a), SUM(a), AVG(a) FROM t;")
    assert r.rows[0][0] == 3
    assert r.rows[0][1] == 0
    assert r.rows[0][2] is None

def test_large_string():
    e = make_engine()
    e.execute("CREATE TABLE t (s TEXT);")
    big = 'x' * 100000
    e.execute(f"INSERT INTO t VALUES ('{big}');")
    r = e.execute("SELECT LENGTH(s) FROM t;")
    assert_value(r, 100000)

def test_special_chars():
    e = make_engine()
    e.execute("CREATE TABLE t (s VARCHAR);")
    e.execute("INSERT INTO t VALUES ('it''s a test');")
    r = e.execute("SELECT s FROM t;")
    assert r.rows[0][0] == "it's a test"

def test_unicode():
    e = make_engine()
    e.execute("CREATE TABLE t (s VARCHAR);")
    e.execute("INSERT INTO t VALUES ('你好世界');")
    r = e.execute("SELECT s FROM t;")
    assert r.rows[0][0] == '你好世界'

def test_very_large_int():
    e = make_engine()
    e.execute("CREATE TABLE t (a BIGINT);")
    e.execute("INSERT INTO t VALUES (9223372036854775807);")  # max int64
    r = e.execute("SELECT a FROM t;")
    assert r.rows[0][0] == 9223372036854775807

def test_very_small_int():
    e = make_engine()
    e.execute("CREATE TABLE t (a BIGINT);")
    e.execute("INSERT INTO t VALUES (-9223372036854775808);")  # min int64
    r = e.execute("SELECT a FROM t;")
    assert r.rows[0][0] == -9223372036854775808

def test_float_precision():
    e = make_engine()
    e.execute("CREATE TABLE t (f DOUBLE);")
    e.execute("INSERT INTO t VALUES (3.141592653589793);")
    r = e.execute("SELECT f FROM t;")
    assert abs(r.rows[0][0] - 3.141592653589793) < 1e-15

def test_division_by_zero():
    e = make_engine()
    assert_error(lambda: e.execute("SELECT 1 / 0;"))

def test_modulo_by_zero():
    e = make_engine()
    assert_error(lambda: e.execute("SELECT 10 % 0;"))

def test_order_by_nulls_first():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (3),(NULL),(1),(NULL),(2);")
    r = e.execute("SELECT a FROM t ORDER BY a NULLS FIRST;")
    assert r.rows[0][0] is None
    assert r.rows[1][0] is None

def test_limit_zero():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3);")
    r = e.execute("SELECT * FROM t LIMIT 0;")
    assert r.row_count == 0

def test_offset_beyond_rows():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3);")
    r = e.execute("SELECT * FROM t OFFSET 100;")
    assert r.row_count == 0

def test_duplicate_column_names():
    """SELECT 中重复列名。"""
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1);")
    r = e.execute("SELECT a, a, a FROM t;")
    assert len(r.columns) == 3
    assert r.rows[0] == [1, 1, 1]
