from __future__ import annotations
from tests.conftest import *

def test_null_equality():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(NULL),(3);")
    r = e.execute("SELECT * FROM t WHERE a = NULL;")
    assert r.row_count == 0  # NULL = NULL → NULL (不匹配)

def test_null_is_null():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(NULL),(3);")
    r = e.execute("SELECT * FROM t WHERE a IS NULL;")
    assert r.row_count == 1

def test_null_is_not_null():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(NULL),(3);")
    r = e.execute("SELECT * FROM t WHERE a IS NOT NULL;")
    assert r.row_count == 2

def test_null_arithmetic():
    e = make_engine()
    r = e.execute("SELECT 1 + NULL, NULL * 5, NULL / 2;")
    assert r.rows[0][0] is None
    assert r.rows[0][1] is None
    assert r.rows[0][2] is None

def test_null_concat():
    e = make_engine()
    r = e.execute("SELECT 'a' || NULL, NULL || 'b';")
    assert r.rows[0][0] is None or r.rows[0][0] == ''
    assert r.rows[0][1] is None or r.rows[0][1] == ''

def test_null_and_true():
    e = make_engine()
    r = e.execute("SELECT NULL AND TRUE;")
    assert r.rows[0][0] is None

def test_null_and_false():
    e = make_engine()
    r = e.execute("SELECT NULL AND FALSE;")
    assert r.rows[0][0] == False

def test_null_or_true():
    e = make_engine()
    r = e.execute("SELECT NULL OR TRUE;")
    assert r.rows[0][0] == True

def test_null_or_false():
    e = make_engine()
    r = e.execute("SELECT NULL OR FALSE;")
    assert r.rows[0][0] is None

def test_null_in_aggregate():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(NULL),(3),(NULL),(5);")
    r = e.execute("SELECT COUNT(*), COUNT(a), SUM(a), AVG(a) FROM t;")
    assert r.rows[0][0] == 5  # COUNT(*)
    assert r.rows[0][1] == 3  # COUNT(a) 忽略 NULL
    assert r.rows[0][2] == 9  # SUM(1+3+5)
    assert abs(r.rows[0][3] - 3.0) < 0.01  # AVG(9/3)

def test_null_group_by():
    e = make_engine()
    e.execute("CREATE TABLE t (g INT, v INT);")
    e.execute("INSERT INTO t VALUES (1,10),(1,20),(NULL,30),(NULL,40);")
    r = e.execute("SELECT g, SUM(v) FROM t GROUP BY g;")
    assert r.row_count == 2  # 1 和 NULL 分别分组

def test_null_order_by():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (3),(NULL),(1),(NULL),(2);")
    r = e.execute("SELECT a FROM t ORDER BY a NULLS LAST;")
    assert r.rows[0][0] == 1
    assert r.rows[-2][0] is None

def test_coalesce_null():
    e = make_engine()
    r = e.execute("SELECT COALESCE(NULL, NULL, 42, 99);")
    assert_value(r, 42)

def test_nullif_null():
    e = make_engine()
    r = e.execute("SELECT NULLIF(NULL, 1);")
    assert r.rows[0][0] is None
