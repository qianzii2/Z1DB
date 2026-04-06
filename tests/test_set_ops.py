from __future__ import annotations
from tests.conftest import *

def _setup(e):
    e.execute("CREATE TABLE t1 (a INT);")
    e.execute("CREATE TABLE t2 (a INT);")
    e.execute("INSERT INTO t1 VALUES (1),(2),(3);")
    e.execute("INSERT INTO t2 VALUES (2),(3),(4);")

def test_union():
    e = make_engine(); _setup(e)
    r = e.execute("SELECT a FROM t1 UNION SELECT a FROM t2;")
    assert r.row_count == 4  # 1,2,3,4 去重

def test_union_all():
    e = make_engine(); _setup(e)
    r = e.execute("SELECT a FROM t1 UNION ALL SELECT a FROM t2;")
    assert r.row_count == 6

def test_intersect():
    e = make_engine(); _setup(e)
    r = e.execute("SELECT a FROM t1 INTERSECT SELECT a FROM t2;")
    assert_rows(r, [(2,),(3,)])

def test_intersect_all():
    e = make_engine()
    e.execute("CREATE TABLE t1 (a INT);")
    e.execute("CREATE TABLE t2 (a INT);")
    e.execute("INSERT INTO t1 VALUES (1),(2),(2),(3);")
    e.execute("INSERT INTO t2 VALUES (2),(2),(3),(4);")
    r = e.execute("SELECT a FROM t1 INTERSECT ALL SELECT a FROM t2;")
    assert r.row_count == 3  # 2,2,3

def test_except():
    e = make_engine(); _setup(e)
    r = e.execute("SELECT a FROM t1 EXCEPT SELECT a FROM t2;")
    assert_rows(r, [(1,)])

def test_except_all():
    e = make_engine()
    e.execute("CREATE TABLE t1 (a INT);")
    e.execute("CREATE TABLE t2 (a INT);")
    e.execute("INSERT INTO t1 VALUES (1),(2),(2),(3);")
    e.execute("INSERT INTO t2 VALUES (2),(3);")
    r = e.execute("SELECT a FROM t1 EXCEPT ALL SELECT a FROM t2;")
    assert_rows(r, [(1,),(2,)])

def test_union_with_order():
    e = make_engine(); _setup(e)
    r = e.execute("SELECT a FROM t1 UNION SELECT a FROM t2 ORDER BY 1 DESC;")
    assert r.rows[0][0] == 4

def test_union_null():
    e = make_engine()
    e.execute("CREATE TABLE t1 (a INT);")
    e.execute("CREATE TABLE t2 (a INT);")
    e.execute("INSERT INTO t1 VALUES (1),(NULL);")
    e.execute("INSERT INTO t2 VALUES (NULL),(2);")
    r = e.execute("SELECT a FROM t1 UNION SELECT a FROM t2;")
    assert r.row_count == 3  # 1, 2, NULL
