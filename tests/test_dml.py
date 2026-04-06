from __future__ import annotations
from tests.conftest import *

def test_insert_single():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, name VARCHAR);")
    r = e.execute("INSERT INTO t VALUES (1, 'hello');")
    assert r.affected_rows == 1

def test_insert_multiple():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1), (2), (3);")
    r = e.execute("SELECT COUNT(*) FROM t;")
    assert_value(r, 3)

def test_insert_columns():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b VARCHAR, c INT);")
    e.execute("INSERT INTO t (a, c) VALUES (1, 3);")
    r = e.execute("SELECT * FROM t;")
    assert r.rows[0] == [1, None, 3]

def test_insert_select():
    e = make_engine()
    e.execute("CREATE TABLE src (a INT);")
    e.execute("CREATE TABLE dst (a INT);")
    e.execute("INSERT INTO src VALUES (10), (20), (30);")
    r = e.execute("INSERT INTO dst SELECT * FROM src;")
    assert r.affected_rows == 3
    r = e.execute("SELECT SUM(a) FROM dst;")
    assert_value(r, 60)

def test_insert_column_count_mismatch():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    assert_error(lambda: e.execute("INSERT INTO t VALUES (1);"))

def test_update_simple():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, val INT);")
    e.execute("INSERT INTO t VALUES (1, 10), (2, 20);")
    r = e.execute("UPDATE t SET val = 99 WHERE id = 1;")
    assert r.affected_rows == 1
    r = e.execute("SELECT val FROM t WHERE id = 1;")
    assert_value(r, 99)

def test_update_expression():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, val INT);")
    e.execute("INSERT INTO t VALUES (1, 10);")
    e.execute("UPDATE t SET val = val * 2 WHERE id = 1;")
    r = e.execute("SELECT val FROM t WHERE id = 1;")
    assert_value(r, 20)

def test_update_all_rows():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1), (2), (3);")
    r = e.execute("UPDATE t SET a = 0;")
    assert r.affected_rows == 3
    r = e.execute("SELECT SUM(a) FROM t;")
    assert_value(r, 0)

def test_update_no_match():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1);")
    r = e.execute("UPDATE t SET a = 99 WHERE a = 999;")
    assert r.affected_rows == 0

def test_delete_where():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1), (2), (3), (4), (5);")
    r = e.execute("DELETE FROM t WHERE a > 3;")
    assert r.affected_rows == 2
    r = e.execute("SELECT COUNT(*) FROM t;")
    assert_value(r, 3)

def test_delete_all():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1), (2);")
    r = e.execute("DELETE FROM t;")
    assert r.affected_rows == 2
    r = e.execute("SELECT COUNT(*) FROM t;")
    assert_value(r, 0)

def test_delete_no_match():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1);")
    r = e.execute("DELETE FROM t WHERE a = 999;")
    assert r.affected_rows == 0

def test_update_then_select():
    """验证 UPDATE 后立即 SELECT 能看到新数据。"""
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    e.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30);")
    e.execute("UPDATE t SET b = b + 1 WHERE a = 2;")
    r = e.execute("SELECT b FROM t WHERE a = 2;")
    assert_value(r, 21)

def test_delete_then_count():
    """验证 DELETE 后行数正确。"""
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
    e.execute("DELETE FROM t WHERE a IN (2, 4);")
    r = e.execute("SELECT COUNT(*) FROM t;")
    assert_value(r, 3)

def test_varchar_max_length():
    e = make_engine()
    e.execute("CREATE TABLE t (s VARCHAR(5));")
    e.execute("INSERT INTO t VALUES ('abcdefgh');")
    r = e.execute("SELECT s FROM t;")
    assert len(r.rows[0][0]) <= 5
