from __future__ import annotations
from tests.conftest import *

def test_count_star():
    e = make_engine_with_data()
    assert_value(e.execute("SELECT COUNT(*) FROM users;"), 5)

def test_count_column():
    e = make_engine_with_data()
    assert_value(e.execute("SELECT COUNT(salary) FROM users;"), 3)

def test_count_distinct():
    e = make_engine_with_data()
    r = e.execute("SELECT COUNT(DISTINCT active) FROM users;")
    assert r.rows[0][0] == 2

def test_sum():
    e = make_engine_with_data()
    assert_value(e.execute("SELECT SUM(age) FROM users;"), 118)

def test_sum_null():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (NULL),(NULL);")
    r = e.execute("SELECT SUM(a) FROM t;")
    assert r.rows[0][0] is None

def test_avg():
    e = make_engine_with_data()
    r = e.execute("SELECT AVG(age) FROM users;")
    assert abs(r.rows[0][0] - 29.5) < 0.01

def test_min_max():
    e = make_engine_with_data()
    r = e.execute("SELECT MIN(age), MAX(age) FROM users;")
    assert r.rows[0] == [25, 35]

def test_group_by():
    e = make_engine_with_data()
    r = e.execute("SELECT active, COUNT(*) FROM users GROUP BY active;")
    assert r.row_count >= 2

def test_group_by_having():
    e = make_engine_with_data()
    r = e.execute("""
        SELECT active, COUNT(*) AS cnt FROM users
        GROUP BY active HAVING COUNT(*) > 1;
    """)
    assert all(row[1] > 1 for row in r.rows)

def test_group_by_expression():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    e.execute("INSERT INTO t VALUES (1,10),(1,20),(2,30),(2,40);")
    r = e.execute("SELECT a, SUM(b) FROM t GROUP BY a ORDER BY a;")
    assert_rows_ordered(r, [(1, 30), (2, 70)])

def test_group_by_null():
    e = make_engine()
    e.execute("CREATE TABLE t (g INT, v INT);")
    e.execute("INSERT INTO t VALUES (1,10),(1,20),(NULL,30),(NULL,40);")
    r = e.execute("SELECT g, SUM(v) FROM t GROUP BY g;")
    assert r.row_count == 2

def test_stddev():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (2),(4),(4),(4),(5),(5),(7),(9);")
    r = e.execute("SELECT STDDEV(a) FROM t;")
    assert r.rows[0][0] is not None and r.rows[0][0] > 0

def test_median():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
    r = e.execute("SELECT MEDIAN(a) FROM t;")
    assert_value(r, 3)

def test_string_agg():
    e = make_engine()
    e.execute("CREATE TABLE t (a VARCHAR);")
    e.execute("INSERT INTO t VALUES ('x'),('y'),('z');")
    r = e.execute("SELECT STRING_AGG(a, ',') FROM t;")
    parts = set(r.rows[0][0].split(','))
    assert parts == {'x', 'y', 'z'}

def test_empty_table_agg():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    r = e.execute("SELECT COUNT(*), SUM(a), AVG(a), MIN(a), MAX(a) FROM t;")
    assert r.rows[0][0] == 0
    assert r.rows[0][1] is None

def test_scalar_agg_no_group_by():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (10),(20),(30);")
    r = e.execute("SELECT COUNT(*), SUM(a), AVG(a) FROM t;")
    assert r.rows[0] == [3, 60, 20.0]
