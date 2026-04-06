from __future__ import annotations
from tests.conftest import *

def _setup_join(e):
    e.execute("CREATE TABLE t1 (id INT, val VARCHAR);")
    e.execute("CREATE TABLE t2 (id INT, ref INT, data VARCHAR);")
    e.execute("INSERT INTO t1 VALUES (1,'a'), (2,'b'), (3,'c');")
    e.execute("INSERT INTO t2 VALUES (10,1,'x'), (20,2,'y'), (30,4,'z');")

def test_inner_join():
    e = make_engine(); _setup_join(e)
    r = e.execute("SELECT t1.val, t2.data FROM t1 JOIN t2 ON t1.id = t2.ref;")
    assert_rows(r, [('a','x'), ('b','y')])

def test_left_join():
    e = make_engine(); _setup_join(e)
    r = e.execute("SELECT t1.val, t2.data FROM t1 LEFT JOIN t2 ON t1.id = t2.ref;")
    assert r.row_count == 3
    vals = {row[0] for row in r.rows}
    assert 'c' in vals  # c 有右侧 NULL

def test_right_join():
    e = make_engine(); _setup_join(e)
    r = e.execute("SELECT t1.val, t2.data FROM t1 RIGHT JOIN t2 ON t1.id = t2.ref;")
    assert r.row_count == 3
    datas = {row[1] for row in r.rows}
    assert 'z' in datas  # z 有左侧 NULL

def test_full_join():
    e = make_engine(); _setup_join(e)
    r = e.execute("SELECT t1.val, t2.data FROM t1 FULL JOIN t2 ON t1.id = t2.ref;")
    assert r.row_count == 4

def test_cross_join():
    e = make_engine(); _setup_join(e)
    r = e.execute("SELECT * FROM t1 CROSS JOIN t2;")
    assert r.row_count == 9

def test_self_join():
    e = make_engine()
    e.execute("CREATE TABLE emp (id INT, name VARCHAR, mgr INT);")
    e.execute("INSERT INTO emp VALUES (1,'Boss',NULL),(2,'Alice',1),(3,'Bob',1);")
    r = e.execute("""
        SELECT e.name, m.name AS mgr_name
        FROM emp e LEFT JOIN emp m ON e.mgr = m.id;
    """)
    assert r.row_count == 3

def test_join_with_where():
    e = make_engine(); _setup_join(e)
    r = e.execute("""
        SELECT t1.val FROM t1
        JOIN t2 ON t1.id = t2.ref
        WHERE t2.data = 'x';
    """)
    assert_value(r, 'a')

def test_multi_join():
    e = make_engine()
    e.execute("CREATE TABLE a (id INT, x INT);")
    e.execute("CREATE TABLE b (id INT, y INT);")
    e.execute("CREATE TABLE c (id INT, z INT);")
    e.execute("INSERT INTO a VALUES (1,10);")
    e.execute("INSERT INTO b VALUES (1,20);")
    e.execute("INSERT INTO c VALUES (1,30);")
    r = e.execute("""
        SELECT a.x, b.y, c.z FROM a
        JOIN b ON a.id = b.id
        JOIN c ON a.id = c.id;
    """)
    assert r.rows[0] == [10, 20, 30]

def test_join_subquery():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    e.execute("INSERT INTO t VALUES (1,10),(2,20);")
    r = e.execute("""
        SELECT t.a, s.total FROM t
        JOIN (SELECT a, SUM(b) AS total FROM t GROUP BY a) s ON t.a = s.a;
    """)
    assert r.row_count == 2

def test_natural_join():
    e = make_engine()
    e.execute("CREATE TABLE t1 (id INT, name VARCHAR);")
    e.execute("CREATE TABLE t2 (id INT, score INT);")
    e.execute("INSERT INTO t1 VALUES (1,'Alice'),(2,'Bob');")
    e.execute("INSERT INTO t2 VALUES (1,90),(2,80);")
    r = e.execute("SELECT * FROM t1 NATURAL JOIN t2;")
    assert r.row_count == 2

def test_comma_join():
    e = make_engine(); _setup_join(e)
    r = e.execute("SELECT * FROM t1, t2 WHERE t1.id = t2.ref;")
    assert r.row_count == 2
