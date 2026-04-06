from __future__ import annotations
from tests.conftest import *

def test_scalar_subquery():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE age = (SELECT MAX(age) FROM users);")
    assert_value(r, 'Charlie')

def test_in_subquery():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE id IN (SELECT id FROM users WHERE active = TRUE);")
    assert r.row_count == 3

def test_not_in_subquery():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE id NOT IN (SELECT id FROM users WHERE active = TRUE);")
    # id 3 (FALSE) 和 5 (NULL active) — NOT IN 有 NULL 语义
    assert r.row_count >= 1

def test_exists():
    e = make_engine()
    e.execute("CREATE TABLE t1 (a INT);")
    e.execute("CREATE TABLE t2 (b INT);")
    e.execute("INSERT INTO t1 VALUES (1),(2),(3);")
    e.execute("INSERT INTO t2 VALUES (1),(2);")
    r = e.execute("SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.b = t1.a);")
    assert_rows(r, [(1,),(2,)])

def test_not_exists():
    e = make_engine()
    e.execute("CREATE TABLE t1 (a INT);")
    e.execute("CREATE TABLE t2 (b INT);")
    e.execute("INSERT INTO t1 VALUES (1),(2),(3);")
    e.execute("INSERT INTO t2 VALUES (1),(2);")
    r = e.execute("SELECT a FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.b = t1.a);")
    assert_rows(r, [(3,)])

def test_cte_simple():
    e = make_engine_with_data()
    r = e.execute("WITH s AS (SELECT * FROM users WHERE age > 30) SELECT name FROM s;")
    assert_value(r, 'Charlie')

def test_cte_multiple():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
    r = e.execute("""
        WITH lo AS (SELECT a FROM t WHERE a <= 2),
             hi AS (SELECT a FROM t WHERE a >= 4)
        SELECT * FROM lo UNION ALL SELECT * FROM hi;
    """)
    assert r.row_count == 4  # 1,2,4,5

def test_cte_with_columns():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    e.execute("INSERT INTO t VALUES (1,10),(2,20);")
    r = e.execute("WITH c(x, y) AS (SELECT a, b FROM t) SELECT x, y FROM c;")
    assert r.row_count == 2
    assert r.columns == ['x', 'y']

def test_subquery_in_from():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3);")
    r = e.execute("SELECT s.a FROM (SELECT a FROM t WHERE a > 1) s;")
    assert_rows(r, [(2,),(3,)])

def test_subquery_in_select():
    e = make_engine_with_data()
    r = e.execute("SELECT name, (SELECT MAX(age) FROM users) AS max_age FROM users WHERE id = 1;")
    assert r.rows[0][0] == 'Alice'
    assert r.rows[0][1] == 35

def test_correlated_subquery():
    e = make_engine()
    e.execute("CREATE TABLE t (dept VARCHAR, salary INT);")
    e.execute("INSERT INTO t VALUES ('A',100),('A',200),('B',150),('B',300);")
    r = e.execute("""
        SELECT dept, salary FROM t t1
        WHERE salary = (SELECT MAX(salary) FROM t t2 WHERE t2.dept = t1.dept);
    """)
    assert r.row_count == 2
    salaries = sorted([row[1] for row in r.rows])
    assert salaries == [200, 300]
