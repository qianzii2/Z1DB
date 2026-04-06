from __future__ import annotations
from tests.conftest import *

def _setup(e):
    e.execute("CREATE TABLE sales (dept VARCHAR, emp VARCHAR, amount INT);")
    e.execute("""INSERT INTO sales VALUES
        ('A','x',100),('A','y',200),('A','z',150),
        ('B','p',300),('B','q',250);""")

def test_row_number():
    e = make_engine(); _setup(e)
    r = e.execute("SELECT emp, ROW_NUMBER() OVER (ORDER BY amount DESC) AS rn FROM sales;")
    assert r.row_count == 5
    assert r.rows[0][1] == 1

def test_rank():
    e = make_engine(); _setup(e)
    r = e.execute("""
        SELECT emp, RANK() OVER (PARTITION BY dept ORDER BY amount DESC) AS rnk
        FROM sales;""")
    assert r.row_count == 5

def test_dense_rank():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (10),(20),(20),(30);")
    r = e.execute("SELECT a, DENSE_RANK() OVER (ORDER BY a) AS dr FROM t;")
    ranks = [row[1] for row in r.rows]
    assert max(ranks) == 3  # 10→1, 20→2, 30→3

def test_sum_over_partition():
    e = make_engine(); _setup(e)
    r = e.execute("""
        SELECT emp, SUM(amount) OVER (PARTITION BY dept) AS total
        FROM sales;""")
    for row in r.rows:
        if row[0] in ('x', 'y', 'z'):
            assert row[1] == 450
        else:
            assert row[1] == 550

def test_lag():
    e = make_engine(); _setup(e)
    r = e.execute("""
        SELECT emp, amount, LAG(amount, 1) OVER (ORDER BY amount) AS prev
        FROM sales;""")
    assert r.rows[0][2] is None  # 第一行无前值

def test_lead():
    e = make_engine(); _setup(e)
    r = e.execute("""
        SELECT emp, LEAD(amount, 1) OVER (ORDER BY amount) AS nxt
        FROM sales;""")
    assert r.rows[-1][1] is None  # 最后一行无后值

def test_ntile():
    e = make_engine(); _setup(e)
    r = e.execute("SELECT emp, NTILE(2) OVER (ORDER BY amount) AS tile FROM sales;")
    tiles = [row[1] for row in r.rows]
    assert set(tiles) == {1, 2}

def test_first_value():
    e = make_engine(); _setup(e)
    r = e.execute("""
        SELECT emp, FIRST_VALUE(amount) OVER (PARTITION BY dept ORDER BY amount) AS fv
        FROM sales;""")
    for row in r.rows:
        if row[0] in ('x', 'y', 'z'): assert row[1] == 100
        else: assert row[1] == 250

def test_running_sum():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
    r = e.execute("SELECT a, SUM(a) OVER (ORDER BY a) AS running FROM t;")
    sums = [row[1] for row in r.rows]
    assert sums == [1, 3, 6, 10, 15]

def test_window_frame():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
    r = e.execute("""
        SELECT a, SUM(a) OVER (ORDER BY a ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s
        FROM t;""")
    assert r.row_count == 5
    # 第一行: sum(1,2) = 3
    assert r.rows[0][1] == 3
