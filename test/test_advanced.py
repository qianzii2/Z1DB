from __future__ import annotations
"""Tests for advanced features: aggregates, functions, sketches."""
from engine import Engine


def _e():
    return Engine()


def test_aggregates():
    e = _e()
    e.execute("CREATE TABLE t (v INT);")
    for i in range(1, 11): e.execute(f"INSERT INTO t VALUES ({i});")
    r = e.execute("SELECT COUNT(*), SUM(v), AVG(v), MIN(v), MAX(v) FROM t;")
    assert r.rows[0] == [10, 55, 5.5, 1, 10]
    r = e.execute("SELECT MEDIAN(v) FROM t;")
    assert r.rows[0][0] == 5.5
    r = e.execute("SELECT STDDEV(v) FROM t;")
    assert r.rows[0][0] is not None
    r = e.execute("SELECT MODE(v) FROM t;")
    assert r.rows[0][0] is not None
    r = e.execute("SELECT COUNT(DISTINCT v) FROM t;")
    assert r.rows[0][0] == 10
    print("    Aggregates ✓")
    e.execute("DROP TABLE t;")

def test_string_functions():
    e = _e()
    r = e.execute("SELECT UPPER('hello'), LOWER('WORLD'), LENGTH('test');")
    assert r.rows[0] == ['HELLO', 'world', 4]
    r = e.execute("SELECT TRIM('  hi  '), REVERSE('abc'), INITCAP('hello world');")
    assert r.rows[0] == ['hi', 'cba', 'Hello World']
    r = e.execute("SELECT SUBSTR('hello', 2, 3);")
    assert r.rows[0][0] == 'ell'
    r = e.execute("SELECT REPLACE('hello', 'l', 'r');")
    assert r.rows[0][0] == 'herro'
    print("    String Functions ✓")

def test_math_functions():
    e = _e()
    r = e.execute("SELECT ABS(-5), CEIL(3.2), FLOOR(3.8), ROUND(3.5);")
    assert r.rows[0] == [5, 4, 3, 4]
    r = e.execute("SELECT POWER(2, 10), SQRT(144), SIGN(-3);")
    assert r.rows[0][0] == 1024.0 and r.rows[0][1] == 12.0 and r.rows[0][2] == -1
    print("    Math Functions ✓")

def test_date_functions():
    e = _e()
    r = e.execute("SELECT YEAR(0), MONTH(0), DAY(0);")
    assert r.rows[0] == [1970, 1, 1]
    r = e.execute("SELECT DATE_ADD(0, 365);")
    assert r.rows[0][0] == 365
    r = e.execute("SELECT DATE_DIFF(365, 0);")
    assert r.rows[0][0] == 365
    print("    Date Functions ✓")

def test_conditional_functions():
    e = _e()
    r = e.execute("SELECT COALESCE(NULL, NULL, 42);")
    assert r.rows[0][0] == 42
    r = e.execute("SELECT NULLIF(1, 1), NULLIF(1, 2);")
    assert r.rows[0][0] is None and r.rows[0][1] == 1
    r = e.execute("SELECT TYPEOF(42), TYPEOF('hello');")
    assert r.rows[0][0] == 'INT' and r.rows[0][1] == 'VARCHAR'
    print("    Conditional Functions ✓")

def test_array_functions():
    e = _e()
    r = e.execute("SELECT ARRAY_LENGTH('[1,2,3]');")
    assert r.rows[0][0] == 3
    r = e.execute("SELECT ARRAY_SORT('[3,1,2]');")
    assert '1' in r.rows[0][0]
    r = e.execute("SELECT ARRAY_JOIN('[1,2,3]', '-');")
    assert r.rows[0][0] == '1-2-3'
    print("    Array Functions ✓")

def test_similarity_functions():
    e = _e()
    r = e.execute("SELECT JACCARD_SIMILARITY('[1,2,3]', '[2,3,4]');")
    assert 0.3 < r.rows[0][0] < 0.6
    r = e.execute("SELECT COSINE_SIMILARITY('[1,0]', '[0,1]');")
    assert abs(r.rows[0][0]) < 0.01
    print("    Similarity ✓")

def test_approx_aggregates():
    e = _e()
    e.execute("CREATE TABLE t (v INT);")
    for i in range(100): e.execute(f"INSERT INTO t VALUES ({i % 20});")
    r = e.execute("SELECT APPROX_COUNT_DISTINCT(v) FROM t;")
    assert 15 <= r.rows[0][0] <= 25
    print(f"    APPROX_COUNT_DISTINCT = {r.rows[0][0]} ✓")
    e.execute("DROP TABLE t;")

def test_string_agg_separator():
    e = _e()
    e.execute("CREATE TABLE t (name VARCHAR(10));")
    e.execute("INSERT INTO t VALUES ('a'),('b'),('c');")
    r = e.execute("SELECT STRING_AGG(name, ' | ') FROM t;")
    assert r.rows[0][0] is not None
    print(f"    STRING_AGG = '{r.rows[0][0]}' ✓")
    e.execute("DROP TABLE t;")

def test_window_advanced():
    e = _e()
    e.execute("CREATE TABLE t (region VARCHAR(10), amount INT);")
    e.execute("INSERT INTO t VALUES ('A',10),('A',20),('A',30),('B',40),('B',50);")
    r = e.execute("""
        SELECT region, amount,
               SUM(amount) OVER (PARTITION BY region ORDER BY amount
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running
        FROM t ORDER BY region, amount;
    """)
    assert r.row_count == 5
    assert r.rows[0][2] == 10  # First row running sum
    assert r.rows[2][2] == 60  # A: 10+20+30
    print("    Window Advanced ✓")
    e.execute("DROP TABLE t;")


ALL_TESTS = [
    ("Aggregates", test_aggregates),
    ("String Functions", test_string_functions),
    ("Math Functions", test_math_functions),
    ("Date Functions", test_date_functions),
    ("Conditional Functions", test_conditional_functions),
    ("Array Functions", test_array_functions),
    ("Similarity", test_similarity_functions),
    ("Approx Aggregates", test_approx_aggregates),
    ("STRING_AGG separator", test_string_agg_separator),
    ("Window Advanced", test_window_advanced),
]
