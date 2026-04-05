from __future__ import annotations
"""Core SQL functionality tests."""
from engine import Engine


def _e():
    return Engine()


def test_create_drop():
    e = _e()
    e.execute("CREATE TABLE t (id INT, name VARCHAR(50));")
    assert 't' in e.get_table_names()
    e.execute("DROP TABLE t;")
    assert 't' not in e.get_table_names()
    e.execute("CREATE TABLE IF NOT EXISTS t2 (x INT);")
    e.execute("CREATE TABLE IF NOT EXISTS t2 (x INT);")
    e.execute("DROP TABLE IF EXISTS t2;")
    e.execute("DROP TABLE IF EXISTS t2;")
    print("    CREATE/DROP ✓")

def test_insert_select():
    e = _e()
    e.execute("CREATE TABLE t (id INT, name VARCHAR(50), score INT);")
    e.execute("INSERT INTO t VALUES (1,'Alice',90),(2,'Bob',85),(3,'Carol',92);")
    r = e.execute("SELECT * FROM t;")
    assert r.row_count == 3
    r = e.execute("SELECT name FROM t WHERE score > 88 ORDER BY score DESC;")
    assert r.rows[0][0] == 'Carol'
    print("    INSERT/SELECT ✓")
    e.execute("DROP TABLE t;")

def test_update_delete():
    e = _e()
    e.execute("CREATE TABLE t (id INT, val INT);")
    e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
    e.execute("UPDATE t SET val = 99 WHERE id = 2;")
    r = e.execute("SELECT val FROM t WHERE id = 2;")
    assert r.rows[0][0] == 99
    e.execute("DELETE FROM t WHERE id = 3;")
    assert e.get_table_row_count('t') == 2
    print("    UPDATE/DELETE ✓")
    e.execute("DROP TABLE t;")

def test_null_handling():
    e = _e()
    e.execute("CREATE TABLE t (id INT, val INT);")
    e.execute("INSERT INTO t VALUES (1,NULL),(2,10),(3,NULL);")
    r = e.execute("SELECT COUNT(*), COUNT(val) FROM t;")
    assert r.rows[0][0] == 3 and r.rows[0][1] == 1
    r = e.execute("SELECT id FROM t WHERE val IS NULL ORDER BY id;")
    assert r.row_count == 2
    r = e.execute("SELECT COALESCE(val, -1) FROM t WHERE id = 1;")
    assert r.rows[0][0] == -1
    print("    NULL handling ✓")
    e.execute("DROP TABLE t;")

def test_order_limit_offset():
    e = _e()
    e.execute("CREATE TABLE t (id INT);")
    for i in range(20): e.execute(f"INSERT INTO t VALUES ({i});")
    r = e.execute("SELECT id FROM t ORDER BY id DESC LIMIT 5 OFFSET 3;")
    assert r.row_count == 5 and r.rows[0][0] == 16
    r = e.execute("SELECT id FROM t ORDER BY id LIMIT 0;")
    assert r.row_count == 0
    print("    ORDER/LIMIT/OFFSET ✓")
    e.execute("DROP TABLE t;")

def test_distinct():
    e = _e()
    e.execute("CREATE TABLE t (val INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(2),(3),(3),(3);")
    r = e.execute("SELECT DISTINCT val FROM t ORDER BY val;")
    assert r.row_count == 3
    print("    DISTINCT ✓")
    e.execute("DROP TABLE t;")

def test_group_by_having():
    e = _e()
    e.execute("CREATE TABLE t (dept VARCHAR(10), salary INT);")
    e.execute("INSERT INTO t VALUES ('A',100),('A',200),('B',150),('B',250),('B',350);")
    r = e.execute("SELECT dept, COUNT(*), SUM(salary) FROM t GROUP BY dept ORDER BY dept;")
    assert r.row_count == 2
    assert r.rows[0][1] == 2 and r.rows[1][1] == 3
    r = e.execute("SELECT dept, AVG(salary) AS a FROM t GROUP BY dept HAVING AVG(salary) > 200;")
    assert r.row_count == 1 and r.rows[0][0] == 'B'
    print("    GROUP BY/HAVING ✓")
    e.execute("DROP TABLE t;")

def test_join():
    e = _e()
    e.execute("CREATE TABLE a (id INT, name VARCHAR(10));")
    e.execute("CREATE TABLE b (id INT, info VARCHAR(10));")
    e.execute("INSERT INTO a VALUES (1,'x'),(2,'y'),(3,'z');")
    e.execute("INSERT INTO b VALUES (2,'p'),(3,'q'),(4,'r');")
    r = e.execute("SELECT a.name, b.info FROM a INNER JOIN b ON a.id = b.id ORDER BY a.name;")
    assert r.row_count == 2
    r = e.execute("SELECT a.name, b.info FROM a LEFT JOIN b ON a.id = b.id;")
    assert r.row_count == 3
    r = e.execute("SELECT a.name, b.info FROM a FULL OUTER JOIN b ON a.id = b.id;")
    assert r.row_count == 4
    print("    JOIN ✓")
    e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

def test_subquery():
    e = _e()
    e.execute("CREATE TABLE t (id INT, val INT);")
    e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
    r = e.execute("SELECT id FROM t WHERE val IN (SELECT val FROM t WHERE val > 15);")
    assert r.row_count == 2
    print("    Subquery ✓")
    e.execute("DROP TABLE t;")

def test_union():
    e = _e()
    e.execute("CREATE TABLE a (x INT);"); e.execute("CREATE TABLE b (y INT);")
    e.execute("INSERT INTO a VALUES (1),(2),(3);")
    e.execute("INSERT INTO b VALUES (2),(3),(4);")
    r = e.execute("SELECT x FROM a UNION SELECT y FROM b;")
    assert r.row_count == 4
    r = e.execute("SELECT x FROM a UNION ALL SELECT y FROM b;")
    assert r.row_count == 6
    print("    UNION ✓")
    e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

def test_cte():
    e = _e()
    e.execute("CREATE TABLE t (id INT, val INT);")
    e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
    r = e.execute("WITH big AS (SELECT * FROM t WHERE val > 15) SELECT COUNT(*) FROM big;")
    assert r.rows[0][0] == 2
    assert 'big' not in e.get_table_names()
    print("    CTE ✓")
    e.execute("DROP TABLE t;")

def test_window():
    e = _e()
    e.execute("CREATE TABLE t (id INT, val INT);")
    for i in range(10): e.execute(f"INSERT INTO t VALUES ({i}, {i*10});")
    r = e.execute("SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t;")
    assert r.row_count == 10 and r.rows[0][1] == 1
    r = e.execute("SELECT id, SUM(val) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t LIMIT 3;")
    assert r.row_count == 3
    print("    Window ✓")
    e.execute("DROP TABLE t;")

def test_case_cast():
    e = _e()
    r = e.execute("SELECT CASE WHEN 1 > 0 THEN 'yes' ELSE 'no' END;")
    assert r.rows[0][0] == 'yes'
    r = e.execute("SELECT CAST(42 AS DOUBLE);")
    assert r.rows[0][0] == 42.0
    print("    CASE/CAST ✓")

def test_like_between_in():
    e = _e()
    e.execute("CREATE TABLE t (name VARCHAR(20), age INT);")
    e.execute("INSERT INTO t VALUES ('Alice',30),('Bob',25),('Carol',35);")
    r = e.execute("SELECT name FROM t WHERE name LIKE 'A%';")
    assert r.rows[0][0] == 'Alice'
    r = e.execute("SELECT name FROM t WHERE age BETWEEN 26 AND 31;")
    assert r.row_count == 1
    r = e.execute("SELECT name FROM t WHERE age IN (25, 35);")
    assert r.row_count == 2
    print("    LIKE/BETWEEN/IN ✓")
    e.execute("DROP TABLE t;")

def test_alter_table():
    e = _e()
    e.execute("CREATE TABLE t (id INT, name VARCHAR(20));")
    e.execute("INSERT INTO t VALUES (1,'hello'),(2,'world');")
    e.execute("ALTER TABLE t ADD COLUMN score INT;")
    r = e.execute("SELECT * FROM t;")
    assert len(r.columns) == 3
    e.execute("ALTER TABLE t DROP COLUMN score;")
    r = e.execute("SELECT * FROM t;")
    assert len(r.columns) == 2
    e.execute("ALTER TABLE t RENAME COLUMN name TO title;")
    r = e.execute("SELECT title FROM t;")
    assert r.row_count == 2
    print("    ALTER TABLE ✓")
    e.execute("DROP TABLE t;")

def test_explain():
    e = _e()
    e.execute("CREATE TABLE t (id INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3);")
    r = e.execute("EXPLAIN SELECT * FROM t WHERE id > 1;")
    assert r.row_count > 0
    assert any('Scan' in str(row) or 'Filter' in str(row) or 'Project' in str(row) for row in r.rows)
    print("    EXPLAIN ✓")
    e.execute("DROP TABLE t;")

def test_expression_arithmetic():
    e = _e()
    r = e.execute("SELECT 1 + 2 * 3;")
    assert r.rows[0][0] == 7
    r = e.execute("SELECT 10 / 3;")
    assert r.rows[0][0] == 3
    r = e.execute("SELECT 10 % 3;")
    assert r.rows[0][0] == 1
    r = e.execute("SELECT 'hello' || ' ' || 'world';")
    assert r.rows[0][0] == 'hello world'
    print("    Arithmetic ✓")

def test_current_date():
    e = _e()
    r = e.execute("SELECT CURRENT_DATE;")
    val = r.rows[0][0]
    # Should be formatted as date string by display, but value is int (epoch days)
    assert isinstance(val, int) and val > 0
    print(f"    CURRENT_DATE = {val} ✓")


ALL_TESTS = [
    ("CREATE/DROP", test_create_drop),
    ("INSERT/SELECT", test_insert_select),
    ("UPDATE/DELETE", test_update_delete),
    ("NULL handling", test_null_handling),
    ("ORDER/LIMIT/OFFSET", test_order_limit_offset),
    ("DISTINCT", test_distinct),
    ("GROUP BY/HAVING", test_group_by_having),
    ("JOIN", test_join),
    ("Subquery", test_subquery),
    ("UNION", test_union),
    ("CTE", test_cte),
    ("Window", test_window),
    ("CASE/CAST", test_case_cast),
    ("LIKE/BETWEEN/IN", test_like_between_in),
    ("ALTER TABLE", test_alter_table),
    ("EXPLAIN", test_explain),
    ("Arithmetic", test_expression_arithmetic),
    ("CURRENT_DATE", test_current_date),
]
