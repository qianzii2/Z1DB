from __future__ import annotations
"""SQL 端到端测试 — 覆盖全部 SQL 功能。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine


def E():
    return Engine()


class TestDDL:
    def test_create_drop(self):
        e = E()
        r = e.execute("CREATE TABLE t (id INT, name VARCHAR(50));")
        assert r.message == 'OK'
        assert 't' in e.get_table_names()
        e.execute("DROP TABLE t;")
        assert 't' not in e.get_table_names()

    def test_if_not_exists(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        r = e.execute("CREATE TABLE IF NOT EXISTS t (id INT);")
        assert r.message == 'OK'
        e.execute("DROP TABLE t;")

    def test_if_exists(self):
        e = E()
        r = e.execute("DROP TABLE IF EXISTS nonexistent;")
        assert r.message == 'OK'

    def test_alter_add(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (1);")
        e.execute("ALTER TABLE t ADD COLUMN name VARCHAR(50);")
        r = e.execute("SELECT * FROM t;")
        assert r.row_count == 1
        assert len(r.columns) == 2
        assert r.rows[0][1] is None  # New column defaults to NULL
        e.execute("DROP TABLE t;")

    def test_alter_drop(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, name VARCHAR(50));")
        e.execute("INSERT INTO t VALUES (1, 'x');")
        e.execute("ALTER TABLE t DROP COLUMN name;")
        r = e.execute("SELECT * FROM t;")
        assert len(r.columns) == 1
        e.execute("DROP TABLE t;")

    def test_alter_rename(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, name VARCHAR(50));")
        e.execute("INSERT INTO t VALUES (1, 'x');")
        e.execute("ALTER TABLE t RENAME COLUMN name TO title;")
        r = e.execute("SELECT title FROM t;")
        assert r.rows[0][0] == 'x'
        e.execute("DROP TABLE t;")


class TestDML:
    def test_insert_select(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val VARCHAR(10));")
        e.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b');")
        r = e.execute("SELECT * FROM t;")
        assert r.row_count == 2

    def test_insert_columns(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, name VARCHAR(10), age INT);")
        e.execute("INSERT INTO t (name, id) VALUES ('Alice', 1);")
        r = e.execute("SELECT * FROM t;")
        assert r.rows[0][2] is None  # age is NULL
        e.execute("DROP TABLE t;")

    def test_update(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val INT);")
        e.execute("INSERT INTO t VALUES (1, 10), (2, 20);")
        e.execute("UPDATE t SET val = 99 WHERE id = 1;")
        r = e.execute("SELECT val FROM t WHERE id = 1;")
        assert r.rows[0][0] == 99
        e.execute("DROP TABLE t;")

    def test_delete(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (1), (2), (3);")
        e.execute("DELETE FROM t WHERE id = 2;")
        r = e.execute("SELECT COUNT(*) FROM t;")
        assert r.rows[0][0] == 2
        e.execute("DROP TABLE t;")

    def test_delete_all(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (1), (2);")
        r = e.execute("DELETE FROM t;")
        assert r.affected_rows == 2
        e.execute("DROP TABLE t;")


class TestSelect:
    def test_where(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val INT);")
        e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
        r = e.execute("SELECT id FROM t WHERE val > 15;")
        ids = sorted(row[0] for row in r.rows)
        assert ids == [2, 3]
        e.execute("DROP TABLE t;")

    def test_order_by(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (3),(1),(2);")
        r = e.execute("SELECT id FROM t ORDER BY id;")
        assert [row[0] for row in r.rows] == [1, 2, 3]
        e.execute("DROP TABLE t;")

    def test_order_by_desc(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (3),(1),(2);")
        r = e.execute("SELECT id FROM t ORDER BY id DESC;")
        assert [row[0] for row in r.rows] == [3, 2, 1]
        e.execute("DROP TABLE t;")

    def test_limit_offset(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        for i in range(10): e.execute(f"INSERT INTO t VALUES ({i});")
        r = e.execute("SELECT id FROM t ORDER BY id LIMIT 3 OFFSET 2;")
        assert [row[0] for row in r.rows] == [2, 3, 4]
        e.execute("DROP TABLE t;")

    def test_distinct(self):
        e = E()
        e.execute("CREATE TABLE t (val INT);")
        e.execute("INSERT INTO t VALUES (1),(2),(1),(3),(2);")
        r = e.execute("SELECT DISTINCT val FROM t ORDER BY val;")
        assert [row[0] for row in r.rows] == [1, 2, 3]
        e.execute("DROP TABLE t;")

    def test_no_from(self):
        e = E()
        r = e.execute("SELECT 1 + 1;")
        assert r.rows[0][0] == 2

    def test_empty_table(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        r = e.execute("SELECT * FROM t;")
        assert r.row_count == 0
        e.execute("DROP TABLE t;")

    def test_null_handling(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val INT);")
        e.execute("INSERT INTO t VALUES (1, NULL), (2, 10);")
        r = e.execute("SELECT id FROM t WHERE val IS NULL;")
        assert r.rows[0][0] == 1
        r = e.execute("SELECT id FROM t WHERE val IS NOT NULL;")
        assert r.rows[0][0] == 2
        e.execute("DROP TABLE t;")

    def test_alias(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (5);")
        r = e.execute("SELECT x + 1 AS y FROM t;")
        assert r.columns[0] == 'y'
        assert r.rows[0][0] == 6
        e.execute("DROP TABLE t;")

    def test_between(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(5),(10),(15);")
        r = e.execute("SELECT x FROM t WHERE x BETWEEN 5 AND 10 ORDER BY x;")
        assert [row[0] for row in r.rows] == [5, 10]
        e.execute("DROP TABLE t;")

    def test_in(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
        r = e.execute("SELECT x FROM t WHERE x IN (2, 4) ORDER BY x;")
        assert [row[0] for row in r.rows] == [2, 4]
        e.execute("DROP TABLE t;")

    def test_like(self):
        e = E()
        e.execute("CREATE TABLE t (name VARCHAR(20));")
        e.execute("INSERT INTO t VALUES ('Alice'),('Bob'),('Anna');")
        r = e.execute("SELECT name FROM t WHERE name LIKE 'A%' ORDER BY name;")
        assert [row[0] for row in r.rows] == ['Alice', 'Anna']
        e.execute("DROP TABLE t;")

    def test_case(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(2),(3);")
        r = e.execute("SELECT x, CASE WHEN x > 2 THEN 'high' ELSE 'low' END AS cat FROM t ORDER BY x;")
        assert r.rows[0][1] == 'low'
        assert r.rows[2][1] == 'high'
        e.execute("DROP TABLE t;")

    def test_cast(self):
        e = E()
        r = e.execute("SELECT CAST(42 AS DOUBLE);")
        assert isinstance(r.rows[0][0], float)


class TestJoin:
    def _setup(self, e):
        e.execute("CREATE TABLE a (id INT, val VARCHAR(10));")
        e.execute("CREATE TABLE b (id INT, info VARCHAR(10));")
        e.execute("INSERT INTO a VALUES (1,'a1'),(2,'a2'),(3,'a3');")
        e.execute("INSERT INTO b VALUES (2,'b2'),(3,'b3'),(4,'b4');")

    def test_inner_join(self):
        e = E(); self._setup(e)
        r = e.execute("SELECT a.val, b.info FROM a INNER JOIN b ON a.id = b.id ORDER BY a.val;")
        assert r.row_count == 2
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

    def test_left_join(self):
        e = E(); self._setup(e)
        r = e.execute("SELECT a.val, b.info FROM a LEFT JOIN b ON a.id = b.id ORDER BY a.val;")
        assert r.row_count == 3
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

    def test_full_join(self):
        e = E(); self._setup(e)
        r = e.execute("SELECT a.val, b.info FROM a FULL OUTER JOIN b ON a.id = b.id;")
        assert r.row_count == 4
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

    def test_cross_join(self):
        e = E()
        e.execute("CREATE TABLE x (a INT);")
        e.execute("CREATE TABLE y (b INT);")
        e.execute("INSERT INTO x VALUES (1),(2);")
        e.execute("INSERT INTO y VALUES (10),(20);")
        r = e.execute("SELECT x.a, y.b FROM x CROSS JOIN y ORDER BY x.a, y.b;")
        assert r.row_count == 4
        e.execute("DROP TABLE x;"); e.execute("DROP TABLE y;")


class TestAggregate:
    def test_count(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(2),(3),(NULL);")
        r = e.execute("SELECT COUNT(*), COUNT(x) FROM t;")
        assert r.rows[0][0] == 4  # COUNT(*)
        assert r.rows[0][1] == 3  # COUNT(x) skips NULL

    def test_sum_avg(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (10),(20),(30);")
        r = e.execute("SELECT SUM(x), AVG(x) FROM t;")
        assert r.rows[0][0] == 60
        assert r.rows[0][1] == 20.0

    def test_min_max(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (5),(1),(9);")
        r = e.execute("SELECT MIN(x), MAX(x) FROM t;")
        assert r.rows[0][0] == 1
        assert r.rows[0][1] == 9

    def test_group_by(self):
        e = E()
        e.execute("CREATE TABLE t (cat VARCHAR(10), val INT);")
        e.execute("INSERT INTO t VALUES ('a',1),('b',2),('a',3),('b',4);")
        r = e.execute("SELECT cat, SUM(val) AS s FROM t GROUP BY cat ORDER BY cat;")
        assert r.rows[0] == ['a', 4]
        assert r.rows[1] == ['b', 6]
        e.execute("DROP TABLE t;")

    def test_having(self):
        e = E()
        e.execute("CREATE TABLE t (cat VARCHAR(10), val INT);")
        e.execute("INSERT INTO t VALUES ('a',1),('b',2),('a',3),('b',4);")
        r = e.execute("SELECT cat, SUM(val) AS s FROM t GROUP BY cat HAVING SUM(val) > 5;")
        assert r.row_count == 1
        assert r.rows[0][0] == 'b'
        e.execute("DROP TABLE t;")

    def test_count_distinct(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(2),(1),(3),(2);")
        r = e.execute("SELECT COUNT(DISTINCT x) FROM t;")
        assert r.rows[0][0] == 3
        e.execute("DROP TABLE t;")

    def test_empty_table_agg(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        r = e.execute("SELECT COUNT(*), SUM(x), AVG(x) FROM t;")
        assert r.rows[0][0] == 0  # COUNT(*) on empty = 0
        assert r.rows[0][1] is None  # SUM on empty = NULL
        e.execute("DROP TABLE t;")

    def test_median(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
        r = e.execute("SELECT MEDIAN(x) FROM t;")
        assert r.rows[0][0] == 3
        e.execute("DROP TABLE t;")

    def test_stddev(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (10),(20),(30);")
        r = e.execute("SELECT STDDEV(x) FROM t;")
        assert r.rows[0][0] is not None and r.rows[0][0] > 0
        e.execute("DROP TABLE t;")


class TestWindow:
    def test_row_number(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val INT);")
        e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
        r = e.execute("SELECT id, ROW_NUMBER() OVER (ORDER BY val DESC) AS rn FROM t;")
        # id=3 should be rn=1
        for row in r.rows:
            if row[0] == 3: assert row[1] == 1
        e.execute("DROP TABLE t;")

    def test_sum_over(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val INT);")
        e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
        r = e.execute("""SELECT id, SUM(val) OVER (ORDER BY id
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running
                         FROM t ORDER BY id;""")
        assert r.rows[0][1] == 10
        assert r.rows[1][1] == 30
        assert r.rows[2][1] == 60
        e.execute("DROP TABLE t;")

    def test_lag_lead(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val INT);")
        e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
        r = e.execute("SELECT id, LAG(val, 1) OVER (ORDER BY id) AS prev FROM t ORDER BY id;")
        assert r.rows[0][1] is None  # No previous for first row
        assert r.rows[1][1] == 10
        e.execute("DROP TABLE t;")

    def test_partition_by(self):
        e = E()
        e.execute("CREATE TABLE t (grp VARCHAR(5), val INT);")
        e.execute("INSERT INTO t VALUES ('A',1),('A',2),('B',10),('B',20);")
        r = e.execute("SELECT grp, val, ROW_NUMBER() OVER (PARTITION BY grp ORDER BY val) AS rn FROM t;")
        assert r.row_count == 4
        e.execute("DROP TABLE t;")


class TestSetOps:
    def test_union(self):
        e = E()
        e.execute("CREATE TABLE a (x INT);")
        e.execute("CREATE TABLE b (x INT);")
        e.execute("INSERT INTO a VALUES (1),(2),(3);")
        e.execute("INSERT INTO b VALUES (2),(3),(4);")
        r = e.execute("SELECT x FROM a UNION SELECT x FROM b ORDER BY x;")
        assert [row[0] for row in r.rows] == [1, 2, 3, 4]
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

    def test_union_all(self):
        e = E()
        e.execute("CREATE TABLE a (x INT);")
        e.execute("CREATE TABLE b (x INT);")
        e.execute("INSERT INTO a VALUES (1),(2);")
        e.execute("INSERT INTO b VALUES (2),(3);")
        r = e.execute("SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x;")
        assert r.row_count == 4
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

    def test_intersect(self):
        e = E()
        e.execute("CREATE TABLE a (x INT);")
        e.execute("CREATE TABLE b (x INT);")
        e.execute("INSERT INTO a VALUES (1),(2),(3);")
        e.execute("INSERT INTO b VALUES (2),(3),(4);")
        r = e.execute("SELECT x FROM a INTERSECT SELECT x FROM b ORDER BY x;")
        assert [row[0] for row in r.rows] == [2, 3]
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

    def test_except(self):
        e = E()
        e.execute("CREATE TABLE a (x INT);")
        e.execute("CREATE TABLE b (x INT);")
        e.execute("INSERT INTO a VALUES (1),(2),(3);")
        e.execute("INSERT INTO b VALUES (2),(3),(4);")
        r = e.execute("SELECT x FROM a EXCEPT SELECT x FROM b;")
        assert r.rows[0][0] == 1
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")


class TestCTE:
    def test_simple_cte(self):
        e = E()
        e.execute("CREATE TABLE t (id INT, val INT);")
        e.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30);")
        r = e.execute("WITH big AS (SELECT * FROM t WHERE val > 15) SELECT * FROM big ORDER BY id;")
        assert r.row_count == 2
        # CTE should not pollute catalog
        assert 'big' not in e.get_table_names()
        e.execute("DROP TABLE t;")

    def test_multiple_ctes(self):
        e = E()
        r = e.execute("""WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y)
                         SELECT a.x, b.y FROM a CROSS JOIN b;""")
        assert r.rows[0] == [1, 2]


class TestExplain:
    def test_explain_select(self):
        e = E()
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (1);")
        r = e.execute("EXPLAIN SELECT * FROM t WHERE id > 0;")
        assert r.row_count > 0
        plan_text = '\n'.join(row[0] for row in r.rows)
        assert 'Scan' in plan_text or 'Filter' in plan_text or 'Project' in plan_text
        e.execute("DROP TABLE t;")

    def test_explain_no_from(self):
        e = E()
        r = e.execute("EXPLAIN SELECT 1 + 1;")
        assert r.row_count > 0


class TestFunctions:
    """Test a representative sample of all 142 functions."""
    def test_math(self):
        e = E()
        assert e.execute("SELECT ABS(-5);").rows[0][0] == 5
        assert e.execute("SELECT CEIL(3.2);").rows[0][0] == 4
        assert e.execute("SELECT FLOOR(3.8);").rows[0][0] == 3
        assert e.execute("SELECT ROUND(3.7);").rows[0][0] == 4
        assert e.execute("SELECT SIGN(-10);").rows[0][0] == -1
        assert e.execute("SELECT POWER(2, 10);").rows[0][0] == 1024.0
        assert abs(e.execute("SELECT SQRT(16);").rows[0][0] - 4.0) < 1e-10

    def test_string(self):
        e = E()
        assert e.execute("SELECT UPPER('hello');").rows[0][0] == 'HELLO'
        assert e.execute("SELECT LOWER('HELLO');").rows[0][0] == 'hello'
        assert e.execute("SELECT LENGTH('hello');").rows[0][0] == 5
        assert e.execute("SELECT TRIM('  hi  ');").rows[0][0] == 'hi'
        assert e.execute("SELECT REVERSE('abc');").rows[0][0] == 'cba'
        assert e.execute("SELECT REPLACE('hello', 'l', 'r');").rows[0][0] == 'herro'
        assert e.execute("SELECT SUBSTR('hello', 2, 3);").rows[0][0] == 'ell'

    def test_date(self):
        e = E()
        r = e.execute("SELECT YEAR(0), MONTH(0), DAY(0);")
        assert r.rows[0] == [1970, 1, 1]
        r = e.execute("SELECT CURRENT_DATE;")
        assert r.rows[0][0] is not None

    def test_conditional(self):
        e = E()
        r = e.execute("SELECT COALESCE(NULL, NULL, 'hello');")
        assert r.rows[0][0] == 'hello'
        r = e.execute("SELECT NULLIF(1, 1);")
        assert r.rows[0][0] is None
        r = e.execute("SELECT TYPEOF(42);")
        assert r.rows[0][0] == 'INT'

    def test_array(self):
        e = E()
        assert e.execute("SELECT ARRAY_LENGTH('[1,2,3]');").rows[0][0] == 3
        assert e.execute("SELECT ARRAY_JOIN('[1,2,3]', '-');").rows[0][0] == '1-2-3'

    def test_similarity(self):
        e = E()
        r = e.execute("SELECT JACCARD_SIMILARITY('[1,2,3]', '[2,3,4]');")
        assert 0.3 < r.rows[0][0] < 0.6
        r = e.execute("SELECT COSINE_SIMILARITY('[1,0]', '[0,1]');")
        assert abs(r.rows[0][0]) < 0.01


class TestEdgeCases:
    def test_limit_zero(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(2);")
        r = e.execute("SELECT * FROM t LIMIT 0;")
        assert r.row_count == 0
        e.execute("DROP TABLE t;")

    def test_offset_beyond(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (1),(2);")
        r = e.execute("SELECT * FROM t LIMIT 10 OFFSET 100;")
        assert r.row_count == 0
        e.execute("DROP TABLE t;")

    def test_null_arithmetic(self):
        e = E()
        r = e.execute("SELECT 1 + NULL;")
        assert r.rows[0][0] is None

    def test_null_comparison(self):
        e = E()
        r = e.execute("SELECT NULL = NULL;")
        assert r.rows[0][0] is None

    def test_null_and_or(self):
        e = E()
        assert e.execute("SELECT NULL AND FALSE;").rows[0][0] == False
        assert e.execute("SELECT NULL OR TRUE;").rows[0][0] == True

    def test_division_by_zero_int(self):
        e = E()
        from utils.errors import DivisionByZeroError
        try:
            e.execute("SELECT 1 / 0;")
            assert False, "should raise"
        except DivisionByZeroError:
            pass

    def test_string_concat(self):
        e = E()
        r = e.execute("SELECT 'hello' || ' ' || 'world';")
        assert r.rows[0][0] == 'hello world'

    def test_expression_in_order_by(self):
        e = E()
        e.execute("CREATE TABLE t (x INT);")
        e.execute("INSERT INTO t VALUES (3),(1),(2);")
        r = e.execute("SELECT x, x * 2 AS double FROM t ORDER BY double;")
        assert [row[1] for row in r.rows] == [2, 4, 6]
        e.execute("DROP TABLE t;")

    def test_subquery_in_where(self):
        e = E()
        e.execute("CREATE TABLE a (id INT);")
        e.execute("CREATE TABLE b (aid INT);")
        e.execute("INSERT INTO a VALUES (1),(2),(3);")
        e.execute("INSERT INTO b VALUES (2),(3);")
        r = e.execute("SELECT id FROM a WHERE id IN (SELECT aid FROM b) ORDER BY id;")
        assert [row[0] for row in r.rows] == [2, 3]
        e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")


def run_sql_tests():
    classes = [TestDDL, TestDML, TestSelect, TestJoin, TestAggregate,
               TestWindow, TestSetOps, TestCTE, TestExplain,
               TestFunctions, TestEdgeCases]
    total = 0; passed = 0; failed = 0
    for cls in classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith('test_'): continue
            total += 1
            try:
                getattr(obj, name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                import traceback
                print(f"  ✗ {cls.__name__}.{name}: {e}")
    print(f"\nSQL: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== SQL End-to-End Tests ===")
    run_sql_tests()
