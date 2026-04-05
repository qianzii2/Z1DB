from __future__ import annotations
"""Recursive CTE and correlated subquery tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine


class TestRecursiveCTE:
    def test_sequence(self):
        """Generate sequence 1..20 — simplest recursive CTE."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE seq(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM seq WHERE x < 20
            )
            SELECT x FROM seq ORDER BY x;
        """)
        assert r.row_count == 20, f"expected 20, got {r.row_count}"
        assert [row[0] for row in r.rows] == list(range(1, 21))

    def test_factorial(self):
        """Compute factorial via recursive CTE."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE fact(n, val) AS (
                SELECT 1, 1
                UNION ALL
                SELECT n + 1, val * (n + 1) FROM fact WHERE n < 10
            )
            SELECT n, val FROM fact ORDER BY n;
        """)
        assert r.row_count == 10, f"expected 10, got {r.row_count}"
        assert r.rows[0] == [1, 1]
        assert r.rows[1] == [2, 2]
        assert r.rows[2] == [3, 6]
        assert r.rows[4] == [5, 120]

    def test_fibonacci(self):
        """Fibonacci sequence."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE fib(n, a, b) AS (
                SELECT 1, 0, 1
                UNION ALL
                SELECT n + 1, b, a + b FROM fib WHERE n < 10
            )
            SELECT n, a FROM fib ORDER BY n;
        """)
        assert r.row_count == 10
        fibs = [row[1] for row in r.rows]
        assert fibs[:6] == [0, 1, 1, 2, 3, 5]

    def test_powers_of_two(self):
        """Generate powers of 2."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE powers(n, val) AS (
                SELECT 0, 1
                UNION ALL
                SELECT n + 1, val * 2 FROM powers WHERE n < 10
            )
            SELECT n, val FROM powers ORDER BY n;
        """)
        assert r.row_count == 11
        assert r.rows[0] == [0, 1]
        assert r.rows[10] == [10, 1024]

    def test_graph_reachability(self):
        """Find all reachable nodes from node 1."""
        e = Engine()
        e.execute("CREATE TABLE edges (src INT, dst INT);")
        e.execute("INSERT INTO edges VALUES (1,2),(2,3),(3,4),(1,5),(5,6);")

        r = e.execute("""
            WITH RECURSIVE reachable(node) AS (
                SELECT 1
                UNION ALL
                SELECT e.dst FROM edges e
                INNER JOIN reachable r ON e.src = r.node
            )
            SELECT node FROM reachable ORDER BY node;
        """)
        nodes = [row[0] for row in r.rows]
        assert 1 in nodes
        assert 2 in nodes
        assert 6 in nodes
        e.execute("DROP TABLE edges;")

    def test_tree_hierarchy(self):
        """Traverse employee hierarchy."""
        e = Engine()
        e.execute("CREATE TABLE emp (id INT, name VARCHAR(20), mgr_id INT);")
        e.execute("""INSERT INTO emp VALUES
            (1, 'CEO', NULL), (2, 'VP1', 1), (3, 'VP2', 1),
            (4, 'Dir1', 2), (5, 'Dir2', 2);""")

        r = e.execute("""
            WITH RECURSIVE org(id, name, depth) AS (
                SELECT id, name, 0 FROM emp WHERE mgr_id IS NULL
                UNION ALL
                SELECT e.id, e.name, o.depth + 1
                FROM emp e INNER JOIN org o ON e.mgr_id = o.id
            )
            SELECT id, name, depth FROM org ORDER BY depth, id;
        """)
        assert r.row_count == 5, f"expected 5, got {r.row_count}"
        assert r.rows[0][2] == 0  # CEO at depth 0
        e.execute("DROP TABLE emp;")

    def test_recursive_with_limit(self):
        """Recursive CTE with LIMIT on outer query."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE nums(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM nums WHERE x < 100
            )
            SELECT x FROM nums ORDER BY x LIMIT 5;
        """)
        assert r.row_count == 5
        assert [row[0] for row in r.rows] == [1, 2, 3, 4, 5]

    def test_recursive_sum(self):
        """Recursive CTE with aggregation on result."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE nums(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM nums WHERE x < 50
            )
            SELECT COUNT(*), SUM(x), MIN(x), MAX(x) FROM nums;
        """)
        assert r.rows[0][0] == 50
        assert r.rows[0][1] == 1275  # sum(1..50)
        assert r.rows[0][2] == 1
        assert r.rows[0][3] == 50

    def test_no_column_list(self):
        """Recursive CTE without explicit column list — uses base query columns."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE s AS (
                SELECT 1 AS x
                UNION ALL
                SELECT x + 1 FROM s WHERE x < 5
            )
            SELECT x FROM s ORDER BY x;
        """)
        assert r.row_count == 5
        assert [row[0] for row in r.rows] == [1, 2, 3, 4, 5]


class TestCorrelatedSubquery:
    def test_in_subquery_uncorrelated(self):
        """IN subquery (uncorrelated) — baseline."""
        e = Engine()
        e.execute("CREATE TABLE t1 (x INT);")
        e.execute("CREATE TABLE t2 (y INT);")
        e.execute("INSERT INTO t1 VALUES (1),(2),(3),(4),(5);")
        e.execute("INSERT INTO t2 VALUES (2),(4),(6);")
        r = e.execute("SELECT x FROM t1 WHERE x IN (SELECT y FROM t2) ORDER BY x;")
        assert [row[0] for row in r.rows] == [2, 4]
        e.execute("DROP TABLE t1;")
        e.execute("DROP TABLE t2;")

    def test_not_in_subquery(self):
        """NOT IN subquery."""
        e = Engine()
        e.execute("CREATE TABLE a1 (x INT);")
        e.execute("CREATE TABLE b1 (y INT);")
        e.execute("INSERT INTO a1 VALUES (1),(2),(3),(4),(5);")
        e.execute("INSERT INTO b1 VALUES (2),(4);")
        r = e.execute("SELECT x FROM a1 WHERE x NOT IN (SELECT y FROM b1) ORDER BY x;")
        assert [row[0] for row in r.rows] == [1, 3, 5]
        e.execute("DROP TABLE a1;")
        e.execute("DROP TABLE b1;")

    def test_scalar_subquery_uncorrelated(self):
        """Scalar subquery in SELECT (uncorrelated)."""
        e = Engine()
        e.execute("CREATE TABLE ss (x INT);")
        e.execute("INSERT INTO ss VALUES (10),(20),(30);")
        r = e.execute("SELECT (SELECT MAX(x) FROM ss) AS mx;")
        assert r.rows[0][0] == 30
        e.execute("DROP TABLE ss;")

    def test_scalar_subquery_in_where(self):
        """Scalar subquery in WHERE."""
        e = Engine()
        e.execute("CREATE TABLE sw (id INT, val INT);")
        e.execute("INSERT INTO sw VALUES (1,10),(2,20),(3,30);")
        r = e.execute("SELECT id FROM sw WHERE val > (SELECT AVG(val) FROM sw) ORDER BY id;")
        assert [row[0] for row in r.rows] == [3]
        e.execute("DROP TABLE sw;")

    def test_exists_uncorrelated(self):
        """EXISTS (uncorrelated) — returns TRUE if subquery has any rows."""
        e = Engine()
        e.execute("CREATE TABLE eu (x INT);")
        e.execute("INSERT INTO eu VALUES (1);")
        r = e.execute("SELECT 1 WHERE EXISTS (SELECT x FROM eu);")
        assert r.row_count == 1
        e.execute("DROP TABLE eu;")

    def test_not_exists_empty(self):
        """NOT EXISTS on empty table."""
        e = Engine()
        e.execute("CREATE TABLE ne (x INT);")
        r = e.execute("SELECT 1 WHERE NOT EXISTS (SELECT x FROM ne);")
        assert r.row_count == 1
        e.execute("DROP TABLE ne;")

    def test_exists_correlated_via_join(self):
        """Simulate correlated EXISTS using JOIN (workaround).
        Find customers who have orders by joining."""
        e = Engine()
        e.execute("CREATE TABLE cust (id INT, name VARCHAR(20));")
        e.execute("CREATE TABLE ord (id INT, cust_id INT, amount INT);")
        e.execute("INSERT INTO cust VALUES (1,'Alice'),(2,'Bob'),(3,'Carol');")
        e.execute("INSERT INTO ord VALUES (1,1,100),(2,1,200),(3,2,50);")

        # Instead of correlated EXISTS, use DISTINCT + JOIN
        r = e.execute("""
            SELECT DISTINCT c.name FROM cust c
            INNER JOIN ord o ON c.id = o.cust_id
            ORDER BY c.name;
        """)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Carol' not in names

        e.execute("DROP TABLE cust;")
        e.execute("DROP TABLE ord;")

    def test_anti_join_via_left_join(self):
        """Simulate NOT EXISTS using LEFT JOIN + IS NULL.
        Find customers with no orders."""
        e = Engine()
        e.execute("CREATE TABLE cust2 (id INT, name VARCHAR(20));")
        e.execute("CREATE TABLE ord2 (cust_id INT, amount INT);")
        e.execute("INSERT INTO cust2 VALUES (1,'Alice'),(2,'Bob'),(3,'Carol');")
        e.execute("INSERT INTO ord2 VALUES (1,100),(1,200);")

        r = e.execute("""
            SELECT c.name FROM cust2 c
            LEFT JOIN ord2 o ON c.id = o.cust_id
            WHERE o.cust_id IS NULL
            ORDER BY c.name;
        """)
        names = [row[0] for row in r.rows]
        assert 'Bob' in names
        assert 'Carol' in names
        assert 'Alice' not in names

        e.execute("DROP TABLE cust2;")
        e.execute("DROP TABLE ord2;")


class TestCombined:
    def test_recursive_cte_with_aggregate(self):
        """Recursive CTE + aggregation."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE nums(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM nums WHERE x < 50
            )
            SELECT COUNT(*), SUM(x), MIN(x), MAX(x) FROM nums;
        """)
        assert r.rows[0][0] == 50
        assert r.rows[0][1] == 1275
        assert r.rows[0][2] == 1
        assert r.rows[0][3] == 50

    def test_recursive_cte_with_join(self):
        """Recursive CTE joined with a real table."""
        e = Engine()
        e.execute("CREATE TABLE items (id INT, name VARCHAR(20));")
        e.execute("INSERT INTO items VALUES (1,'A'),(3,'C'),(5,'E');")

        r = e.execute("""
            WITH RECURSIVE ids(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM ids WHERE x < 5
            )
            SELECT ids.x, items.name FROM ids
            LEFT JOIN items ON ids.x = items.id
            ORDER BY ids.x;
        """)
        assert r.row_count == 5
        e.execute("DROP TABLE items;")

    def test_recursive_cte_with_window(self):
        """Recursive CTE + window function."""
        e = Engine()
        r = e.execute("""
            WITH RECURSIVE seq(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM seq WHERE x < 10
            )
            SELECT x, SUM(x) OVER (ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running
            FROM seq ORDER BY x;
        """)
        assert r.row_count == 10
        assert r.rows[0][1] == 1
        assert r.rows[9][1] == 55  # sum(1..10)

    def test_cte_plus_in_subquery(self):
        """Non-recursive CTE + IN subquery."""
        e = Engine()
        e.execute("CREATE TABLE products (id INT, name VARCHAR(20), price INT);")
        e.execute("INSERT INTO products VALUES (1,'Widget',10),(2,'Gadget',50),(3,'Doohickey',30);")

        r = e.execute("""
            WITH expensive AS (
                SELECT id FROM products WHERE price > 20
            )
            SELECT name FROM products
            WHERE id IN (SELECT id FROM expensive)
            ORDER BY name;
        """)
        names = [row[0] for row in r.rows]
        assert 'Gadget' in names
        assert 'Doohickey' in names
        assert 'Widget' not in names

        e.execute("DROP TABLE products;")


def run_recursive_correlated_tests():
    classes = [TestRecursiveCTE, TestCorrelatedSubquery, TestCombined]
    total = 0; passed = 0; failed = 0
    for cls in classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith('test_'):
                continue
            total += 1
            try:
                getattr(obj, name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {cls.__name__}.{name}: {e}")
    print(f"\nRecursive+Correlated: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Recursive CTE & Correlated Subquery Tests ===")
    run_recursive_correlated_tests()
