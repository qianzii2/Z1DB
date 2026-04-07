"""Z1DB 扩展测试套件 — 覆盖更多边界条件、组件单测、回归测试。
运行: pytest tests/test_z1db_extended.py -v
"""
from __future__ import annotations
import os, sys, math, tempfile, threading, struct

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import pytest
from engine import Engine
from utils.errors import (
    Z1Error, ParseError, SemanticError, ExecutionError,
    TableNotFoundError, ColumnNotFoundError, DuplicateError,
    DivisionByZeroError, NumericOverflowError,
)


def scalar(result):
    assert result.rows, "结果为空"
    return result.rows[0][0]

def col_values(result, idx=0):
    return [r[idx] for r in result.rows]

@pytest.fixture
def eng():
    return Engine(':memory:')

@pytest.fixture
def populated(eng):
    eng.execute("CREATE TABLE users (id INT, name VARCHAR, age INT);")
    eng.execute("INSERT INTO users VALUES (1, 'Alice', 30);")
    eng.execute("INSERT INTO users VALUES (2, 'Bob', 25);")
    eng.execute("INSERT INTO users VALUES (3, 'Charlie', 35);")
    eng.execute("INSERT INTO users VALUES (4, 'Diana', 28);")
    eng.execute("INSERT INTO users VALUES (5, 'Eve', 30);")
    return eng

@pytest.fixture
def multi_table(eng):
    eng.execute("CREATE TABLE dept (id INT, name VARCHAR);")
    eng.execute("INSERT INTO dept VALUES (1, 'Engineering');")
    eng.execute("INSERT INTO dept VALUES (2, 'Sales');")
    eng.execute("CREATE TABLE emp (id INT, name VARCHAR, dept_id INT, salary INT);")
    eng.execute("INSERT INTO emp VALUES (1,'Alice',1,80000);")
    eng.execute("INSERT INTO emp VALUES (2,'Bob',2,60000);")
    eng.execute("INSERT INTO emp VALUES (3,'Charlie',1,90000);")
    eng.execute("INSERT INTO emp VALUES (4,'Diana',2,70000);")
    eng.execute("INSERT INTO emp VALUES (5,'Eve',1,85000);")
    return eng


# ═══════════════════════════════════════════
#  33. 更多数组函数
# ═══════════════════════════════════════════

class TestArrayFunctionsExtended:
    def test_array_sort(self, eng):
        r = eng.execute("SELECT ARRAY_SORT('[3,1,2]');")
        val = scalar(r)
        assert '1' in val and '2' in val and '3' in val

    def test_array_reverse(self, eng):
        r = eng.execute("SELECT ARRAY_REVERSE('[1,2,3]');")
        val = scalar(r)
        assert val.startswith('[3')

    def test_array_distinct(self, eng):
        r = eng.execute("SELECT ARRAY_DISTINCT('[1,2,2,3,3,3]');")
        val = scalar(r)
        assert '1' in val and '2' in val and '3' in val

    def test_array_append(self, eng):
        r = eng.execute("SELECT ARRAY_APPEND('[1,2]', 3);")
        val = scalar(r)
        assert '3' in val

    def test_array_prepend(self, eng):
        r = eng.execute("SELECT ARRAY_PREPEND('[2,3]', 1);")
        val = scalar(r)
        assert val.startswith('[1')

    def test_array_remove(self, eng):
        r = eng.execute("SELECT ARRAY_REMOVE('[1,2,3,2]', 2);")
        val = scalar(r)
        assert '2' not in val

    def test_array_concat(self, eng):
        r = eng.execute("SELECT ARRAY_CONCAT('[1,2]', '[3,4]');")
        val = scalar(r)
        for d in ['1', '2', '3', '4']:
            assert d in val

    def test_array_slice(self, eng):
        r = eng.execute("SELECT ARRAY_SLICE('[10,20,30,40,50]', 2, 4);")
        val = scalar(r)
        assert '20' in val and '30' in val

    def test_array_length_empty(self, eng):
        r = eng.execute("SELECT ARRAY_LENGTH('[]');")
        assert scalar(r) == 0

    def test_array_contains_false(self, eng):
        r = eng.execute("SELECT ARRAY_CONTAINS('[1,2,3]', 99);")
        assert scalar(r) is False

    def test_array_position_missing(self, eng):
        r = eng.execute("SELECT ARRAY_POSITION('[1,2,3]', 99);")
        assert scalar(r) == 0

    def test_array_intersect(self, eng):
        r = eng.execute("SELECT ARRAY_INTERSECT('[1,2,3]', '[2,3,4]');")
        val = scalar(r)
        assert '2' in val and '3' in val

    def test_array_union(self, eng):
        r = eng.execute("SELECT ARRAY_UNION('[1,2]', '[2,3]');")
        val = scalar(r)
        for d in ['1', '2', '3']:
            assert d in val

    def test_array_except(self, eng):
        r = eng.execute("SELECT ARRAY_EXCEPT('[1,2,3]', '[2,3]');")
        val = scalar(r)
        assert '1' in val


# ═══════════════════════════════════════════
#  34. 更多日期/时间
# ═══════════════════════════════════════════

class TestDatetimeExtended:
    def test_month_extraction(self, eng):
        r = eng.execute("SELECT MONTH(TO_DATE('2024-07-15'));")
        assert scalar(r) == 7

    def test_day_extraction(self, eng):
        r = eng.execute("SELECT DAY(TO_DATE('2024-07-15'));")
        assert scalar(r) == 15

    def test_quarter(self, eng):
        r = eng.execute("SELECT QUARTER(TO_DATE('2024-07-15'));")
        assert scalar(r) == 3

    def test_date_add(self, eng):
        r = eng.execute(
            "SELECT DATE_ADD(TO_DATE('2024-01-01'), 31);")
        val = scalar(r)
        assert val is not None

    def test_date_sub(self, eng):
        r = eng.execute(
            "SELECT DATE_SUB(TO_DATE('2024-02-01'), 1);")
        val = scalar(r)
        assert val is not None

    def test_date_trunc_year(self, eng):
        r = eng.execute(
            "SELECT DATE_TRUNC('YEAR', TO_DATE('2024-07-15'));")
        assert scalar(r) is not None

    def test_date_trunc_month(self, eng):
        r = eng.execute(
            "SELECT DATE_TRUNC('MONTH', TO_DATE('2024-07-15'));")
        assert scalar(r) is not None

    def test_to_timestamp(self, eng):
        r = eng.execute("SELECT TO_TIMESTAMP('2024-01-15 10:30:00');")
        assert scalar(r) is not None

    def test_date_column_operations(self, eng):
        eng.execute("CREATE TABLE events (id INT, d DATE);")
        eng.execute("INSERT INTO events VALUES (1, CAST('2024-01-15' AS DATE));")
        eng.execute("INSERT INTO events VALUES (2, CAST('2024-03-20' AS DATE));")
        r = eng.execute("SELECT id FROM events WHERE d > TO_DATE('2024-02-01');")
        assert col_values(r) == [2]

    def test_date_format(self, eng):
        r = eng.execute(
            "SELECT DATE_FORMAT(TO_DATE('2024-07-15'), 'YYYY-MM-DD');")
        assert '2024' in scalar(r)


# ═══════════════════════════════════════════
#  35. 更多字符串函数
# ═══════════════════════════════════════════

class TestStringFunctionsExtended:
    def test_regexp_replace(self, eng):
        r = eng.execute(
            "SELECT REGEXP_REPLACE('hello 123 world', '[0-9]+', 'NUM');")
        assert scalar(r) == 'hello NUM world'

    def test_regexp_match(self, eng):
        r = eng.execute("SELECT REGEXP_MATCH('hello123', '[0-9]+');")
        assert scalar(r) is True

    def test_regexp_match_false(self, eng):
        r = eng.execute("SELECT REGEXP_MATCH('hello', '[0-9]+');")
        assert scalar(r) is False

    def test_regexp_extract(self, eng):
        r = eng.execute("SELECT REGEXP_EXTRACT('price: 42.5', '[0-9.]+');")
        assert '42' in scalar(r)

    def test_lpad_default_pad(self, eng):
        r = eng.execute("SELECT LPAD('hi', 5);")
        assert scalar(r) == '   hi'

    def test_rpad_default_pad(self, eng):
        r = eng.execute("SELECT RPAD('hi', 5);")
        assert scalar(r) == 'hi   '

    def test_length_unicode(self, eng):
        r = eng.execute("SELECT LENGTH('café');")
        assert scalar(r) == 4

    def test_upper_with_column(self, populated):
        r = populated.execute(
            "SELECT UPPER(name) FROM users WHERE id = 1;")
        assert scalar(r) == 'ALICE'

    def test_lower_with_column(self, populated):
        r = populated.execute(
            "SELECT LOWER(name) FROM users WHERE id = 1;")
        assert scalar(r) == 'alice'

    def test_concat_with_null(self, eng):
        r = eng.execute("SELECT CONCAT('a', NULL, 'b');")
        # CONCAT 跳过 NULL
        assert scalar(r) == 'ab'

    def test_concat_ws_with_null(self, eng):
        r = eng.execute("SELECT CONCAT_WS('-', 'a', NULL, 'b');")
        # CONCAT_WS 跳过 NULL
        assert scalar(r) == 'a-b'

    def test_starts_with_false(self, eng):
        r = eng.execute("SELECT STARTS_WITH('hello', 'xyz');")
        assert scalar(r) is False

    def test_ends_with_false(self, eng):
        r = eng.execute("SELECT ENDS_WITH('hello', 'xyz');")
        assert scalar(r) is False

    def test_contains_false(self, eng):
        r = eng.execute("SELECT CONTAINS('hello', 'xyz');")
        assert scalar(r) is False

    def test_like_escape_percent(self, eng):
        eng.execute("CREATE TABLE t (v VARCHAR);")
        eng.execute("INSERT INTO t VALUES ('100%');")
        eng.execute("INSERT INTO t VALUES ('abc');")
        r = eng.execute("SELECT v FROM t WHERE v LIKE '%100%';")
        assert scalar(r) == '100%'


# ═══════════════════════════════════════════
#  36. 更多数学函数
# ═══════════════════════════════════════════

class TestMathFunctionsExtended:
    def test_cbrt(self, eng):
        r = eng.execute("SELECT CBRT(27);")
        assert abs(scalar(r) - 3.0) < 0.001

    def test_ln(self, eng):
        r = eng.execute("SELECT LN(1);")
        assert abs(scalar(r)) < 0.001

    def test_log2(self, eng):
        r = eng.execute("SELECT LOG2(8);")
        assert abs(scalar(r) - 3.0) < 0.001

    def test_log_base(self, eng):
        r = eng.execute("SELECT LOG(10, 1000);")
        assert abs(scalar(r) - 3.0) < 0.001

    def test_exp(self, eng):
        r = eng.execute("SELECT EXP(0);")
        assert abs(scalar(r) - 1.0) < 0.001

    def test_trunc(self, eng):
        r = eng.execute("SELECT TRUNC(3.99);")
        assert scalar(r) == 3

    def test_trunc_negative(self, eng):
        r = eng.execute("SELECT TRUNC(-3.99);")
        assert scalar(r) == -3

    def test_width_bucket(self, eng):
        r = eng.execute("SELECT WIDTH_BUCKET(5.5, 0, 10, 4);")
        # 5.5 in [0,10) with 4 buckets → bucket 3
        assert scalar(r) == 3

    def test_width_bucket_below(self, eng):
        r = eng.execute("SELECT WIDTH_BUCKET(-1, 0, 10, 4);")
        assert scalar(r) == 0

    def test_width_bucket_above(self, eng):
        r = eng.execute("SELECT WIDTH_BUCKET(11, 0, 10, 4);")
        assert scalar(r) == 5

    def test_abs_float(self, eng):
        r = eng.execute("SELECT ABS(-3.14);")
        assert abs(scalar(r) - 3.14) < 0.001

    def test_ceil_negative(self, eng):
        r = eng.execute("SELECT CEIL(-3.2);")
        assert scalar(r) == -3

    def test_floor_negative(self, eng):
        r = eng.execute("SELECT FLOOR(-3.2);")
        assert scalar(r) == -4

    def test_power_fractional(self, eng):
        r = eng.execute("SELECT POWER(4, 0.5);")
        assert abs(scalar(r) - 2.0) < 0.001

    def test_random(self, eng):
        r = eng.execute("SELECT RANDOM();")
        val = scalar(r)
        assert 0 <= val <= 1

    def test_sign_float(self, eng):
        r = eng.execute("SELECT SIGN(-0.5);")
        assert scalar(r) == -1


# ═══════════════════════════════════════════
#  37. 更多窗口函数
# ═══════════════════════════════════════════

class TestWindowExtended:
    def test_count_window(self, populated):
        r = populated.execute(
            "SELECT name, COUNT(*) OVER () AS total FROM users;")
        totals = col_values(r, 1)
        assert all(t == 5 for t in totals)

    def test_min_max_window(self, populated):
        r = populated.execute(
            "SELECT name, MIN(age) OVER () AS mn, "
            "MAX(age) OVER () AS mx FROM users LIMIT 1;")
        assert r.rows[0][1] == 25
        assert r.rows[0][2] == 35

    def test_percent_rank(self, populated):
        r = populated.execute(
            "SELECT name, PERCENT_RANK() OVER (ORDER BY age) AS pr "
            "FROM users;")
        prs = col_values(r, 1)
        assert min(prs) == 0.0
        assert max(prs) <= 1.0

    def test_cume_dist(self, populated):
        r = populated.execute(
            "SELECT name, CUME_DIST() OVER (ORDER BY age) AS cd "
            "FROM users;")
        cds = col_values(r, 1)
        assert max(cds) == 1.0

    def test_last_value(self, populated):
        r = populated.execute(
            "SELECT name, LAST_VALUE(name) OVER ("
            "ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND UNBOUNDED FOLLOWING) AS lv FROM users;")
        lvs = col_values(r, 1)
        assert all(v == 'Eve' for v in lvs)

    def test_window_multiple_partitions(self, multi_table):
        r = multi_table.execute(
            "SELECT name, dept_id, "
            "SUM(salary) OVER (PARTITION BY dept_id) AS dept_total "
            "FROM emp ORDER BY dept_id, name;")
        # All Engineering rows should have same dept_total
        eng_totals = [row[2] for row in r.rows if row[1] == 1]
        assert len(set(eng_totals)) == 1

    def test_lag_with_default(self, populated):
        r = populated.execute(
            "SELECT name, LAG(name, 1, 'NONE') OVER (ORDER BY id) AS prev "
            "FROM users ORDER BY id;")
        assert r.rows[0][1] == 'NONE'

    def test_lead_offset_2(self, populated):
        r = populated.execute(
            "SELECT name, LEAD(name, 2) OVER (ORDER BY id) AS nxt2 "
            "FROM users ORDER BY id;")
        assert r.rows[0][1] == 'Charlie'
        assert r.rows[3][1] is None


# ═══════════════════════════════════════════
#  38. 更多 JOIN 测试
# ═══════════════════════════════════════════

class TestJoinExtended:
    def test_full_outer_join(self, multi_table):
        multi_table.execute("INSERT INTO dept VALUES (3, 'HR');")
        multi_table.execute("INSERT INTO emp VALUES (6,'Frank',99,50000);")
        r = multi_table.execute(
            "SELECT e.name, d.name "
            "FROM emp e FULL JOIN dept d ON e.dept_id = d.id "
            "ORDER BY e.id;")
        # 应有 Frank(无部门) 和 HR(无员工)
        names_e = col_values(r, 0)
        names_d = col_values(r, 1)
        assert None in names_e or None in names_d

    def test_join_on_expression(self, eng):
        eng.execute("CREATE TABLE a (id INT, x INT);")
        eng.execute("CREATE TABLE b (id INT, y INT);")
        eng.execute("INSERT INTO a VALUES (1, 10), (2, 20);")
        eng.execute("INSERT INTO b VALUES (1, 10), (2, 30);")
        r = eng.execute(
            "SELECT a.id FROM a JOIN b ON a.x = b.y;")
        assert col_values(r) == [1]

    def test_join_multiple_conditions(self, eng):
        eng.execute("CREATE TABLE a (x INT, y INT);")
        eng.execute("CREATE TABLE b (x INT, y INT);")
        eng.execute("INSERT INTO a VALUES (1, 1), (1, 2), (2, 1);")
        eng.execute("INSERT INTO b VALUES (1, 1), (1, 3), (2, 2);")
        r = eng.execute(
            "SELECT a.x, a.y FROM a "
            "JOIN b ON a.x = b.x AND a.y = b.y;")
        assert r.rows == [[1, 1]]

    def test_left_join_preserves_left(self, eng):
        eng.execute("CREATE TABLE a (id INT);")
        eng.execute("CREATE TABLE b (id INT, val INT);")
        eng.execute("INSERT INTO a VALUES (1),(2),(3);")
        eng.execute("INSERT INTO b VALUES (1, 10);")
        r = eng.execute(
            "SELECT a.id, b.val FROM a "
            "LEFT JOIN b ON a.id = b.id ORDER BY a.id;")
        assert r.row_count == 3
        assert r.rows[0] == [1, 10]
        assert r.rows[1][1] is None
        assert r.rows[2][1] is None

    def test_join_empty_table(self, eng):
        eng.execute("CREATE TABLE a (id INT);")
        eng.execute("CREATE TABLE b (id INT);")
        eng.execute("INSERT INTO a VALUES (1),(2);")
        r = eng.execute(
            "SELECT a.id FROM a JOIN b ON a.id = b.id;")
        assert r.row_count == 0

    def test_left_join_empty_right(self, eng):
        eng.execute("CREATE TABLE a (id INT);")
        eng.execute("CREATE TABLE b (id INT);")
        eng.execute("INSERT INTO a VALUES (1),(2);")
        r = eng.execute(
            "SELECT a.id, b.id FROM a "
            "LEFT JOIN b ON a.id = b.id ORDER BY a.id;")
        assert r.row_count == 2
        assert all(row[1] is None for row in r.rows)


# ═══════════════════════════════════════════
#  39. 更多 CTE 测试
# ═══════════════════════════════════════════

class TestCTEExtended:
    def test_cte_used_twice(self, populated):
        r = populated.execute("""
            WITH young AS (SELECT * FROM users WHERE age < 30)
            SELECT
                (SELECT COUNT(*) FROM young) AS cnt,
                (SELECT MIN(age) FROM young) AS min_age;
        """)
        assert r.rows[0][0] == 2
        assert r.rows[0][1] == 25

    def test_recursive_cte_large(self, eng):
        r = eng.execute("""
            WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM nums WHERE n < 50
            ) SELECT COUNT(*) FROM nums;
        """)
        assert scalar(r) == 50

    def test_recursive_cte_union_distinct(self, eng):
        """UNION (非 ALL) 应去重。"""
        r = eng.execute("""
            WITH RECURSIVE t(x) AS (
                SELECT 1
                UNION
                SELECT 1 FROM t WHERE x < 3
            ) SELECT COUNT(*) FROM t;
        """)
        assert scalar(r) == 1

    def test_cte_with_join(self, multi_table):
        r = multi_table.execute("""
            WITH eng_dept AS (
                SELECT id FROM dept WHERE name = 'Engineering'
            )
            SELECT e.name FROM emp e
            JOIN eng_dept ed ON e.dept_id = ed.id
            ORDER BY e.name;
        """)
        assert set(col_values(r)) == {'Alice', 'Charlie', 'Eve'}


# ═══════════════════════════════════════════
#  40. 更多 SET 操作
# ═══════════════════════════════════════════

class TestSetOpsExtended:
    def test_except_all(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(1),(2),(3);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (1),(3);")
        r = eng.execute(
            "SELECT v FROM a EXCEPT ALL SELECT v FROM b ORDER BY v;")
        assert col_values(r) == [1, 2]

    def test_union_different_column_names(self, eng):
        eng.execute("CREATE TABLE a (x INT);")
        eng.execute("INSERT INTO a VALUES (1);")
        eng.execute("CREATE TABLE b (y INT);")
        eng.execute("INSERT INTO b VALUES (2);")
        r = eng.execute(
            "SELECT x FROM a UNION ALL SELECT y FROM b;")
        assert r.row_count == 2

    def test_union_with_null(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(NULL);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (NULL),(2);")
        r = eng.execute("SELECT v FROM a UNION SELECT v FROM b;")
        vals = col_values(r)
        assert None in vals
        # 两个 NULL 应合并为一个
        assert vals.count(None) == 1

    def test_intersect_empty(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(2);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (3),(4);")
        r = eng.execute(
            "SELECT v FROM a INTERSECT SELECT v FROM b;")
        assert r.row_count == 0


# ═══════════════════════════════════════════
#  41. 更多 NULL 边界
# ═══════════════════════════════════════════

class TestNullEdgeCases:
    def test_null_in_in_list(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1),(2),(NULL);")
        r = eng.execute("SELECT v FROM t WHERE v IN (1, NULL);")
        # v=1 匹配，v=NULL IN (...) 返回 NULL（不匹配），v=2 不匹配
        assert col_values(r) == [1]

    def test_null_not_in_list(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1),(2),(NULL);")
        r = eng.execute("SELECT v FROM t WHERE v NOT IN (1);")
        # v=2 匹配，v=NULL NOT IN (...) → NULL
        assert col_values(r) == [2]

    def test_null_between(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1),(NULL),(3);")
        r = eng.execute(
            "SELECT COUNT(*) FROM t WHERE v BETWEEN 1 AND 3;")
        assert scalar(r) == 2

    def test_null_like(self, eng):
        eng.execute("CREATE TABLE t (v VARCHAR);")
        eng.execute("INSERT INTO t VALUES ('abc'),(NULL),('def');")
        r = eng.execute(
            "SELECT COUNT(*) FROM t WHERE v LIKE '%b%';")
        assert scalar(r) == 1

    def test_null_order_mixed(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (3),(NULL),(1),(NULL),(2);")
        r = eng.execute("SELECT v FROM t ORDER BY v ASC NULLS LAST;")
        vals = col_values(r)
        assert vals[:3] == [1, 2, 3]
        assert vals[3] is None
        assert vals[4] is None

    def test_coalesce_all_null(self, eng):
        r = eng.execute("SELECT COALESCE(NULL, NULL, NULL);")
        assert scalar(r) is None

    def test_case_all_null_result(self, eng):
        r = eng.execute("""
            SELECT CASE WHEN FALSE THEN 1
                        WHEN FALSE THEN 2
                        END;
        """)
        assert scalar(r) is None

    def test_null_cast(self, eng):
        r = eng.execute("SELECT CAST(NULL AS INT);")
        assert scalar(r) is None


# ═══════════════════════════════════════════
#  42. 更多复杂查询
# ═══════════════════════════════════════════

class TestComplexExtended:
    def test_double_group_by_with_having(self, eng):
        eng.execute("CREATE TABLE sales (region VARCHAR, product VARCHAR, amount INT);")
        for r, p, a in [('N','A',10),('N','A',20),('N','B',30),
                         ('S','A',40),('S','B',50),('S','B',60)]:
            eng.execute(f"INSERT INTO sales VALUES ('{r}','{p}',{a});")
        r = eng.execute("""
            SELECT region, product, SUM(amount) AS total
            FROM sales GROUP BY region, product
            HAVING SUM(amount) >= 30
            ORDER BY total DESC;
        """)
        assert r.row_count >= 2
        assert r.rows[0][2] >= 30

    def test_subquery_in_select_list(self, populated):
        r = populated.execute("""
            SELECT name,
                   (SELECT AVG(age) FROM users) AS avg_age,
                   age - (SELECT AVG(age) FROM users) AS diff
            FROM users WHERE id = 3;
        """)
        assert r.rows[0][0] == 'Charlie'
        assert r.rows[0][1] is not None

    def test_deeply_nested_expression(self, eng):
        r = eng.execute("SELECT ((1 + 2) * 3 - 4) / 5 + 6;")
        # (3*3-4)/5+6 = 5/5+6 = 1+6 = 7
        assert scalar(r) == 7

    def test_case_in_where(self, populated):
        r = populated.execute("""
            SELECT name FROM users
            WHERE CASE WHEN age >= 30 THEN TRUE ELSE FALSE END = TRUE
            ORDER BY name;
        """)
        assert set(col_values(r)) == {'Alice', 'Charlie', 'Eve'}

    def test_function_in_where(self, populated):
        r = populated.execute(
            "SELECT name FROM users WHERE LENGTH(name) > 4 ORDER BY name;")
        assert 'Alice' in col_values(r)
        assert 'Charlie' in col_values(r)

    def test_function_in_order_by(self, populated):
        r = populated.execute(
            "SELECT name FROM users ORDER BY LENGTH(name) DESC LIMIT 1;")
        assert scalar(r) == 'Charlie'

    def test_agg_with_expression(self, populated):
        r = populated.execute("SELECT SUM(age * 2) FROM users;")
        assert scalar(r) == 296

    def test_group_by_with_function(self, populated):
        r = populated.execute(
            "SELECT LENGTH(name) AS nlen, COUNT(*) "
            "FROM users GROUP BY LENGTH(name) ORDER BY nlen;")
        assert r.row_count >= 2

    def test_multiple_subqueries_in_where(self, populated):
        r = populated.execute("""
            SELECT name FROM users
            WHERE age > (SELECT MIN(age) FROM users)
              AND age < (SELECT MAX(age) FROM users)
            ORDER BY name;
        """)
        names = col_values(r)
        assert 'Bob' not in names  # min
        assert 'Charlie' not in names  # max
        assert 'Diana' in names

    def test_insert_from_complex_select(self, eng):
        eng.execute("CREATE TABLE src (id INT, v INT);")
        eng.execute("INSERT INTO src VALUES (1,10),(2,20),(3,30);")
        eng.execute("CREATE TABLE dst (total INT);")
        eng.execute("INSERT INTO dst SELECT SUM(v) FROM src;")
        r = eng.execute("SELECT total FROM dst;")
        assert scalar(r) == 60


# ═══════════════════════════════════════════
#  43. 组件单测 — Bitmap
# ═══════════════════════════════════════════

class TestBitmap:
    def test_basic_ops(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(64)
        assert bm.popcount() == 0
        bm.set_bit(0)
        bm.set_bit(63)
        assert bm.get_bit(0)
        assert bm.get_bit(63)
        assert not bm.get_bit(1)
        assert bm.popcount() == 2

    def test_clear_bit(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(32)
        bm.set_bit(5)
        assert bm.get_bit(5)
        bm.clear_bit(5)
        assert not bm.get_bit(5)

    def test_to_indices(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(16)
        bm.set_bit(1)
        bm.set_bit(5)
        bm.set_bit(10)
        assert bm.to_indices() == [1, 5, 10]

    def test_and_op(self):
        from metal.bitmap import Bitmap
        a = Bitmap(8); b = Bitmap(8)
        a.set_bit(1); a.set_bit(2); a.set_bit(3)
        b.set_bit(2); b.set_bit(3); b.set_bit(4)
        r = a.and_op(b)
        assert r.to_indices() == [2, 3]

    def test_or_op(self):
        from metal.bitmap import Bitmap
        a = Bitmap(8); b = Bitmap(8)
        a.set_bit(1); a.set_bit(2)
        b.set_bit(3); b.set_bit(4)
        r = a.or_op(b)
        assert set(r.to_indices()) == {1, 2, 3, 4}

    def test_not_op(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(8)
        bm.set_bit(0)
        r = bm.not_op()
        assert not r.get_bit(0)
        assert r.get_bit(1)

    def test_empty_bitmap(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(0)
        assert bm.popcount() == 0
        assert bm.to_indices() == []

    def test_large_bitmap(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(1000)
        bm.set_bit(999)
        assert bm.get_bit(999)
        assert bm.popcount() == 1

    def test_pooled(self):
        from metal.bitmap import Bitmap
        bm = Bitmap.pooled(64)
        bm.set_bit(10)
        assert bm.get_bit(10)


# ═══════════════════════════════════════════
#  44. 组件单测 — TypedVector
# ═══════════════════════════════════════════

class TestTypedVector:
    def test_append_and_get(self):
        from metal.typed_vector import TypedVector
        tv = TypedVector('q')
        tv.append(10)
        tv.append(20)
        assert tv[0] == 10
        assert tv[1] == 20
        assert len(tv) == 2

    def test_initial_data(self):
        from metal.typed_vector import TypedVector
        tv = TypedVector('i', [1, 2, 3])
        assert len(tv) == 3
        assert tv[2] == 3

    def test_extend(self):
        from metal.typed_vector import TypedVector
        a = TypedVector('i', [1, 2])
        b = TypedVector('i', [3, 4])
        a.extend(b)
        assert len(a) == 4
        assert a[3] == 4

    def test_copy(self):
        from metal.typed_vector import TypedVector
        tv = TypedVector('i', [1, 2, 3])
        cp = tv.copy()
        cp.append(4)
        assert len(tv) == 3
        assert len(cp) == 4

    def test_to_list(self):
        from metal.typed_vector import TypedVector
        tv = TypedVector('d', [1.5, 2.5])
        lst = tv.to_list()
        assert lst == [1.5, 2.5]

    def test_filter_by_indices(self):
        from metal.typed_vector import TypedVector
        tv = TypedVector('i', [10, 20, 30, 40, 50])
        filtered = tv.filter_by_indices([1, 3])
        assert len(filtered) == 2
        assert filtered[0] == 20
        assert filtered[1] == 40


# ═══════════════════════════════════════════
#  45. 组件单测 — Hash
# ═══════════════════════════════════════════

class TestHash:
    def test_z1hash64_deterministic(self):
        from metal.hash import z1hash64
        h1 = z1hash64(b'hello')
        h2 = z1hash64(b'hello')
        assert h1 == h2

    def test_z1hash64_different_inputs(self):
        from metal.hash import z1hash64
        h1 = z1hash64(b'hello')
        h2 = z1hash64(b'world')
        assert h1 != h2

    def test_z1hash128(self):
        from metal.hash import z1hash128
        h1, h2 = z1hash128(b'test')
        assert isinstance(h1, int)
        assert isinstance(h2, int)
        assert h1 != h2

    def test_hash_value(self):
        from metal.hash import hash_value
        h_none = hash_value(None)
        assert h_none == 0
        h_int = hash_value(42)
        h_str = hash_value('hello')
        assert h_int != h_str

    def test_hash_combine(self):
        from metal.hash import hash_combine
        h = hash_combine(123, 456)
        assert isinstance(h, int)


# ═══════════════════════════════════════════
#  46. 组件单测 — NaN-Boxing
# ═══════════════════════════════════════════

class TestNanBoxing:
    def test_pack_unpack_int(self):
        from metal.bitmagic import nan_pack_int, nan_unpack
        packed = nan_pack_int(42)
        tag, val = nan_unpack(packed)
        assert tag == 'INT'
        assert val == 42

    def test_pack_unpack_negative(self):
        from metal.bitmagic import nan_pack_int, nan_unpack
        packed = nan_pack_int(-100)
        tag, val = nan_unpack(packed)
        assert val == -100

    def test_pack_unpack_float(self):
        from metal.bitmagic import nan_pack_float, nan_unpack
        packed = nan_pack_float(3.14)
        tag, val = nan_unpack(packed)
        assert tag == 'FLOAT'
        assert abs(val - 3.14) < 1e-10

    def test_null_tag(self):
        from metal.bitmagic import nan_pack_null, nan_is_null, NULL_TAG
        packed = nan_pack_null()
        assert packed == NULL_TAG
        assert nan_is_null(packed)

    def test_bool_tag(self):
        from metal.bitmagic import nan_pack_bool, nan_unpack
        packed_t = nan_pack_bool(True)
        packed_f = nan_pack_bool(False)
        _, vt = nan_unpack(packed_t)
        _, vf = nan_unpack(packed_f)
        assert vt is True
        assert vf is False


# ═══════════════════════════════════════════
#  47. 组件单测 — BloomFilter
# ═══════════════════════════════════════════

class TestBloomFilter:
    def test_basic(self):
        from structures.bloom_filter import BloomFilter
        bf = BloomFilter(100, 0.01)
        bf.add(42)
        bf.add('hello')
        assert bf.contains(42)
        assert bf.contains('hello')
        assert bf.count == 2

    def test_no_false_negative(self):
        from structures.bloom_filter import BloomFilter
        bf = BloomFilter(1000, 0.01)
        items = list(range(500))
        for i in items:
            bf.add(i)
        for i in items:
            assert bf.contains(i)

    def test_serialize(self):
        from structures.bloom_filter import BloomFilter
        bf = BloomFilter(100, 0.01)
        bf.add(1); bf.add(2); bf.add(3)
        data = bf.to_bytes()
        bf2 = BloomFilter.from_bytes(data)
        assert bf2.contains(1)
        assert bf2.contains(2)
        assert bf2.contains(3)


# ═══════════════════════════════════════════
#  48. 组件单测 — SkipList
# ═══════════════════════════════════════════

class TestSkipList:
    def test_basic(self):
        from structures.skip_list import SkipList
        sl = SkipList()
        sl.insert(3, 'c')
        sl.insert(1, 'a')
        sl.insert(2, 'b')
        assert sl.search(2) == 'b'
        assert sl.search(4) is None
        assert sl.size == 3

    def test_range_query(self):
        from structures.skip_list import SkipList
        sl = SkipList()
        for i in range(10):
            sl.insert(i, str(i))
        results = sl.range_query(3, 6)
        keys = [r[0] for r in results]
        assert keys == [3, 4, 5, 6]

    def test_delete(self):
        from structures.skip_list import SkipList
        sl = SkipList()
        sl.insert(1, 'a')
        sl.insert(2, 'b')
        assert sl.delete(1)
        assert sl.search(1) is None
        assert sl.size == 1

    def test_update(self):
        from structures.skip_list import SkipList
        sl = SkipList()
        sl.insert(1, 'old')
        sl.insert(1, 'new')
        assert sl.search(1) == 'new'


# ═══════════════════════════════════════════
#  49. 组件单测 — RLE
# ═══════════════════════════════════════════

class TestRLE:
    def test_encode_decode(self):
        from storage.compression.rle import rle_encode, rle_decode
        data = [1, 1, 1, 2, 2, 3]
        rv, rl = rle_encode(data)
        assert rv == [1, 2, 3]
        assert rl == [3, 2, 1]
        decoded = rle_decode(rv, rl)
        assert decoded == data

    def test_empty(self):
        from storage.compression.rle import rle_encode, rle_decode
        rv, rl = rle_encode([])
        assert rv == []
        assert rl == []
        assert rle_decode(rv, rl) == []

    def test_no_runs(self):
        from storage.compression.rle import rle_encode, rle_decode
        data = [1, 2, 3, 4, 5]
        rv, rl = rle_encode(data)
        assert rv == data
        assert rl == [1, 1, 1, 1, 1]
        assert rle_decode(rv, rl) == data

    def test_single_run(self):
        from storage.compression.rle import rle_encode
        data = [42] * 100
        rv, rl = rle_encode(data)
        assert rv == [42]
        assert rl == [100]


# ═══════════════════════════════════════════
#  50. 组件单测 — Delta
# ═══════════════════════════════════════════

class TestDelta:
    def test_encode_decode(self):
        from storage.compression.delta import delta_encode, delta_decode
        data = [10, 12, 15, 20]
        base, deltas = delta_encode(data)
        assert base == 10
        decoded = delta_decode(base, deltas)
        assert decoded == data

    def test_sorted_data(self):
        from storage.compression.delta import delta_encode, delta_decode
        data = list(range(100))
        base, deltas = delta_encode(data)
        assert all(d == 1 for d in deltas[1:])
        assert delta_decode(base, deltas) == data

    def test_constant(self):
        from storage.compression.delta import delta_encode
        data = [5, 5, 5, 5]
        base, deltas = delta_encode(data)
        assert all(d == 0 for d in deltas)


# ═══════════════════════════════════════════
#  51. 组件单测 — FenwickTree
# ═══════════════════════════════════════════

class TestFenwickTree:
    def test_prefix_sum(self):
        from structures.fenwick_tree import FenwickTree
        ft = FenwickTree.from_list([1, 2, 3, 4, 5])
        assert ft.prefix_sum(0) == 1
        assert ft.prefix_sum(4) == 15

    def test_range_sum(self):
        from structures.fenwick_tree import FenwickTree
        ft = FenwickTree.from_list([1, 2, 3, 4, 5])
        assert ft.range_sum(1, 3) == 9  # 2+3+4

    def test_update(self):
        from structures.fenwick_tree import FenwickTree
        ft = FenwickTree.from_list([1, 2, 3])
        ft.update(1, 5)  # 2 → 7
        assert ft.prefix_sum(2) == 11  # 1+7+3


# ═══════════════════════════════════════════
#  52. 组件单测 — SparseTable
# ═══════════════════════════════════════════

class TestSparseTable:
    def test_min_query(self):
        from structures.sparse_table import SparseTableMin
        st = SparseTableMin([3, 1, 4, 1, 5, 9])
        assert st.query(0, 5) == 1
        assert st.query(2, 4) == 1
        assert st.query(4, 5) == 5

    def test_max_query(self):
        from structures.sparse_table import SparseTableMax
        st = SparseTableMax([3, 1, 4, 1, 5, 9])
        assert st.query(0, 5) == 9
        assert st.query(0, 2) == 4

    def test_single_element(self):
        from structures.sparse_table import SparseTableMin
        st = SparseTableMin([42])
        assert st.query(0, 0) == 42


# ═══════════════════════════════════════════
#  53. 组件单测 — ResultCache
# ═══════════════════════════════════════════

class TestResultCacheUnit:
    def test_sql_hash_whitespace(self):
        from executor.result_cache import ResultCache
        h1 = ResultCache.hash_sql("SELECT  *  FROM  t;")
        h2 = ResultCache.hash_sql("SELECT * FROM t;")
        assert h1 == h2

    def test_sql_hash_string_preserved(self):
        from executor.result_cache import ResultCache
        h1 = ResultCache.hash_sql("SELECT 'hello  world';")
        h2 = ResultCache.hash_sql("SELECT 'hello world';")
        # 字符串内空白不同，哈希应不同
        assert h1 != h2

    def test_cache_eviction(self):
        from executor.result_cache import ResultCache
        cache = ResultCache(max_size=2)
        cache.put(1, 'r1', {})
        cache.put(2, 'r2', {})
        cache.put(3, 'r3', {})
        # 最旧的应被淘汰
        assert cache.get(1, {}) is None
        assert cache.get(2, {}) == 'r2'
        assert cache.get(3, {}) == 'r3'

    def test_cache_invalidate(self):
        from executor.result_cache import ResultCache
        cache = ResultCache()
        cache.put(1, 'r1', {'t': 0})
        cache.invalidate_table('t')
        assert cache.get(1, {'t': 0}) is None


# ═══════════════════════════════════════════
#  54. 组件单测 — Optimizer 常量折叠
# ═══════════════════════════════════════════

class TestConstantFolding:
    def test_fold_addition(self, eng):
        r = eng.execute("SELECT 1 + 2 + 3;")
        assert scalar(r) == 6

    def test_fold_multiplication(self, eng):
        r = eng.execute("SELECT 2 * 3 * 4;")
        assert scalar(r) == 24

    def test_fold_boolean(self, eng):
        r = eng.execute("SELECT TRUE AND TRUE;")
        assert scalar(r) is True

    def test_fold_identity(self, eng):
        r = eng.execute("SELECT 42 + 0;")
        assert scalar(r) == 42

    def test_fold_multiply_by_one(self, eng):
        r = eng.execute("SELECT 42 * 1;")
        assert scalar(r) == 42

    def test_fold_concat(self, eng):
        r = eng.execute("SELECT 'hello' || ' ' || 'world';")
        assert scalar(r) == 'hello world'

    def test_fold_negative_division(self, eng):
        """SQL 截断除法常量折叠。"""
        r = eng.execute("SELECT -7 / 2;")
        assert scalar(r) == -3


# ═══════════════════════════════════════════
#  55. 回归测试 — 已修复的 Bug
# ═══════════════════════════════════════════

class TestRegressions:
    def test_alter_add_refreshes_system_tables(self, eng):
        """回归：ALTER ADD COLUMN 后 z1db_columns 应更新。"""
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("ALTER TABLE t ADD COLUMN v INT;")
        r = eng.execute(
            "SELECT COUNT(*) FROM z1db_columns "
            "WHERE table_name = 't';")
        assert scalar(r) == 2

    def test_alter_drop_refreshes_system_tables(self, eng):
        eng.execute("CREATE TABLE t (id INT, v INT, w INT);")
        r0 = eng.execute("SELECT COUNT(*) FROM z1db_columns WHERE table_name = 't';")
        assert scalar(r0) == 3
        eng.execute("ALTER TABLE t DROP COLUMN v;")
        r = eng.execute("SELECT COUNT(*) FROM z1db_columns WHERE table_name = 't';")
        assert scalar(r) == 2

    def test_alter_rename_refreshes_system_tables(self, eng):
        """回归：ALTER RENAME COLUMN 后 z1db_columns 应更新。"""
        eng.execute("CREATE TABLE t (id INT, old_name INT);")
        eng.execute("ALTER TABLE t RENAME COLUMN old_name TO new_name;")
        r = eng.execute(
            "SELECT column_name FROM z1db_columns "
            "WHERE table_name = 't' AND ordinal_position = 2;")
        assert scalar(r) == 'new_name'

    def test_system_table_filter_empty_result(self, eng):
        """回归：系统表过滤空结果不应返回空 batch。"""
        r = eng.execute(
            "SELECT * FROM z1db_tables "
            "WHERE table_name = 'nonexistent';")
        assert r.row_count == 0

    def test_system_table_limit_no_recursion(self, eng):
        """回归：系统表 LIMIT 不应栈溢出。"""
        for i in range(20):
            eng.execute(f"CREATE TABLE t{i} (id INT);")
        r = eng.execute(
            "SELECT table_name FROM z1db_tables "
            "WHERE is_system = FALSE LIMIT 5;")
        assert r.row_count == 5

    def test_union_all_close_safety(self, eng):
        """回归：UNION ALL close 不应重复关闭子算子。"""
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (2);")
        # 多次执行不应崩溃
        for _ in range(5):
            r = eng.execute(
                "SELECT v FROM a UNION ALL SELECT v FROM b;")
            assert r.row_count == 2

    def test_dispatch_thread_safety(self, eng):
        """回归：函数分发表初始化线程安全。"""
        errors = []

        def run_func():
            try:
                r = eng.execute("SELECT UPPER('hello');")
                assert scalar(r) == 'HELLO'
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_func)
                   for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_integer_truncation_division(self, eng):
        """回归：SQL 标准截断除法。"""
        cases = [
            ("SELECT 7 / 2;", 3),
            ("SELECT -7 / 2;", -3),
            ("SELECT 7 / -2;", -3),
            ("SELECT -7 / -2;", 3),
        ]
        for sql, expected in cases:
            r = eng.execute(sql)
            assert scalar(r) == expected, f"{sql} expected {expected}"

    def test_create_table_system_name_blocked(self, eng):
        """回归：不允许创建系统表名的表。"""
        with pytest.raises(ExecutionError):
            eng.execute("CREATE TABLE z1db_tables (id INT);")
        with pytest.raises(ExecutionError):
            eng.execute("CREATE TABLE z1db_columns (id INT);")

    def test_drop_system_table_blocked(self, eng):
        """回归：不允许删除系统表。"""
        with pytest.raises(ExecutionError):
            eng.execute("DROP TABLE z1db_tables;")

    def test_update_system_table_blocked(self, eng):
        """回归：不允许修改系统表。"""
        with pytest.raises(ExecutionError):
            eng.execute(
                "UPDATE z1db_tables SET table_name = 'x' "
                "WHERE table_name = 'z1db_tables';")

    def test_cte_cleanup(self, eng):
        """回归：CTE 临时表应在查询完成后清理。"""
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("INSERT INTO t VALUES (1),(2),(3);")
        eng.execute("""
            WITH tmp AS (SELECT id FROM t WHERE id < 3)
            SELECT * FROM tmp;
        """)
        # CTE 表不应出现在 table 列表中
        names = eng.get_table_names()
        assert not any(n.startswith('__cte_') for n in names)

    def test_empty_group_by_result_columns(self, eng):
        """回归：空表 GROUP BY 应返回正确的列。"""
        eng.execute("CREATE TABLE t (k INT, v INT);")
        r = eng.execute(
            "SELECT k, SUM(v) AS total FROM t GROUP BY k;")
        assert r.columns == ['k', 'total']
        assert r.row_count == 0

    def test_persist_system_tables_excluded(self):
        """回归：持久化不应保存系统表 schema。"""
        with tempfile.TemporaryDirectory() as td:
            e1 = Engine(td)
            e1.execute("CREATE TABLE t (id INT);")
            e1.close()

            import json
            catalog_path = os.path.join(td, 'catalog.json')
            with open(catalog_path, 'r') as f:
                data = json.load(f)
            assert 'z1db_tables' not in data
            assert 'z1db_columns' not in data


# ═══════════════════════════════════════════
#  56. 压力测试
# ═══════════════════════════════════════════

class TestStress:
    def test_many_columns(self, eng):
        cols = ", ".join(f"c{i} INT" for i in range(20))
        eng.execute(f"CREATE TABLE wide ({cols});")
        vals = ", ".join(str(i) for i in range(20))
        eng.execute(f"INSERT INTO wide VALUES ({vals});")
        r = eng.execute("SELECT * FROM wide;")
        assert len(r.columns) == 20
        assert r.rows[0] == list(range(20))

    def test_many_rows_agg(self, eng):
        eng.execute("CREATE TABLE big (id INT, v INT);")
        batch = ", ".join(f"({i}, {i % 10})" for i in range(500))
        eng.execute(f"INSERT INTO big VALUES {batch};")
        r = eng.execute("SELECT COUNT(*), SUM(v), AVG(v) FROM big;")
        assert r.rows[0][0] == 500

    def test_many_group_by_keys(self, eng):
        eng.execute("CREATE TABLE t (k INT, v INT);")
        batch = ", ".join(f"({i}, {i*10})" for i in range(200))
        eng.execute(f"INSERT INTO t VALUES {batch};")
        r = eng.execute(
            "SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k LIMIT 5;")
        assert r.row_count == 5
        assert r.rows[0] == [0, 0]

    def test_repeated_queries(self, populated):
        for _ in range(50):
            r = populated.execute("SELECT COUNT(*) FROM users;")
            assert scalar(r) == 5

    def test_many_tables(self, eng):
        for i in range(30):
            eng.execute(f"CREATE TABLE t{i} (id INT);")
        names = eng.get_table_names()
        user_tables = [n for n in names
                       if not n.startswith('z1db_')]
        assert len(user_tables) == 30

    def test_deep_subquery(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1),(2),(3);")
        r = eng.execute("""
            SELECT * FROM (
                SELECT * FROM (
                    SELECT * FROM t WHERE v > 0
                ) sub1 WHERE v > 1
            ) sub2 WHERE v > 2;
        """)
        assert col_values(r) == [3]

    def test_long_string_operations(self, eng):
        eng.execute("CREATE TABLE t (v TEXT);")
        s = 'a' * 1000
        eng.execute(f"INSERT INTO t VALUES ('{s}');")
        r = eng.execute("SELECT LENGTH(v), UPPER(v) FROM t;")
        assert r.rows[0][0] == 1000
        assert r.rows[0][1] == 'A' * 1000
