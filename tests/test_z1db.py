### tests/test_z1db.py
"""Z1DB 综合测试套件 — 覆盖 DDL/DML/查询/函数/边界条件/BugFix。
运行: pytest tests/test_z1db.py -v
"""
from __future__ import annotations
import os, sys, math, tempfile, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import pytest
from engine import Engine
from utils.errors import (
    Z1Error, ParseError, SemanticError, ExecutionError,
    TableNotFoundError, ColumnNotFoundError, DuplicateError,
    DivisionByZeroError, NumericOverflowError, TypeMismatchError,
)


# ═══════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════

def rows(result):
    return result.rows

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
    eng.execute(
        "CREATE TABLE emp (id INT, name VARCHAR, "
        "dept_id INT, salary INT);")
    eng.execute("INSERT INTO emp VALUES (1,'Alice',1,80000);")
    eng.execute("INSERT INTO emp VALUES (2,'Bob',2,60000);")
    eng.execute("INSERT INTO emp VALUES (3,'Charlie',1,90000);")
    eng.execute("INSERT INTO emp VALUES (4,'Diana',2,70000);")
    eng.execute("INSERT INTO emp VALUES (5,'Eve',1,85000);")
    return eng


@pytest.fixture
def nullable_table(eng):
    """含大量 NULL 的测试表。"""
    eng.execute("CREATE TABLE ntest (id INT, a INT, b VARCHAR, c BOOLEAN);")
    eng.execute("INSERT INTO ntest VALUES (1, 10, 'x', TRUE);")
    eng.execute("INSERT INTO ntest VALUES (2, NULL, 'y', FALSE);")
    eng.execute("INSERT INTO ntest VALUES (3, 30, NULL, NULL);")
    eng.execute("INSERT INTO ntest VALUES (4, NULL, NULL, TRUE);")
    eng.execute("INSERT INTO ntest VALUES (5, 50, 'z', FALSE);")
    return eng


@pytest.fixture
def types_table(eng):
    """多类型表。"""
    eng.execute("""CREATE TABLE tt (
        i INT, bi BIGINT, f FLOAT, d DOUBLE,
        b BOOLEAN, v VARCHAR(50), t TEXT
    );""")
    eng.execute("INSERT INTO tt VALUES (1, 100000000000, 3.14, 2.718, TRUE, 'hello', 'world');")
    eng.execute("INSERT INTO tt VALUES (2, -99999999999, -1.5, 0.0, FALSE, 'foo', 'bar');")
    return eng


# ═══════════════════════════════════════════
#  1. DDL — CREATE / DROP / ALTER
# ═══════════════════════════════════════════

class TestDDL:
    def test_create_table(self, eng):
        r = eng.execute("CREATE TABLE t (id INT, val VARCHAR);")
        assert r.message == 'OK'
        assert eng.table_exists('t')

    def test_create_if_not_exists(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        r = eng.execute("CREATE TABLE IF NOT EXISTS t (id INT);")
        assert r.message == 'OK'

    def test_create_duplicate_raises(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        with pytest.raises(DuplicateError):
            eng.execute("CREATE TABLE t (id INT);")

    def test_drop_table(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("DROP TABLE t;")
        assert not eng.table_exists('t')

    def test_drop_nonexistent_raises(self, eng):
        with pytest.raises(TableNotFoundError):
            eng.execute("DROP TABLE no_such_table;")

    def test_drop_if_exists(self, eng):
        r = eng.execute("DROP TABLE IF EXISTS no_such;")
        assert r.message == 'OK'

    def test_alter_add_column(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("INSERT INTO t VALUES (1);")
        eng.execute("ALTER TABLE t ADD COLUMN name VARCHAR;")
        r = eng.execute("SELECT id, name FROM t;")
        assert r.rows == [[1, None]]

    def test_alter_drop_column(self, eng):
        eng.execute("CREATE TABLE t (id INT, name VARCHAR);")
        eng.execute("INSERT INTO t VALUES (1, 'hello');")
        eng.execute("ALTER TABLE t DROP COLUMN name;")
        r = eng.execute("SELECT id FROM t;")
        assert r.rows == [[1]]

    def test_alter_rename_column(self, eng):
        eng.execute("CREATE TABLE t (id INT, old_name VARCHAR);")
        eng.execute("INSERT INTO t VALUES (1, 'hello');")
        eng.execute("ALTER TABLE t RENAME COLUMN old_name TO new_name;")
        r = eng.execute("SELECT new_name FROM t;")
        assert r.rows[0][0] == 'hello'

    def test_alter_drop_only_column_raises(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        with pytest.raises(ExecutionError):
            eng.execute("ALTER TABLE t DROP COLUMN id;")

    def test_alter_add_duplicate_column_raises(self, eng):
        eng.execute("CREATE TABLE t (id INT, name VARCHAR);")
        with pytest.raises(DuplicateError):
            eng.execute("ALTER TABLE t ADD COLUMN name VARCHAR;")

    def test_alter_drop_nonexistent_column_raises(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        with pytest.raises(ColumnNotFoundError):
            eng.execute("ALTER TABLE t DROP COLUMN nope;")

    def test_alter_rename_to_existing_raises(self, eng):
        eng.execute("CREATE TABLE t (id INT, name VARCHAR);")
        with pytest.raises(DuplicateError):
            eng.execute("ALTER TABLE t RENAME COLUMN name TO id;")

    def test_create_all_types(self, eng):
        eng.execute("""CREATE TABLE types (
            a INT, b BIGINT, c FLOAT, d DOUBLE,
            e BOOLEAN, f VARCHAR(100), g TEXT,
            h DATE, i TIMESTAMP
        );""")
        assert eng.table_exists('types')

    def test_create_table_primary_key(self, eng):
        eng.execute("CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR);")
        schema = eng.get_table_schema('t')
        assert schema.primary_key == 'id'

    def test_create_table_not_null(self, eng):
        eng.execute("CREATE TABLE t (id INT NOT NULL, name VARCHAR);")
        schema = eng.get_table_schema('t')
        assert not schema.columns[0].nullable
        assert schema.columns[1].nullable

    def test_alter_preserves_data(self, eng):
        eng.execute("CREATE TABLE t (id INT, a INT, b INT);")
        eng.execute("INSERT INTO t VALUES (1, 10, 100);")
        eng.execute("INSERT INTO t VALUES (2, 20, 200);")
        eng.execute("ALTER TABLE t ADD COLUMN c INT;")
        r = eng.execute("SELECT * FROM t ORDER BY id;")
        assert r.rows[0] == [1, 10, 100, None]
        assert r.rows[1] == [2, 20, 200, None]


# ═══════════════════════════════════════════
#  2. DML — INSERT / UPDATE / DELETE
# ═══════════════════════════════════════════

class TestDML:
    def test_insert_single(self, eng):
        eng.execute("CREATE TABLE t (id INT, val VARCHAR);")
        r = eng.execute("INSERT INTO t VALUES (1, 'hello');")
        assert r.affected_rows == 1

    def test_insert_multiple_rows(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("INSERT INTO t VALUES (1), (2), (3);")
        r = eng.execute("SELECT COUNT(*) FROM t;")
        assert scalar(r) == 3

    def test_insert_named_columns(self, eng):
        eng.execute("CREATE TABLE t (id INT, name VARCHAR, age INT);")
        eng.execute("INSERT INTO t (id, name) VALUES (1, 'Alice');")
        r = eng.execute("SELECT id, name, age FROM t;")
        assert r.rows == [[1, 'Alice', None]]

    def test_insert_select(self, eng):
        eng.execute("CREATE TABLE src (id INT);")
        eng.execute("INSERT INTO src VALUES (1), (2);")
        eng.execute("CREATE TABLE dst (id INT);")
        eng.execute("INSERT INTO dst SELECT id FROM src;")
        r = eng.execute("SELECT COUNT(*) FROM dst;")
        assert scalar(r) == 2

    def test_insert_column_order(self, eng):
        eng.execute("CREATE TABLE t (a INT, b INT, c INT);")
        eng.execute("INSERT INTO t (c, a) VALUES (30, 10);")
        r = eng.execute("SELECT a, b, c FROM t;")
        assert r.rows == [[10, None, 30]]

    def test_insert_duplicate_column_raises(self, eng):
        eng.execute("CREATE TABLE t (a INT, b INT);")
        with pytest.raises(ExecutionError):
            eng.execute("INSERT INTO t (a, a) VALUES (1, 2);")

    def test_insert_wrong_column_count_raises(self, eng):
        eng.execute("CREATE TABLE t (a INT, b INT);")
        with pytest.raises(ExecutionError):
            eng.execute("INSERT INTO t VALUES (1, 2, 3);")

    def test_update_all(self, populated):
        populated.execute("UPDATE users SET age = 99;")
        r = populated.execute("SELECT DISTINCT age FROM users;")
        assert r.rows == [[99]]

    def test_update_where(self, populated):
        populated.execute("UPDATE users SET age = 99 WHERE name = 'Alice';")
        r = populated.execute("SELECT age FROM users WHERE name = 'Alice';")
        assert scalar(r) == 99

    def test_update_expression(self, populated):
        populated.execute("UPDATE users SET age = age + 10 WHERE id = 1;")
        r = populated.execute("SELECT age FROM users WHERE id = 1;")
        assert scalar(r) == 40

    def test_update_multiple_assignments(self, populated):
        populated.execute("UPDATE users SET name = 'X', age = 0 WHERE id = 1;")
        r = populated.execute("SELECT name, age FROM users WHERE id = 1;")
        assert r.rows[0] == ['X', 0]

    def test_delete_where(self, populated):
        populated.execute("DELETE FROM users WHERE age < 28;")
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r) == 4

    def test_delete_all(self, populated):
        populated.execute("DELETE FROM users;")
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r) == 0

    def test_insert_null(self, eng):
        eng.execute("CREATE TABLE t (id INT, val INT);")
        eng.execute("INSERT INTO t VALUES (1, NULL);")
        r = eng.execute("SELECT val FROM t;")
        assert r.rows[0][0] is None

    def test_insert_not_null_violation(self, eng):
        eng.execute("CREATE TABLE t (id INT NOT NULL);")
        with pytest.raises(ExecutionError):
            eng.execute("INSERT INTO t VALUES (NULL);")

    def test_insert_boolean(self, eng):
        eng.execute("CREATE TABLE t (id INT, flag BOOLEAN);")
        eng.execute("INSERT INTO t VALUES (1, TRUE);")
        eng.execute("INSERT INTO t VALUES (2, FALSE);")
        r = eng.execute("SELECT flag FROM t ORDER BY id;")
        assert col_values(r) == [True, False]

    def test_update_no_match(self, populated):
        r = populated.execute("UPDATE users SET age = 99 WHERE id = 999;")
        assert r.affected_rows == 0

    def test_delete_no_match(self, populated):
        r = populated.execute("DELETE FROM users WHERE id = 999;")
        assert r.affected_rows == 0

    def test_delete_then_insert(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("INSERT INTO t VALUES (1), (2), (3);")
        eng.execute("DELETE FROM t WHERE id = 2;")
        eng.execute("INSERT INTO t VALUES (4);")
        r = eng.execute("SELECT id FROM t ORDER BY id;")
        assert col_values(r) == [1, 3, 4]

    def test_update_then_select(self, eng):
        eng.execute("CREATE TABLE t (id INT, v INT);")
        eng.execute("INSERT INTO t VALUES (1, 10), (2, 20);")
        eng.execute("UPDATE t SET v = v * 2 WHERE id = 1;")
        r = eng.execute("SELECT v FROM t WHERE id = 1;")
        assert scalar(r) == 20


# ═══════════════════════════════════════════
#  3. SELECT 基础查询
# ═══════════════════════════════════════════

class TestSelect:
    def test_select_star(self, populated):
        r = populated.execute("SELECT * FROM users;")
        assert r.row_count == 5
        assert len(r.columns) == 3

    def test_select_columns(self, populated):
        r = populated.execute("SELECT name, age FROM users;")
        assert r.columns == ['name', 'age']

    def test_select_where_eq(self, populated):
        r = populated.execute("SELECT name FROM users WHERE id = 1;")
        assert scalar(r) == 'Alice'

    def test_select_where_neq(self, populated):
        r = populated.execute("SELECT COUNT(*) FROM users WHERE id != 1;")
        assert scalar(r) == 4

    def test_select_where_lt(self, populated):
        r = populated.execute("SELECT COUNT(*) FROM users WHERE age < 30;")
        assert scalar(r) == 2

    def test_select_where_gt(self, populated):
        r = populated.execute("SELECT COUNT(*) FROM users WHERE age > 30;")
        assert scalar(r) == 1

    def test_select_where_lte(self, populated):
        r = populated.execute("SELECT COUNT(*) FROM users WHERE age <= 30;")
        assert scalar(r) == 4

    def test_select_where_gte(self, populated):
        r = populated.execute("SELECT COUNT(*) FROM users WHERE age >= 30;")
        assert scalar(r) == 3

    def test_select_where_and(self, populated):
        r = populated.execute(
            "SELECT name FROM users WHERE age >= 30 AND age <= 30;")
        assert set(col_values(r)) == {'Alice', 'Eve'}

    def test_select_where_or(self, populated):
        r = populated.execute(
            "SELECT name FROM users WHERE age = 25 OR age = 35;")
        assert set(col_values(r)) == {'Bob', 'Charlie'}

    def test_select_where_not(self, populated):
        r = populated.execute(
            "SELECT COUNT(*) FROM users WHERE NOT age = 30;")
        assert scalar(r) == 3

    def test_select_order_by_asc(self, populated):
        r = populated.execute("SELECT name FROM users ORDER BY age ASC;")
        assert col_values(r)[0] == 'Bob'

    def test_select_order_by_desc(self, populated):
        r = populated.execute("SELECT name FROM users ORDER BY age DESC;")
        assert col_values(r)[0] == 'Charlie'

    def test_select_order_by_multiple(self, populated):
        r = populated.execute(
            "SELECT name FROM users ORDER BY age DESC, name ASC;")
        names = col_values(r)
        assert names[0] == 'Charlie'

    def test_select_limit(self, populated):
        r = populated.execute("SELECT name FROM users ORDER BY id LIMIT 2;")
        assert r.row_count == 2

    def test_select_offset(self, populated):
        r = populated.execute("SELECT id FROM users ORDER BY id LIMIT 2 OFFSET 3;")
        assert col_values(r) == [4, 5]

    def test_select_distinct(self, populated):
        r = populated.execute("SELECT DISTINCT age FROM users;")
        ages = col_values(r)
        assert len(ages) == len(set(ages))

    def test_select_alias(self, populated):
        r = populated.execute("SELECT name AS n, age AS a FROM users LIMIT 1;")
        assert r.columns == ['n', 'a']

    def test_select_expression(self, eng):
        r = eng.execute("SELECT 1 + 2;")
        assert scalar(r) == 3

    def test_select_no_from(self, eng):
        r = eng.execute("SELECT 42;")
        assert scalar(r) == 42

    def test_select_multiple_no_from(self, eng):
        r = eng.execute("SELECT 1, 'hello', TRUE;")
        assert r.rows[0] == [1, 'hello', True]

    def test_select_string_concat(self, eng):
        r = eng.execute("SELECT 'hello' || ' ' || 'world';")
        assert scalar(r) == 'hello world'

    def test_select_null_arithmetic(self, eng):
        r = eng.execute("SELECT NULL + 1;")
        assert scalar(r) is None

    def test_select_between(self, populated):
        r = populated.execute(
            "SELECT COUNT(*) FROM users WHERE age BETWEEN 28 AND 32;")
        assert scalar(r) == 3

    def test_select_not_between(self, populated):
        r = populated.execute(
            "SELECT COUNT(*) FROM users WHERE age NOT BETWEEN 28 AND 32;")
        assert scalar(r) == 2

    def test_select_in(self, populated):
        r = populated.execute(
            "SELECT COUNT(*) FROM users WHERE age IN (25, 30);")
        assert scalar(r) == 3

    def test_select_not_in(self, populated):
        r = populated.execute(
            "SELECT COUNT(*) FROM users WHERE age NOT IN (25, 30);")
        assert scalar(r) == 2

    def test_select_like(self, populated):
        r = populated.execute("SELECT name FROM users WHERE name LIKE 'A%';")
        assert col_values(r) == ['Alice']

    def test_select_like_underscore(self, populated):
        r = populated.execute("SELECT name FROM users WHERE name LIKE 'Bo_';")
        assert col_values(r) == ['Bob']

    def test_select_not_like(self, populated):
        r = populated.execute(
            "SELECT COUNT(*) FROM users WHERE name NOT LIKE 'A%';")
        assert scalar(r) == 4

    def test_select_like_percent_middle(self, populated):
        r = populated.execute(
            "SELECT name FROM users WHERE name LIKE '%li%';")
        names = col_values(r)
        assert 'Alice' in names
        assert 'Charlie' in names

    def test_select_is_null(self, eng):
        eng.execute("CREATE TABLE t (id INT, val INT);")
        eng.execute("INSERT INTO t VALUES (1, NULL), (2, 10);")
        r = eng.execute("SELECT id FROM t WHERE val IS NULL;")
        assert col_values(r) == [1]

    def test_select_is_not_null(self, eng):
        eng.execute("CREATE TABLE t (id INT, val INT);")
        eng.execute("INSERT INTO t VALUES (1, NULL), (2, 10);")
        r = eng.execute("SELECT id FROM t WHERE val IS NOT NULL;")
        assert col_values(r) == [2]

    def test_select_case_simple(self, populated):
        r = populated.execute("""
            SELECT name, CASE age
                WHEN 25 THEN 'young'
                WHEN 30 THEN 'middle'
                ELSE 'other' END AS cat
            FROM users WHERE id <= 2 ORDER BY id;
        """)
        assert r.rows[0][1] == 'middle'
        assert r.rows[1][1] == 'young'

    def test_select_case_searched(self, populated):
        r = populated.execute("""
            SELECT name, CASE WHEN age >= 30 THEN 'senior'
                              ELSE 'junior' END AS level
            FROM users WHERE id = 1;
        """)
        assert r.rows[0][1] == 'senior'

    def test_select_case_no_else(self, populated):
        r = populated.execute("""
            SELECT CASE WHEN age = 25 THEN 'young' END
            FROM users WHERE id = 1;
        """)
        assert scalar(r) is None

    def test_select_cast(self, eng):
        r = eng.execute("SELECT CAST(42 AS VARCHAR);")
        assert scalar(r) == '42'

    def test_select_cast_float_to_int(self, eng):
        r = eng.execute("SELECT CAST(3.7 AS INT);")
        assert scalar(r) == 3

    def test_select_cast_string_to_int(self, eng):
        r = eng.execute("SELECT CAST('123' AS INT);")
        assert scalar(r) == 123

    def test_select_order_by_alias(self, populated):
        r = populated.execute(
            "SELECT name, age AS a FROM users ORDER BY a LIMIT 1;")
        assert r.rows[0][0] == 'Bob'

    def test_select_order_by_ordinal(self, populated):
        r = populated.execute(
            "SELECT name, age FROM users ORDER BY 2 LIMIT 1;")
        assert r.rows[0][0] == 'Bob'

    def test_select_where_boolean_col(self, nullable_table):
        r = nullable_table.execute(
            "SELECT id FROM ntest WHERE c = TRUE ORDER BY id;")
        assert col_values(r) == [1, 4]

    def test_select_qualified_column(self, populated):
        r = populated.execute(
            "SELECT users.name FROM users WHERE users.id = 1;")
        assert scalar(r) == 'Alice'


# ═══════════════════════════════════════════
#  4. 聚合函数
# ═══════════════════════════════════════════

class TestAggregates:
    def test_count_star(self, populated):
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r) == 5

    def test_count_column(self, eng):
        eng.execute("CREATE TABLE t (id INT, v INT);")
        eng.execute("INSERT INTO t VALUES (1,10),(2,NULL),(3,30);")
        r = eng.execute("SELECT COUNT(v) FROM t;")
        assert scalar(r) == 2

    def test_sum(self, populated):
        r = populated.execute("SELECT SUM(age) FROM users;")
        assert scalar(r) == 148

    def test_avg(self, populated):
        r = populated.execute("SELECT AVG(age) FROM users;")
        assert abs(scalar(r) - 29.6) < 0.01

    def test_min_max(self, populated):
        r = populated.execute("SELECT MIN(age), MAX(age) FROM users;")
        assert r.rows[0] == [25, 35]

    def test_min_string(self, populated):
        r = populated.execute("SELECT MIN(name), MAX(name) FROM users;")
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 'Eve'

    def test_count_distinct(self, populated):
        r = populated.execute("SELECT COUNT(DISTINCT age) FROM users;")
        assert scalar(r) == 4

    def test_group_by(self, populated):
        r = populated.execute(
            "SELECT age, COUNT(*) AS cnt FROM users "
            "GROUP BY age ORDER BY age;")
        assert r.row_count == 4
        for row in r.rows:
            if row[0] == 30:
                assert row[1] == 2

    def test_group_by_having(self, populated):
        r = populated.execute(
            "SELECT age, COUNT(*) AS cnt FROM users "
            "GROUP BY age HAVING COUNT(*) > 1;")
        assert r.row_count == 1
        assert r.rows[0][0] == 30

    def test_sum_distinct(self, populated):
        r = populated.execute("SELECT SUM(DISTINCT age) FROM users;")
        assert scalar(r) == 118

    def test_avg_distinct(self, populated):
        r = populated.execute("SELECT AVG(DISTINCT age) FROM users;")
        assert abs(scalar(r) - 29.5) < 0.01

    def test_stddev(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            eng.execute(f"INSERT INTO t VALUES ({v});")
        r = eng.execute("SELECT STDDEV(v) FROM t;")
        assert abs(scalar(r) - 2.0) < 0.2

    def test_variance(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            eng.execute(f"INSERT INTO t VALUES ({v});")
        r = eng.execute("SELECT VARIANCE(v) FROM t;")
        assert scalar(r) is not None and scalar(r) > 0

    def test_median(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        for v in [1, 3, 5, 7, 9]:
            eng.execute(f"INSERT INTO t VALUES ({v});")
        r = eng.execute("SELECT MEDIAN(v) FROM t;")
        assert scalar(r) == 5

    def test_agg_empty_table(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        r = eng.execute("SELECT COUNT(*), SUM(v) FROM t;")
        assert r.rows[0][0] == 0
        assert r.rows[0][1] is None

    def test_agg_all_nulls(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (NULL), (NULL);")
        r = eng.execute("SELECT SUM(v), AVG(v), MIN(v), MAX(v) FROM t;")
        for i in range(4):
            assert r.rows[0][i] is None

    def test_group_by_multiple_keys(self, eng):
        eng.execute("CREATE TABLE t (a INT, b INT, v INT);")
        eng.execute("INSERT INTO t VALUES (1,1,10),(1,2,20),(2,1,30),(1,1,40);")
        r = eng.execute(
            "SELECT a, b, SUM(v) FROM t GROUP BY a, b ORDER BY a, b;")
        assert r.row_count == 3
        assert r.rows[0] == [1, 1, 50]

    def test_group_by_expression(self, populated):
        r = populated.execute(
            "SELECT age > 28 AS senior, COUNT(*) FROM users "
            "GROUP BY age > 28;")
        assert r.row_count == 2

    def test_string_agg(self, populated):
        r = populated.execute(
            "SELECT STRING_AGG(name, ', ') FROM users ORDER BY name;")
        val = scalar(r)
        assert val is not None
        assert 'Alice' in val

    def test_array_agg(self, populated):
        r = populated.execute("SELECT ARRAY_AGG(age) FROM users;")
        val = scalar(r)
        assert val is not None


# ═══════════════════════════════════════════
#  5. JOIN
# ═══════════════════════════════════════════

class TestJoin:
    def test_inner_join(self, multi_table):
        r = multi_table.execute(
            "SELECT e.name, d.name FROM emp e "
            "JOIN dept d ON e.dept_id = d.id ORDER BY e.id;")
        assert r.row_count == 5
        assert r.rows[0][1] == 'Engineering'

    def test_left_join(self, multi_table):
        multi_table.execute("INSERT INTO emp VALUES (6,'Frank',99,50000);")
        r = multi_table.execute(
            "SELECT e.name, d.name FROM emp e "
            "LEFT JOIN dept d ON e.dept_id = d.id "
            "WHERE e.name = 'Frank';")
        assert r.rows[0][1] is None

    def test_right_join(self, multi_table):
        multi_table.execute("INSERT INTO dept VALUES (3, 'HR');")
        r = multi_table.execute(
            "SELECT e.name, d.name FROM emp e "
            "RIGHT JOIN dept d ON e.dept_id = d.id "
            "WHERE d.name = 'HR';")
        assert r.row_count == 1
        assert r.rows[0][0] is None

    def test_cross_join(self, multi_table):
        r = multi_table.execute(
            "SELECT COUNT(*) FROM dept d1 CROSS JOIN dept d2;")
        assert scalar(r) == 4

    def test_cross_join_comma(self, multi_table):
        r = multi_table.execute(
            "SELECT COUNT(*) FROM dept d1, dept d2;")
        assert scalar(r) == 4

    def test_join_with_agg(self, multi_table):
        r = multi_table.execute(
            "SELECT d.name, SUM(e.salary) AS total "
            "FROM emp e JOIN dept d ON e.dept_id = d.id "
            "GROUP BY d.name ORDER BY total DESC;")
        assert r.row_count == 2
        assert r.rows[0][0] == 'Engineering'

    def test_self_join(self, populated):
        r = populated.execute(
            "SELECT a.name, b.name FROM users a "
            "JOIN users b ON a.age = b.age AND a.id < b.id;")
        pairs = [(row[0], row[1]) for row in r.rows]
        assert ('Alice', 'Eve') in pairs

    def test_join_three_tables(self, multi_table):
        multi_table.execute("CREATE TABLE bonus (emp_id INT, amount INT);")
        multi_table.execute("INSERT INTO bonus VALUES (1, 5000), (3, 8000);")
        r = multi_table.execute(
            "SELECT e.name, d.name, b.amount "
            "FROM emp e JOIN dept d ON e.dept_id = d.id "
            "JOIN bonus b ON e.id = b.emp_id ORDER BY b.amount DESC;")
        assert r.row_count == 2
        assert r.rows[0][0] == 'Charlie'

    def test_join_subquery(self, multi_table):
        r = multi_table.execute(
            "SELECT e.name FROM emp e "
            "JOIN (SELECT id FROM dept WHERE name = 'Sales') d "
            "ON e.dept_id = d.id ORDER BY e.name;")
        assert set(col_values(r)) == {'Bob', 'Diana'}

    def test_join_where_filter(self, multi_table):
        r = multi_table.execute(
            "SELECT e.name, d.name FROM emp e "
            "JOIN dept d ON e.dept_id = d.id "
            "WHERE e.salary > 80000 ORDER BY e.name;")
        assert set(col_values(r)) == {'Charlie', 'Eve'}


# ═══════════════════════════════════════════
#  6. SET 操作
# ═══════════════════════════════════════════

class TestSetOps:
    def test_union_all(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(2);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (2),(3);")
        r = eng.execute("SELECT v FROM a UNION ALL SELECT v FROM b;")
        assert r.row_count == 4

    def test_union_distinct(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(2);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (2),(3);")
        r = eng.execute("SELECT v FROM a UNION SELECT v FROM b;")
        assert r.row_count == 3

    def test_intersect(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(2),(3);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (2),(3),(4);")
        r = eng.execute("SELECT v FROM a INTERSECT SELECT v FROM b;")
        assert set(col_values(r)) == {2, 3}

    def test_except(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(2),(3);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (2),(3),(4);")
        r = eng.execute("SELECT v FROM a EXCEPT SELECT v FROM b;")
        assert col_values(r) == [1]

    def test_union_all_preserves_duplicates(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(1);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (1);")
        r = eng.execute("SELECT v FROM a UNION ALL SELECT v FROM b;")
        assert r.row_count == 3

    def test_intersect_all(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1),(1),(2);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (1),(2),(2);")
        r = eng.execute("SELECT v FROM a INTERSECT ALL SELECT v FROM b;")
        assert r.row_count == 2

    def test_chained_union(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (1);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (2);")
        eng.execute("CREATE TABLE c (v INT);")
        eng.execute("INSERT INTO c VALUES (3);")
        r = eng.execute(
            "SELECT v FROM a UNION ALL SELECT v FROM b "
            "UNION ALL SELECT v FROM c;")
        assert r.row_count == 3


# ═══════════════════════════════════════════
#  7. 子查询
# ═══════════════════════════════════════════

class TestSubquery:
    def test_scalar_subquery(self, populated):
        r = populated.execute(
            "SELECT name FROM users "
            "WHERE age = (SELECT MAX(age) FROM users);")
        assert scalar(r) == 'Charlie'

    def test_in_subquery(self, multi_table):
        r = multi_table.execute(
            "SELECT name FROM emp WHERE dept_id IN "
            "(SELECT id FROM dept WHERE name = 'Sales') "
            "ORDER BY name;")
        assert set(col_values(r)) == {'Bob', 'Diana'}

    def test_exists_subquery(self, multi_table):
        r = multi_table.execute(
            "SELECT name FROM dept d WHERE EXISTS "
            "(SELECT 1 FROM emp WHERE dept_id = d.id);")
        assert r.row_count == 2

    def test_not_exists(self, multi_table):
        multi_table.execute("INSERT INTO dept VALUES (3, 'HR');")
        r = multi_table.execute(
            "SELECT name FROM dept d WHERE NOT EXISTS "
            "(SELECT 1 FROM emp WHERE dept_id = d.id);")
        assert col_values(r) == ['HR']

    def test_subquery_in_select(self, multi_table):
        r = multi_table.execute(
            "SELECT name, "
            "(SELECT COUNT(*) FROM emp WHERE dept_id = d.id) AS cnt "
            "FROM dept d ORDER BY d.id;")
        assert r.rows[0][1] == 3  # Engineering
        assert r.rows[1][1] == 2  # Sales


# ═══════════════════════════════════════════
#  8. 窗口函数
# ═══════════════════════════════════════════

class TestWindow:
    def test_row_number(self, populated):
        r = populated.execute(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY age) AS rn "
            "FROM users;")
        rns = col_values(r, 1)
        assert sorted(rns) == [1, 2, 3, 4, 5]

    def test_rank(self, populated):
        r = populated.execute(
            "SELECT name, RANK() OVER (ORDER BY age) AS rnk "
            "FROM users;")
        assert r.row_count == 5

    def test_dense_rank(self, populated):
        r = populated.execute(
            "SELECT name, DENSE_RANK() OVER (ORDER BY age) AS dr "
            "FROM users;")
        drs = col_values(r, 1)
        # ages: 25,28,30,30,35 → dense_rank: 1,2,3,3,4
        assert max(drs) == 4

    def test_sum_window(self, populated):
        r = populated.execute(
            "SELECT name, SUM(age) OVER (ORDER BY id) AS running "
            "FROM users ORDER BY id;")
        running = col_values(r, 1)
        assert running[0] == 30
        assert running[1] == 55

    def test_avg_window(self, populated):
        r = populated.execute(
            "SELECT name, AVG(age) OVER () AS avg_all FROM users LIMIT 1;")
        assert abs(r.rows[0][1] - 29.6) < 0.01

    def test_partition_by(self, multi_table):
        r = multi_table.execute(
            "SELECT name, dept_id, "
            "ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) "
            "AS rn FROM emp;")
        assert r.row_count == 5

    def test_lag(self, populated):
        r = populated.execute(
            "SELECT name, LAG(name, 1) OVER (ORDER BY id) AS prev "
            "FROM users ORDER BY id;")
        assert r.rows[0][1] is None
        assert r.rows[1][1] == 'Alice'

    def test_lead(self, populated):
        r = populated.execute(
            "SELECT name, LEAD(name, 1) OVER (ORDER BY id) AS nxt "
            "FROM users ORDER BY id;")
        assert r.rows[0][1] == 'Bob'
        assert r.rows[4][1] is None

    def test_first_value(self, populated):
        r = populated.execute(
            "SELECT name, FIRST_VALUE(name) OVER (ORDER BY age) AS fv "
            "FROM users;")
        fvs = col_values(r, 1)
        assert all(v == 'Bob' for v in fvs)

    def test_ntile(self, populated):
        r = populated.execute(
            "SELECT name, NTILE(2) OVER (ORDER BY id) AS tile "
            "FROM users;")
        tiles = col_values(r, 1)
        assert set(tiles) == {1, 2}

    def test_window_frame(self, populated):
        r = populated.execute(
            "SELECT name, SUM(age) OVER ("
            "ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING"
            ") AS s FROM users ORDER BY id;")
        assert r.row_count == 5


# ═══════════════════════════════════════════
#  9. 标量函数
# ═══════════════════════════════════════════

class TestScalarFunctions:
    def test_upper_lower(self, eng):
        r = eng.execute("SELECT UPPER('hello'), LOWER('WORLD');")
        assert r.rows[0] == ['HELLO', 'world']

    def test_length(self, eng):
        r = eng.execute("SELECT LENGTH('hello');")
        assert scalar(r) == 5

    def test_trim(self, eng):
        r = eng.execute("SELECT TRIM('  hi  ');")
        assert scalar(r) == 'hi'

    def test_ltrim_rtrim(self, eng):
        r = eng.execute("SELECT LTRIM('  hi'), RTRIM('hi  ');")
        assert r.rows[0] == ['hi', 'hi']

    def test_substr(self, eng):
        r = eng.execute("SELECT SUBSTR('hello', 2, 3);")
        assert scalar(r) == 'ell'

    def test_replace(self, eng):
        r = eng.execute("SELECT REPLACE('hello world', 'world', 'z1db');")
        assert scalar(r) == 'hello z1db'

    def test_reverse(self, eng):
        r = eng.execute("SELECT REVERSE('abc');")
        assert scalar(r) == 'cba'

    def test_repeat(self, eng):
        r = eng.execute("SELECT REPEAT('ab', 3);")
        assert scalar(r) == 'ababab'

    def test_abs(self, eng):
        r = eng.execute("SELECT ABS(-42);")
        assert scalar(r) == 42

    def test_round(self, eng):
        r = eng.execute("SELECT ROUND(3.14159, 2);")
        assert abs(scalar(r) - 3.14) < 0.01

    def test_round_no_decimal(self, eng):
        r = eng.execute("SELECT ROUND(3.7);")
        assert scalar(r) == 4

    def test_ceil_floor(self, eng):
        r = eng.execute("SELECT CEIL(3.2), FLOOR(3.8);")
        assert r.rows[0][0] == 4
        assert r.rows[0][1] == 3

    def test_power_sqrt(self, eng):
        r = eng.execute("SELECT POWER(2, 10), SQRT(144);")
        assert r.rows[0][0] == 1024.0
        assert r.rows[0][1] == 12.0

    def test_sign(self, eng):
        r = eng.execute("SELECT SIGN(-5), SIGN(0), SIGN(5);")
        assert r.rows[0] == [-1, 0, 1]

    def test_mod(self, eng):
        r = eng.execute("SELECT MOD(10, 3);")
        assert scalar(r) == 1

    def test_coalesce(self, eng):
        r = eng.execute("SELECT COALESCE(NULL, NULL, 42);")
        assert scalar(r) == 42

    def test_coalesce_first_non_null(self, eng):
        r = eng.execute("SELECT COALESCE(1, 2);")
        assert scalar(r) == 1

    def test_nullif(self, eng):
        r = eng.execute("SELECT NULLIF(1, 1);")
        assert scalar(r) is None
        r = eng.execute("SELECT NULLIF(1, 2);")
        assert scalar(r) == 1

    def test_if_function(self, eng):
        r = eng.execute("SELECT IF(1 > 0, 'yes', 'no');")
        assert scalar(r) == 'yes'

    def test_concat(self, eng):
        r = eng.execute("SELECT CONCAT('a', 'b', 'c');")
        assert scalar(r) == 'abc'

    def test_concat_ws(self, eng):
        r = eng.execute("SELECT CONCAT_WS('-', 'a', 'b', 'c');")
        assert scalar(r) == 'a-b-c'

    def test_position(self, eng):
        r = eng.execute("SELECT POSITION('lo', 'hello');")
        assert scalar(r) == 4

    def test_left_right(self, eng):
        r = eng.execute("SELECT LEFT('hello', 3), RIGHT('hello', 3);")
        assert r.rows[0] == ['hel', 'llo']

    def test_starts_with(self, eng):
        r = eng.execute("SELECT STARTS_WITH('hello', 'hel');")
        assert scalar(r) is True

    def test_ends_with(self, eng):
        r = eng.execute("SELECT ENDS_WITH('hello', 'llo');")
        assert scalar(r) is True

    def test_contains(self, eng):
        r = eng.execute("SELECT CONTAINS('hello world', 'lo wo');")
        assert scalar(r) is True

    def test_typeof(self, eng):
        r = eng.execute("SELECT TYPEOF(42), TYPEOF('hi'), TYPEOF(NULL);")
        assert 'INT' in r.rows[0][0]
        assert 'VARCHAR' in r.rows[0][1]
        assert r.rows[0][2] == 'NULL'

    def test_initcap(self, eng):
        r = eng.execute("SELECT INITCAP('hello world');")
        assert scalar(r) == 'Hello World'

    def test_ascii_chr(self, eng):
        r = eng.execute("SELECT ASCII('A'), CHR(65);")
        assert r.rows[0] == [65, 'A']

    def test_lpad_rpad(self, eng):
        r = eng.execute("SELECT LPAD('hi', 5, '*'), RPAD('hi', 5, '*');")
        assert r.rows[0] == ['***hi', 'hi***']

    def test_split_part(self, eng):
        r = eng.execute("SELECT SPLIT_PART('a-b-c', '-', 2);")
        assert scalar(r) == 'b'

    def test_ln_exp(self, eng):
        r = eng.execute("SELECT LN(EXP(1));")
        assert abs(scalar(r) - 1.0) < 0.0001

    def test_log10(self, eng):
        r = eng.execute("SELECT LOG10(100);")
        assert abs(scalar(r) - 2.0) < 0.0001

    def test_greatest_least(self, eng):
        r = eng.execute("SELECT GREATEST(1, 5, 3), LEAST(1, 5, 3);")
        assert r.rows[0] == [5, 1]

    def test_function_with_null_arg(self, eng):
        r = eng.execute("SELECT UPPER(NULL);")
        assert scalar(r) is None

    def test_current_date(self, eng):
        r = eng.execute("SELECT CURRENT_DATE;")
        assert scalar(r) is not None

    def test_bit_count(self, eng):
        r = eng.execute("SELECT BIT_COUNT(7);")
        assert scalar(r) == 3


# ═══════════════════════════════════════════
#  10. CTE
# ═══════════════════════════════════════════

class TestCTE:
    def test_simple_cte(self, populated):
        r = populated.execute("""
            WITH young AS (SELECT * FROM users WHERE age < 30)
            SELECT name FROM young ORDER BY name;
        """)
        assert set(col_values(r)) == {'Bob', 'Diana'}

    def test_multiple_ctes(self, populated):
        r = populated.execute("""
            WITH young AS (SELECT * FROM users WHERE age < 30),
                 old AS (SELECT * FROM users WHERE age >= 30)
            SELECT (SELECT COUNT(*) FROM young) AS y,
                   (SELECT COUNT(*) FROM old) AS o;
        """)
        assert r.rows[0][0] == 2
        assert r.rows[0][1] == 3

    def test_recursive_cte(self, eng):
        r = eng.execute("""
            WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 5
            ) SELECT x FROM cnt ORDER BY x;
        """)
        assert col_values(r) == [1, 2, 3, 4, 5]

    def test_cte_with_column_names(self, populated):
        r = populated.execute("""
            WITH info(n, a) AS (SELECT name, age FROM users WHERE id = 1)
            SELECT n, a FROM info;
        """)
        assert r.rows[0] == ['Alice', 30]

    def test_recursive_cte_fibonacci(self, eng):
        r = eng.execute("""
            WITH RECURSIVE fib(n, a, b) AS (
                SELECT 1, 0, 1
                UNION ALL
                SELECT n + 1, b, a + b FROM fib WHERE n < 10
            ) SELECT a FROM fib ORDER BY n;
        """)
        fibs = col_values(r)
        assert fibs[:6] == [0, 1, 1, 2, 3, 5]


# ═══════════════════════════════════════════
#  11. 系统表
# ═══════════════════════════════════════════

class TestSystemTables:
    def test_z1db_tables(self, populated):
        r = populated.execute(
            "SELECT table_name FROM z1db_tables WHERE is_system = FALSE;")
        assert 'users' in col_values(r)

    def test_z1db_columns(self, populated):
        r = populated.execute(
            "SELECT column_name FROM z1db_columns "
            "WHERE table_name = 'users' ORDER BY ordinal_position;")
        assert col_values(r) == ['id', 'name', 'age']

    def test_system_table_after_alter_add(self, eng):
        eng.execute("CREATE TABLE t (id INT, name VARCHAR);")
        eng.execute("ALTER TABLE t ADD COLUMN age INT;")
        r = eng.execute(
            "SELECT column_name FROM z1db_columns "
            "WHERE table_name = 't' ORDER BY ordinal_position;")
        assert col_values(r) == ['id', 'name', 'age']

    def test_system_table_after_alter_drop(self, eng):
        eng.execute("CREATE TABLE t (id INT, a INT, b INT);")
        eng.execute("ALTER TABLE t DROP COLUMN a;")
        r = eng.execute(
            "SELECT column_name FROM z1db_columns "
            "WHERE table_name = 't' ORDER BY ordinal_position;")
        assert col_values(r) == ['id', 'b']

    def test_system_table_after_alter_rename(self, eng):
        eng.execute("CREATE TABLE t (id INT, old_col INT);")
        eng.execute("ALTER TABLE t RENAME COLUMN old_col TO new_col;")
        r = eng.execute(
            "SELECT column_name FROM z1db_columns "
            "WHERE table_name = 't' ORDER BY ordinal_position;")
        assert col_values(r) == ['id', 'new_col']

    def test_system_table_after_drop_table(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("DROP TABLE t;")
        r = eng.execute(
            "SELECT COUNT(*) FROM z1db_tables WHERE table_name = 't';")
        assert scalar(r) == 0
        r = eng.execute(
            "SELECT COUNT(*) FROM z1db_columns WHERE table_name = 't';")
        assert scalar(r) == 0

    def test_system_table_readonly(self, eng):
        with pytest.raises(ExecutionError):
            eng.execute("INSERT INTO z1db_tables VALUES ('hack', FALSE);")
        with pytest.raises(ExecutionError):
            eng.execute("DELETE FROM z1db_tables WHERE table_name = 'z1db_tables';")
        with pytest.raises(ExecutionError):
            eng.execute("DROP TABLE z1db_tables;")

    def test_system_table_where(self, eng):
        eng.execute("CREATE TABLE t1 (id INT);")
        eng.execute("CREATE TABLE t2 (id INT);")
        r = eng.execute(
            "SELECT table_name FROM z1db_tables "
            "WHERE table_name = 't1' AND is_system = FALSE;")
        assert col_values(r) == ['t1']

    def test_system_table_limit(self, eng):
        eng.execute("CREATE TABLE t1 (id INT);")
        eng.execute("CREATE TABLE t2 (id INT);")
        eng.execute("CREATE TABLE t3 (id INT);")
        r = eng.execute(
            "SELECT table_name FROM z1db_tables "
            "WHERE is_system = FALSE ORDER BY table_name LIMIT 2;")
        assert r.row_count == 2

    def test_system_table_count(self, eng):
        eng.execute("CREATE TABLE t1 (id INT);")
        eng.execute("CREATE TABLE t2 (id INT);")
        r = eng.execute(
            "SELECT COUNT(*) FROM z1db_tables WHERE is_system = FALSE;")
        assert scalar(r) == 2

    def test_system_table_star(self, eng):
        eng.execute("CREATE TABLE t1 (id INT);")
        r = eng.execute("SELECT * FROM z1db_tables WHERE table_name = 't1';")
        assert r.row_count == 1
        assert 't1' in r.rows[0]

    def test_system_columns_data_types(self, eng):
        eng.execute("CREATE TABLE t (a INT, b VARCHAR, c BOOLEAN);")
        r = eng.execute(
            "SELECT column_name, data_type FROM z1db_columns "
            "WHERE table_name = 't' ORDER BY ordinal_position;")
        assert r.rows[0] == ['a', 'INT']
        assert r.rows[1] == ['b', 'VARCHAR']
        assert r.rows[2] == ['c', 'BOOLEAN']


# ═══════════════════════════════════════════
#  12. EXPLAIN
# ═══════════════════════════════════════════

class TestExplain:
    def test_explain_select(self, populated):
        r = populated.execute("EXPLAIN SELECT * FROM users WHERE age > 30;")
        assert r.row_count > 0
        assert r.columns == ['Plan']

    def test_explain_join(self, multi_table):
        r = multi_table.execute(
            "EXPLAIN SELECT e.name FROM emp e "
            "JOIN dept d ON e.dept_id = d.id;")
        assert r.row_count > 0

    def test_explain_group_by(self, populated):
        r = populated.execute(
            "EXPLAIN SELECT age, COUNT(*) FROM users GROUP BY age;")
        assert r.row_count > 0


# ═══════════════════════════════════════════
#  13. 事务
# ═══════════════════════════════════════════

class TestTransaction:
    def test_begin_commit(self, populated):
        populated.execute("BEGIN;")
        populated.execute("INSERT INTO users VALUES (6, 'Frank', 40);")
        populated.execute("COMMIT;")
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r) == 6

    def test_begin_rollback(self, populated):
        populated.execute("BEGIN;")
        populated.execute("INSERT INTO users VALUES (6, 'Frank', 40);")
        populated.execute("ROLLBACK;")
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r) == 5

    def test_auto_commit(self, populated):
        populated.execute("INSERT INTO users VALUES (6, 'Frank', 40);")
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r) == 6

    def test_rollback_restores_data(self, populated):
        populated.execute("BEGIN;")
        populated.execute("DELETE FROM users WHERE id = 1;")
        populated.execute("ROLLBACK;")
        r = populated.execute("SELECT name FROM users WHERE id = 1;")
        assert scalar(r) == 'Alice'


# ═══════════════════════════════════════════
#  14. 算术和类型
# ═══════════════════════════════════════════

class TestArithmetic:
    def test_integer_division(self, eng):
        r = eng.execute("SELECT -7 / 2;")
        assert scalar(r) == -3

    def test_positive_division(self, eng):
        r = eng.execute("SELECT 7 / 2;")
        assert scalar(r) == 3

    def test_modulo(self, eng):
        r = eng.execute("SELECT 10 % 3;")
        assert scalar(r) == 1

    def test_division_by_zero(self, eng):
        with pytest.raises(DivisionByZeroError):
            eng.execute("SELECT 1 / 0;")

    def test_modulo_by_zero(self, eng):
        with pytest.raises(DivisionByZeroError):
            eng.execute("SELECT 1 % 0;")

    def test_float_division(self, eng):
        r = eng.execute("SELECT 7.0 / 2;")
        assert abs(scalar(r) - 3.5) < 0.001

    def test_type_promotion(self, eng):
        r = eng.execute("SELECT 1 + 1.5;")
        assert abs(scalar(r) - 2.5) < 0.001

    def test_boolean_in_arithmetic(self, eng):
        r = eng.execute("SELECT TRUE + 1;")
        assert scalar(r) == 2

    def test_constant_folding(self, eng):
        r = eng.execute("SELECT 2 * 3 + 4;")
        assert scalar(r) == 10

    def test_nested_arithmetic(self, eng):
        r = eng.execute("SELECT (1 + 2) * (3 + 4);")
        assert scalar(r) == 21

    def test_unary_minus(self, eng):
        r = eng.execute("SELECT -42;")
        assert scalar(r) == -42

    def test_unary_plus(self, eng):
        r = eng.execute("SELECT +42;")
        assert scalar(r) == 42

    def test_unary_not(self, eng):
        r = eng.execute("SELECT NOT TRUE;")
        assert scalar(r) is False

    def test_string_comparison(self, eng):
        r = eng.execute("SELECT 'abc' < 'abd';")
        assert scalar(r) is True

    def test_int_bigint_promotion(self, eng):
        r = eng.execute("SELECT 1 + CAST(100000000000 AS BIGINT);")
        assert scalar(r) == 100000000001


# ═══════════════════════════════════════════
#  15. NULL 语义
# ═══════════════════════════════════════════

class TestNullSemantics:
    def test_null_eq_null(self, eng):
        r = eng.execute("SELECT NULL = NULL;")
        assert scalar(r) is None

    def test_null_neq_null(self, eng):
        r = eng.execute("SELECT NULL != NULL;")
        assert scalar(r) is None

    def test_null_and_true(self, eng):
        r = eng.execute("SELECT NULL AND TRUE;")
        assert scalar(r) is None

    def test_null_and_false(self, eng):
        r = eng.execute("SELECT NULL AND FALSE;")
        assert scalar(r) is False

    def test_null_or_true(self, eng):
        r = eng.execute("SELECT NULL OR TRUE;")
        assert scalar(r) is True

    def test_null_or_false(self, eng):
        r = eng.execute("SELECT NULL OR FALSE;")
        assert scalar(r) is None

    def test_null_in_agg(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1),(NULL),(3);")
        r = eng.execute("SELECT SUM(v) FROM t;")
        assert scalar(r) == 4

    def test_count_star_vs_count_col(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1),(NULL),(3);")
        r1 = eng.execute("SELECT COUNT(*) FROM t;")
        r2 = eng.execute("SELECT COUNT(v) FROM t;")
        assert scalar(r1) == 3
        assert scalar(r2) == 2

    def test_null_concat(self, eng):
        r = eng.execute("SELECT 'a' || NULL;")
        assert scalar(r) is None

    def test_null_in_where(self, nullable_table):
        r = nullable_table.execute(
            "SELECT COUNT(*) FROM ntest WHERE a > 0;")
        assert scalar(r) == 3

    def test_null_ordering_nulls_last(self, nullable_table):
        r = nullable_table.execute(
            "SELECT id FROM ntest ORDER BY a ASC NULLS LAST;")
        ids = col_values(r)
        assert ids[-1] in (2, 4)
        assert ids[-2] in (2, 4)

    def test_null_ordering_nulls_first(self, nullable_table):
        r = nullable_table.execute(
            "SELECT id FROM ntest ORDER BY a ASC NULLS FIRST;")
        ids = col_values(r)
        assert ids[0] in (2, 4)
        assert ids[1] in (2, 4)

    def test_null_in_group_by(self, nullable_table):
        r = nullable_table.execute(
            "SELECT a, COUNT(*) FROM ntest GROUP BY a ORDER BY a;")
        has_null_group = any(row[0] is None for row in r.rows)
        assert has_null_group

    def test_null_distinct(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1),(NULL),(NULL),(2);")
        r = eng.execute("SELECT DISTINCT v FROM t;")
        assert r.row_count == 3


# ═══════════════════════════════════════════
#  16. 边界条件
# ═══════════════════════════════════════════

class TestEdgeCases:
    def test_empty_table_select(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        r = eng.execute("SELECT * FROM t;")
        assert r.row_count == 0
        assert r.columns == ['id']

    def test_empty_table_agg(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        r = eng.execute("SELECT COUNT(*), SUM(v), MIN(v), MAX(v) FROM t;")
        assert r.rows[0][0] == 0
        for i in range(1, 4):
            assert r.rows[0][i] is None

    def test_single_row(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (42);")
        r = eng.execute("SELECT AVG(v), STDDEV(v) FROM t;")
        assert r.rows[0][0] == 42.0
        assert r.rows[0][1] is None

    def test_limit_zero(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        eng.execute("INSERT INTO t VALUES (1),(2),(3);")
        r = eng.execute("SELECT * FROM t LIMIT 0;")
        assert r.row_count == 0

    def test_offset_beyond_rows(self, populated):
        r = populated.execute("SELECT * FROM users OFFSET 100;")
        assert r.row_count == 0

    def test_large_offset_small_limit(self, populated):
        r = populated.execute("SELECT * FROM users LIMIT 2 OFFSET 4;")
        assert r.row_count == 1

    def test_group_by_no_rows(self, eng):
        eng.execute("CREATE TABLE t (k INT, v INT);")
        r = eng.execute("SELECT k, SUM(v) FROM t GROUP BY k;")
        assert r.row_count == 0

    def test_multiple_inserts_then_count(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        for i in range(100):
            eng.execute(f"INSERT INTO t VALUES ({i});")
        r = eng.execute("SELECT COUNT(*) FROM t;")
        assert scalar(r) == 100

    def test_comment_handling(self, eng):
        r = eng.execute("""
            -- this is a comment
            SELECT 1 + 1; -- inline comment
        """)
        assert scalar(r) == 2

    def test_block_comment(self, eng):
        r = eng.execute("SELECT /* comment */ 42;")
        assert scalar(r) == 42

    def test_empty_string(self, eng):
        eng.execute("CREATE TABLE t (v VARCHAR);")
        eng.execute("INSERT INTO t VALUES ('');")
        r = eng.execute("SELECT LENGTH(v) FROM t;")
        assert scalar(r) == 0

    def test_special_chars_in_string(self, eng):
        eng.execute("CREATE TABLE t (v VARCHAR);")
        eng.execute("INSERT INTO t VALUES ('hello''s world');")
        r = eng.execute("SELECT v FROM t;")
        assert scalar(r) == "hello's world"

    def test_large_string(self, eng):
        eng.execute("CREATE TABLE t (v TEXT);")
        big = 'x' * 10000
        eng.execute(f"INSERT INTO t VALUES ('{big}');")
        r = eng.execute("SELECT LENGTH(v) FROM t;")
        assert scalar(r) == 10000

    def test_negative_numbers(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (-100);")
        r = eng.execute("SELECT v FROM t;")
        assert scalar(r) == -100

    def test_float_precision(self, eng):
        r = eng.execute("SELECT 0.1 + 0.2;")
        assert abs(scalar(r) - 0.3) < 1e-10

    def test_select_star_single_column(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1);")
        r = eng.execute("SELECT * FROM t;")
        assert r.columns == ['v']
        assert r.rows == [[1]]


# ═══════════════════════════════════════════
#  17. 持久化
# ═══════════════════════════════════════════

class TestPersistence:
    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as td:
            e1 = Engine(td)
            e1.execute("CREATE TABLE t (id INT, name VARCHAR);")
            e1.execute("INSERT INTO t VALUES (1, 'Alice');")
            e1.execute("INSERT INTO t VALUES (2, 'Bob');")
            e1.close()

            e2 = Engine(td)
            r = e2.execute("SELECT COUNT(*) FROM t;")
            assert scalar(r) == 2
            r = e2.execute("SELECT name FROM t ORDER BY id;")
            assert col_values(r) == ['Alice', 'Bob']
            e2.close()

    def test_persist_multiple_tables(self):
        with tempfile.TemporaryDirectory() as td:
            e1 = Engine(td)
            e1.execute("CREATE TABLE t1 (id INT);")
            e1.execute("INSERT INTO t1 VALUES (1);")
            e1.execute("CREATE TABLE t2 (v VARCHAR);")
            e1.execute("INSERT INTO t2 VALUES ('hello');")
            e1.close()

            e2 = Engine(td)
            assert e2.table_exists('t1')
            assert e2.table_exists('t2')
            r = e2.execute("SELECT * FROM t2;")
            assert scalar(r) == 'hello'
            e2.close()

    def test_persist_after_delete(self):
        with tempfile.TemporaryDirectory() as td:
            e1 = Engine(td)
            e1.execute("CREATE TABLE t (id INT);")
            e1.execute("INSERT INTO t VALUES (1),(2),(3);")
            e1.execute("DELETE FROM t WHERE id = 2;")
            e1.close()

            e2 = Engine(td)
            r = e2.execute("SELECT id FROM t ORDER BY id;")
            assert col_values(r) == [1, 3]
            e2.close()

    def test_persist_drop_table(self):
        with tempfile.TemporaryDirectory() as td:
            e1 = Engine(td)
            e1.execute("CREATE TABLE t (id INT);")
            e1.execute("DROP TABLE t;")
            e1.close()

            e2 = Engine(td)
            assert not e2.table_exists('t')
            e2.close()


# ═══════════════════════════════════════════
#  18. VACUUM
# ═══════════════════════════════════════════

class TestVacuum:
    def test_vacuum_all(self, populated):
        populated.execute("DELETE FROM users WHERE age < 30;")
        r = populated.execute("VACUUM;")
        assert 'VACUUM' in r.message
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r) == 3

    def test_vacuum_table(self, populated):
        populated.execute("DELETE FROM users WHERE id = 1;")
        r = populated.execute("VACUUM users;")
        assert 'VACUUM' in r.message

    def test_vacuum_preserves_data(self, populated):
        populated.execute("DELETE FROM users WHERE id IN (2, 4);")
        populated.execute("VACUUM;")
        r = populated.execute("SELECT id FROM users ORDER BY id;")
        assert col_values(r) == [1, 3, 5]


# ═══════════════════════════════════════════
#  19. 解析错误
# ═══════════════════════════════════════════

class TestParseErrors:
    def test_syntax_error(self, eng):
        with pytest.raises(ParseError):
            eng.execute("SELEC * FROM t;")

    def test_unclosed_string(self, eng):
        with pytest.raises(ParseError):
            eng.execute("SELECT 'unclosed;")

    def test_extra_token(self, eng):
        with pytest.raises(ParseError):
            eng.execute("SELECT 1 EXTRA;")

    def test_missing_from(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        with pytest.raises((ParseError, TableNotFoundError)):
            eng.execute("SELECT * WHERE id = 1;")

    def test_empty_sql(self, eng):
        with pytest.raises((ParseError, Z1Error)):
            eng.execute(";")


# ═══════════════════════════════════════════
#  20. 语义错误
# ═══════════════════════════════════════════

class TestSemanticErrors:
    def test_table_not_found(self, eng):
        with pytest.raises(TableNotFoundError):
            eng.execute("SELECT * FROM no_such;")

    def test_nested_agg(self, eng):
        eng.execute("CREATE TABLE t (v INT);")
        eng.execute("INSERT INTO t VALUES (1);")
        with pytest.raises(SemanticError):
            eng.execute("SELECT SUM(COUNT(*)) FROM t;")

    def test_bare_col_without_group_by(self, populated):
        with pytest.raises(SemanticError):
            populated.execute("SELECT name, COUNT(*) FROM users;")

    def test_insert_into_nonexistent_table(self, eng):
        with pytest.raises(TableNotFoundError):
            eng.execute("INSERT INTO nope VALUES (1);")

    def test_update_nonexistent_table(self, eng):
        with pytest.raises(TableNotFoundError):
            eng.execute("UPDATE nope SET x = 1;")

    def test_delete_nonexistent_table(self, eng):
        with pytest.raises(TableNotFoundError):
            eng.execute("DELETE FROM nope;")

    def test_insert_nonexistent_column(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        with pytest.raises(ColumnNotFoundError):
            eng.execute("INSERT INTO t (nope) VALUES (1);")

    def test_duplicate_column_in_create(self, eng):
        with pytest.raises(SemanticError):
            eng.execute("CREATE TABLE t (a INT, a INT);")


# ═══════════════════════════════════════════
#  21. 索引
# ═══════════════════════════════════════════

class TestIndex:
    def test_create_and_use_index(self, populated):
        populated.execute("CREATE INDEX idx_age ON users (age);")
        r = populated.execute(
            "SELECT name FROM users WHERE age = 30 ORDER BY name;")
        assert set(col_values(r)) == {'Alice', 'Eve'}

    def test_drop_index(self, populated):
        populated.execute("CREATE INDEX idx_age ON users (age);")
        r = populated.execute("DROP INDEX idx_age;")
        assert r.message == 'OK'

    def test_create_unique_index(self, populated):
        populated.execute("CREATE UNIQUE INDEX idx_id ON users (id);")
        r = populated.execute("SELECT name FROM users WHERE id = 3;")
        assert scalar(r) == 'Charlie'

    def test_create_duplicate_index_raises(self, populated):
        populated.execute("CREATE INDEX idx1 ON users (age);")
        with pytest.raises((DuplicateError, ExecutionError)):
            populated.execute("CREATE INDEX idx1 ON users (name);")

    def test_create_index_if_not_exists(self, populated):
        populated.execute("CREATE INDEX idx1 ON users (age);")
        populated.execute("CREATE INDEX IF NOT EXISTS idx1 ON users (age);")

    def test_drop_nonexistent_index_raises(self, eng):
        with pytest.raises(ExecutionError):
            eng.execute("DROP INDEX nope;")

    def test_drop_index_if_exists(self, eng):
        r = eng.execute("DROP INDEX IF EXISTS nope;")
        assert r.message == 'OK'

    def test_index_survives_insert(self, populated):
        populated.execute("CREATE INDEX idx_age ON users (age);")
        populated.execute("INSERT INTO users VALUES (6, 'Frank', 30);")
        r = populated.execute(
            "SELECT COUNT(*) FROM users WHERE age = 30;")
        assert scalar(r) == 3


# ═══════════════════════════════════════════
#  22. 线程安全
# ═══════════════════════════════════════════

class TestThreadSafety:
    def test_concurrent_reads(self, populated):
        errors = []

        def read_query():
            try:
                r = populated.execute("SELECT COUNT(*) FROM users;")
                assert scalar(r) == 5
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_query) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"线程错误: {errors}"

    def test_concurrent_select_expressions(self, eng):
        errors = []

        def compute():
            try:
                r = eng.execute("SELECT 1 + 2;")
                assert scalar(r) == 3
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ═══════════════════════════════════════════
#  23. generate_series
# ═══════════════════════════════════════════

class TestGenerateSeries:
    def test_basic(self, eng):
        r = eng.execute("SELECT * FROM generate_series(1, 5) AS gs;")
        assert col_values(r) == [1, 2, 3, 4, 5]

    def test_with_step(self, eng):
        r = eng.execute("SELECT * FROM generate_series(0, 10, 3) AS gs;")
        assert col_values(r) == [0, 3, 6, 9]

    def test_negative_step(self, eng):
        r = eng.execute("SELECT * FROM generate_series(5, 1, -1) AS gs;")
        assert col_values(r) == [5, 4, 3, 2, 1]

    def test_single_value(self, eng):
        r = eng.execute("SELECT * FROM generate_series(1, 1) AS gs;")
        assert col_values(r) == [1]


# ═══════════════════════════════════════════
#  24. 混合复杂查询
# ═══════════════════════════════════════════

class TestComplex:
    def test_nested_subquery_with_agg(self, multi_table):
        r = multi_table.execute("""
            SELECT name FROM emp
            WHERE salary > (SELECT AVG(salary) FROM emp)
            ORDER BY name;
        """)
        assert 'Charlie' in col_values(r)

    def test_group_order_limit(self, populated):
        r = populated.execute("""
            SELECT age, COUNT(*) AS cnt FROM users
            GROUP BY age ORDER BY cnt DESC LIMIT 1;
        """)
        assert r.rows[0][0] == 30
        assert r.rows[0][1] == 2

    def test_case_in_agg(self, populated):
        r = populated.execute("""
            SELECT SUM(CASE WHEN age >= 30 THEN 1 ELSE 0 END) FROM users;
        """)
        assert scalar(r) == 3

    def test_having_with_expression(self, multi_table):
        r = multi_table.execute("""
            SELECT dept_id, AVG(salary) AS avg_sal FROM emp
            GROUP BY dept_id HAVING AVG(salary) > 70000;
        """)
        assert r.row_count == 1
        assert r.rows[0][0] == 1

    def test_distinct_order_limit(self, populated):
        r = populated.execute("""
            SELECT DISTINCT age FROM users ORDER BY age DESC LIMIT 2;
        """)
        assert col_values(r) == [35, 30]

    def test_subquery_in_from(self, populated):
        r = populated.execute("""
            SELECT sub.name FROM (
                SELECT name, age FROM users WHERE age >= 30
            ) sub ORDER BY sub.name;
        """)
        assert set(col_values(r)) == {'Alice', 'Charlie', 'Eve'}

    def test_correlated_subquery_count(self, multi_table):
        r = multi_table.execute("""
            SELECT d.name,
                   (SELECT COUNT(*) FROM emp WHERE dept_id = d.id) AS cnt
            FROM dept d ORDER BY d.id;
        """)
        assert r.rows[0][1] == 3
        assert r.rows[1][1] == 2

    def test_union_with_order_limit(self, eng):
        eng.execute("CREATE TABLE a (v INT);")
        eng.execute("INSERT INTO a VALUES (3),(1);")
        eng.execute("CREATE TABLE b (v INT);")
        eng.execute("INSERT INTO b VALUES (4),(2);")
        r = eng.execute("""
            SELECT v FROM a UNION ALL SELECT v FROM b
            ORDER BY v LIMIT 3;
        """)
        assert col_values(r) == [1, 2, 3]

    def test_multiple_aggs_with_filter(self, populated):
        r = populated.execute("""
            SELECT COUNT(*), MIN(age), MAX(age), SUM(age)
            FROM users WHERE age >= 28;
        """)
        assert r.rows[0][0] == 4
        assert r.rows[0][1] == 28
        assert r.rows[0][2] == 35

    def test_join_group_having_order_limit(self, multi_table):
        r = multi_table.execute("""
            SELECT d.name, COUNT(*) AS cnt, AVG(e.salary) AS avg_s
            FROM emp e JOIN dept d ON e.dept_id = d.id
            GROUP BY d.name
            HAVING COUNT(*) >= 2
            ORDER BY avg_s DESC LIMIT 1;
        """)
        assert r.row_count == 1

    def test_nested_case(self, populated):
        r = populated.execute("""
            SELECT name, CASE
                WHEN age < 26 THEN 'A'
                WHEN age < 30 THEN 'B'
                WHEN age = 30 THEN 'C'
                ELSE 'D'
            END AS tier FROM users ORDER BY id;
        """)
        tiers = col_values(r, 1)
        assert tiers == ['C', 'A', 'D', 'B', 'C']

    def test_window_with_filter(self, populated):
        r = populated.execute("""
            SELECT name, age,
                   ROW_NUMBER() OVER (ORDER BY age) AS rn
            FROM users WHERE age >= 28 ORDER BY rn;
        """)
        assert r.row_count == 4
        assert col_values(r, 2) == [1, 2, 3, 4]


# ═══════════════════════════════════════════
#  25. 缓存
# ═══════════════════════════════════════════

class TestResultCache:
    def test_cache_invalidation_on_insert(self, populated):
        r1 = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r1) == 5
        populated.execute("INSERT INTO users VALUES (6, 'Frank', 40);")
        r2 = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r2) == 6

    def test_cache_invalidation_on_delete(self, populated):
        r1 = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r1) == 5
        populated.execute("DELETE FROM users WHERE id = 1;")
        r2 = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r2) == 4

    def test_cache_invalidation_on_update(self, populated):
        r1 = populated.execute("SELECT SUM(age) FROM users;")
        populated.execute("UPDATE users SET age = 100 WHERE id = 1;")
        r2 = populated.execute("SELECT SUM(age) FROM users;")
        assert scalar(r2) != scalar(r1)

    def test_same_query_uses_cache(self, populated):
        r1 = populated.execute("SELECT COUNT(*) FROM users;")
        r2 = populated.execute("SELECT COUNT(*) FROM users;")
        assert scalar(r1) == scalar(r2)


# ═══════════════════════════════════════════
#  26. ANALYZE / 统计信息
# ═══════════════════════════════════════════

class TestAnalyze:
    def test_analyze_table(self, populated):
        stats = populated.analyze_table('users')
        assert stats.row_count == 5
        assert 'age' in stats.column_stats
        cs = stats.column_stats['age']
        assert cs.min_val == 25
        assert cs.max_val == 35
        assert cs.ndv == 4

    def test_get_table_stats(self, populated):
        populated.analyze_table('users')
        stats = populated.get_table_stats('users')
        assert stats is not None
        assert stats.row_count == 5

    def test_stats_null_column(self, nullable_table):
        stats = nullable_table.analyze_table('ntest')
        cs = stats.column_stats['a']
        assert cs.null_count == 2
        assert cs.ndv == 3


# ═══════════════════════════════════════════
#  27. 数据类型
# ═══════════════════════════════════════════

class TestDataTypes:
    def test_bigint(self, eng):
        eng.execute("CREATE TABLE t (v BIGINT);")
        eng.execute("INSERT INTO t VALUES (9999999999999);")
        r = eng.execute("SELECT v FROM t;")
        assert scalar(r) == 9999999999999

    def test_float(self, types_table):
        r = types_table.execute("SELECT f FROM tt WHERE i = 1;")
        assert abs(scalar(r) - 3.14) < 0.01

    def test_boolean(self, types_table):
        r = types_table.execute("SELECT b FROM tt WHERE i = 1;")
        assert scalar(r) is True

    def test_varchar_max_length(self, eng):
        eng.execute("CREATE TABLE t (v VARCHAR(5));")
        eng.execute("INSERT INTO t VALUES ('hello world');")
        r = eng.execute("SELECT v FROM t;")
        assert len(scalar(r)) <= 5

    def test_text_unlimited(self, eng):
        eng.execute("CREATE TABLE t (v TEXT);")
        long_str = 'a' * 5000
        eng.execute(f"INSERT INTO t VALUES ('{long_str}');")
        r = eng.execute("SELECT LENGTH(v) FROM t;")
        assert scalar(r) == 5000


# ═══════════════════════════════════════════
#  28. 多行 INSERT + 大数据量
# ═══════════════════════════════════════════

class TestBulk:
    def test_bulk_insert(self, eng):
        eng.execute("CREATE TABLE t (id INT, v INT);")
        values = ", ".join(f"({i}, {i*10})" for i in range(50))
        eng.execute(f"INSERT INTO t VALUES {values};")
        r = eng.execute("SELECT COUNT(*) FROM t;")
        assert scalar(r) == 50

    def test_bulk_agg(self, eng):
        eng.execute("CREATE TABLE t (id INT, v INT);")
        for i in range(200):
            eng.execute(f"INSERT INTO t VALUES ({i}, {i % 10});")
        r = eng.execute("SELECT v, COUNT(*) FROM t GROUP BY v ORDER BY v;")
        assert r.row_count == 10
        for row in r.rows:
            assert row[1] == 20

    def test_bulk_delete_insert(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        for i in range(100):
            eng.execute(f"INSERT INTO t VALUES ({i});")
        eng.execute("DELETE FROM t WHERE id < 50;")
        r = eng.execute("SELECT COUNT(*) FROM t;")
        assert scalar(r) == 50
        for i in range(100, 150):
            eng.execute(f"INSERT INTO t VALUES ({i});")
        r = eng.execute("SELECT COUNT(*) FROM t;")
        assert scalar(r) == 100


# ═══════════════════════════════════════════
#  29. 结果元数据
# ═══════════════════════════════════════════

class TestResultMetadata:
    def test_result_columns(self, populated):
        r = populated.execute("SELECT name, age FROM users LIMIT 0;")
        assert r.columns == ['name', 'age']

    def test_result_column_types(self, populated):
        from storage.types import DataType
        r = populated.execute("SELECT id, name, age FROM users LIMIT 1;")
        assert DataType.INT in r.column_types or DataType.BIGINT in r.column_types
        assert DataType.VARCHAR in r.column_types

    def test_result_affected_rows(self, populated):
        r = populated.execute("INSERT INTO users VALUES (6, 'Frank', 40);")
        assert r.affected_rows == 1

    def test_result_message(self, eng):
        r = eng.execute("CREATE TABLE t (id INT);")
        assert r.message == 'OK'

    def test_result_timing(self, populated):
        r = populated.execute("SELECT COUNT(*) FROM users;")
        assert r.timing >= 0


# ═══════════════════════════════════════════
#  30. 引擎公共接口
# ═══════════════════════════════════════════

class TestEngineAPI:
    def test_get_table_names(self, populated):
        names = populated.get_table_names()
        assert 'users' in names

    def test_get_table_schema(self, populated):
        schema = populated.get_table_schema('users')
        assert schema.name == 'users'
        assert len(schema.columns) == 3

    def test_get_table_row_count(self, populated):
        count = populated.get_table_row_count('users')
        assert count == 5

    def test_table_exists(self, eng):
        assert not eng.table_exists('nope')
        eng.execute("CREATE TABLE t (id INT);")
        assert eng.table_exists('t')

    def test_get_catalog(self, eng):
        cat = eng.get_catalog()
        assert cat is not None

    def test_get_store(self, populated):
        store = populated.get_store('users')
        assert store.row_count == 5

    def test_memory_budget(self, eng):
        budget = eng.memory_budget
        assert budget is not None
        assert budget.total_limit > 0

    def test_close(self, eng):
        eng.execute("CREATE TABLE t (id INT);")
        eng.close()
        # 关闭后不应崩溃（幂等）


# ═══════════════════════════════════════════
#  31. 日期/时间（基础）
# ═══════════════════════════════════════════

class TestDatetime:
    def test_date_arithmetic(self, eng):
        eng.execute("CREATE TABLE t (d DATE);")
        eng.execute("INSERT INTO t VALUES (CAST('2024-01-15' AS DATE));")
        r = eng.execute("SELECT d + 10 FROM t;")
        val = scalar(r)
        assert val is not None

    def test_to_date(self, eng):
        r = eng.execute("SELECT TO_DATE('2024-01-15');")
        assert scalar(r) is not None

    def test_year_month_day(self, eng):
        r = eng.execute("SELECT YEAR(TO_DATE('2024-03-15'));")
        assert scalar(r) == 2024

    def test_date_diff(self, eng):
        r = eng.execute(
            "SELECT DATE_DIFF(TO_DATE('2024-01-10'), TO_DATE('2024-01-01'));")
        assert scalar(r) == 9


# ═══════════════════════════════════════════
#  32. 数组函数（基础）
# ═══════════════════════════════════════════

class TestArrayFunctions:
    def test_array_length(self, eng):
        r = eng.execute("SELECT ARRAY_LENGTH('[1,2,3]');")
        assert scalar(r) == 3

    def test_array_contains(self, eng):
        r = eng.execute("SELECT ARRAY_CONTAINS('[1,2,3]', 2);")
        assert scalar(r) is True

    def test_array_position(self, eng):
        r = eng.execute("SELECT ARRAY_POSITION('[10,20,30]', 20);")
        assert scalar(r) == 2

    def test_array_join(self, eng):
        r = eng.execute("SELECT ARRAY_JOIN('[1,2,3]', '-');")
        assert scalar(r) == '1-2-3'
