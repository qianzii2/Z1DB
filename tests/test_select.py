from __future__ import annotations
from tests.conftest import *

def test_select_star():
    e = make_engine_with_data()
    r = e.execute("SELECT * FROM users;")
    assert r.row_count == 5
    assert len(r.columns) == 5

def test_select_columns():
    e = make_engine_with_data()
    r = e.execute("SELECT name, age FROM users;")
    assert r.columns == ['name', 'age']

def test_where_eq():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE id = 1;")
    assert_value(r, 'Alice')

def test_where_gt():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE age > 28;")
    assert_rows(r, [('Alice',), ('Charlie',)])

def test_where_and():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE age > 25 AND active = TRUE;")
    assert_rows(r, [('Alice',), ('Diana',)])

def test_where_or():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE age < 26 OR age > 34;")
    assert_rows(r, [('Bob',), ('Charlie',)])

def test_where_in():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE id IN (1, 3);")
    assert_rows(r, [('Alice',), ('Charlie',)])

def test_where_not_in():
    e = make_engine_with_data()
    r = e.execute("SELECT id FROM users WHERE id NOT IN (1, 2, 3);")
    assert_rows(r, [(4,), (5,)])

def test_where_between():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE age BETWEEN 25 AND 30;")
    assert_rows(r, [('Alice',), ('Bob',), ('Diana',)])

def test_where_like():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE name LIKE 'A%';")
    assert_value(r, 'Alice')

def test_where_is_null():
    e = make_engine_with_data()
    r = e.execute("SELECT id FROM users WHERE salary IS NULL;")
    assert_rows(r, [(4,), (5,)])

def test_distinct():
    e = make_engine_with_data()
    r = e.execute("SELECT DISTINCT active FROM users WHERE active IS NOT NULL;")
    assert r.row_count == 2

def test_order_by_asc():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE age IS NOT NULL ORDER BY age ASC;")
    assert r.rows[0][0] == 'Bob'

def test_order_by_desc():
    e = make_engine_with_data()
    r = e.execute("SELECT name FROM users WHERE age IS NOT NULL ORDER BY age DESC;")
    assert r.rows[0][0] == 'Charlie'

def test_limit():
    e = make_engine_with_data()
    r = e.execute("SELECT * FROM users LIMIT 2;")
    assert r.row_count == 2

def test_offset():
    e = make_engine_with_data()
    r = e.execute("SELECT id FROM users ORDER BY id LIMIT 2 OFFSET 2;")
    assert_rows_ordered(r, [(3,), (4,)])

def test_expression_select():
    e = make_engine_with_data()
    r = e.execute("SELECT name, salary * 1.1 AS new_sal FROM users WHERE id = 1;")
    assert abs(r.rows[0][1] - 82500.0) < 0.01

def test_alias():
    e = make_engine_with_data()
    r = e.execute("SELECT name AS n FROM users WHERE id = 1;")
    assert r.columns == ['n']

def test_no_from():
    e = make_engine()
    r = e.execute("SELECT 1 + 2, 'hello';")
    assert r.rows[0][0] == 3
    assert r.rows[0][1] == 'hello'

def test_select_case():
    e = make_engine_with_data()
    r = e.execute("""
        SELECT name, CASE WHEN age > 30 THEN 'senior'
                          WHEN age IS NOT NULL THEN 'junior'
                          ELSE 'unknown' END AS level
        FROM users;
    """)
    assert r.row_count == 5

def test_select_cast():
    e = make_engine_with_data()
    r = e.execute("SELECT CAST(age AS DOUBLE) FROM users WHERE id = 1;")
    assert isinstance(r.rows[0][0], float)

def test_explain():
    e = make_engine_with_data()
    r = e.execute("EXPLAIN SELECT * FROM users WHERE id = 1;")
    assert r.row_count > 0
