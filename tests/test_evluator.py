from __future__ import annotations
from tests.conftest import *

def test_upper():
    e = make_engine()
    assert_value(e.execute("SELECT UPPER('hello');"), 'HELLO')

def test_lower():
    e = make_engine()
    assert_value(e.execute("SELECT LOWER('WORLD');"), 'world')

def test_length():
    e = make_engine()
    assert_value(e.execute("SELECT LENGTH('test');"), 4)

def test_trim():
    e = make_engine()
    assert_value(e.execute("SELECT TRIM('  hi  ');"), 'hi')

def test_substr():
    e = make_engine()
    assert_value(e.execute("SELECT SUBSTR('hello', 2, 3);"), 'ell')

def test_replace():
    e = make_engine()
    assert_value(e.execute("SELECT REPLACE('aabbcc', 'bb', 'XX');"), 'aaXXcc')

def test_concat():
    e = make_engine()
    assert_value(e.execute("SELECT CONCAT('a', 'b', 'c');"), 'abc')

def test_concat_ws():
    e = make_engine()
    assert_value(e.execute("SELECT CONCAT_WS('-', 'a', 'b', 'c');"), 'a-b-c')

def test_left_right():
    e = make_engine()
    assert_value(e.execute("SELECT LEFT('hello', 3);"), 'hel')
    assert_value(e.execute("SELECT RIGHT('hello', 3);"), 'llo')

def test_repeat():
    e = make_engine()
    assert_value(e.execute("SELECT REPEAT('ab', 3);"), 'ababab')

def test_reverse():
    e = make_engine()
    assert_value(e.execute("SELECT REVERSE('abc');"), 'cba')

def test_position():
    e = make_engine()
    assert_value(e.execute("SELECT POSITION('lo' IN 'hello');"), 4)

def test_lpad_rpad():
    e = make_engine()
    assert_value(e.execute("SELECT LPAD('hi', 5, '*');"), '***hi')
    assert_value(e.execute("SELECT RPAD('hi', 5, '*');"), 'hi***')

def test_lpad_empty_fill():
    e = make_engine()
    # 空填充字符应回退到空格
    r = e.execute("SELECT LPAD('hi', 5);")
    assert len(r.rows[0][0]) == 5

def test_starts_ends_contains():
    e = make_engine()
    assert_value(e.execute("SELECT STARTS_WITH('hello', 'hel');"), True)
    assert_value(e.execute("SELECT ENDS_WITH('hello', 'llo');"), True)
    assert_value(e.execute("SELECT CONTAINS('hello world', 'lo w');"), True)

def test_split_part():
    e = make_engine()
    assert_value(e.execute("SELECT SPLIT_PART('a-b-c', '-', 2);"), 'b')

def test_abs():
    e = make_engine()
    assert_value(e.execute("SELECT ABS(-42);"), 42)

def test_ceil_floor():
    e = make_engine()
    assert_value(e.execute("SELECT CEIL(1.2);"), 2)
    assert_value(e.execute("SELECT FLOOR(1.8);"), 1)

def test_round():
    e = make_engine()
    assert_value(e.execute("SELECT ROUND(3.456, 2);"), 3.46)
    assert_value(e.execute("SELECT ROUND(3.5);"), 4)

def test_power_sqrt():
    e = make_engine()
    assert_value(e.execute("SELECT POWER(2, 10);"), 1024.0)
    assert_value(e.execute("SELECT SQRT(144);"), 12.0)

def test_sign():
    e = make_engine()
    assert_value(e.execute("SELECT SIGN(-5);"), -1)
    assert_value(e.execute("SELECT SIGN(0);"), 0)
    assert_value(e.execute("SELECT SIGN(5);"), 1)

def test_log_exp():
    e = make_engine()
    r = e.execute("SELECT LN(1);")
    assert abs(r.rows[0][0]) < 1e-9
    r = e.execute("SELECT EXP(0);")
    assert abs(r.rows[0][0] - 1.0) < 1e-9

def test_mod():
    e = make_engine()
    assert_value(e.execute("SELECT MOD(10, 3);"), 1)

def test_greatest_least():
    e = make_engine()
    assert_value(e.execute("SELECT GREATEST(1, 5, 3);"), 5)
    assert_value(e.execute("SELECT LEAST(1, 5, 3);"), 1)

def test_coalesce():
    e = make_engine()
    assert_value(e.execute("SELECT COALESCE(NULL, NULL, 42, 99);"), 42)

def test_nullif():
    e = make_engine()
    r = e.execute("SELECT NULLIF(1, 1);")
    assert r.rows[0][0] is None
    assert_value(e.execute("SELECT NULLIF(1, 2);"), 1)

def test_typeof():
    e = make_engine()
    r = e.execute("SELECT TYPEOF(42), TYPEOF('hi'), TYPEOF(NULL);")
    assert r.rows[0][0] == 'INT'
    assert r.rows[0][1] == 'VARCHAR'
    assert r.rows[0][2] == 'NULL'

def test_if_function():
    e = make_engine()
    assert_value(e.execute("SELECT IF(1 > 0, 'yes', 'no');"), 'yes')
    assert_value(e.execute("SELECT IF(1 < 0, 'yes', 'no');"), 'no')

def test_cast():
    e = make_engine()
    assert_value(e.execute("SELECT CAST(42 AS DOUBLE);"), 42.0)
    assert_value(e.execute("SELECT CAST('123' AS INT);"), 123)

def test_between():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
    r = e.execute("SELECT a FROM t WHERE a BETWEEN 2 AND 4;")
    assert_rows(r, [(2,),(3,),(4,)])

def test_like():
    e = make_engine()
    e.execute("CREATE TABLE t (s VARCHAR);")
    e.execute("INSERT INTO t VALUES ('apple'),('banana'),('apricot');")
    r = e.execute("SELECT s FROM t WHERE s LIKE 'ap%';")
    assert_rows(r, [('apple',),('apricot',)])

def test_not_like():
    e = make_engine()
    e.execute("CREATE TABLE t (s VARCHAR);")
    e.execute("INSERT INTO t VALUES ('apple'),('banana');")
    r = e.execute("SELECT s FROM t WHERE s NOT LIKE 'a%';")
    assert_rows(r, [('banana',)])

def test_is_null_expr():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(NULL),(3);")
    r = e.execute("SELECT a FROM t WHERE a IS NULL;")
    assert r.row_count == 1
    assert r.rows[0][0] is None

def test_case_simple():
    e = make_engine()
    r = e.execute("SELECT CASE 1 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END;")
    assert_value(r, 'one')

def test_case_searched():
    e = make_engine()
    r = e.execute("SELECT CASE WHEN 5 > 3 THEN 'yes' ELSE 'no' END;")
    assert_value(r, 'yes')

def test_string_concat_operator():
    e = make_engine()
    assert_value(e.execute("SELECT 'hello' || ' ' || 'world';"), 'hello world')

def test_unary_minus():
    e = make_engine()
    assert_value(e.execute("SELECT -42;"), -42)

def test_unary_not():
    e = make_engine()
    assert_value(e.execute("SELECT NOT TRUE;"), False)
    assert_value(e.execute("SELECT NOT FALSE;"), True)

def test_date_functions():
    e = make_engine()
    r = e.execute("SELECT CURRENT_DATE;")
    assert r.rows[0][0] is not None

def test_width_bucket():
    e = make_engine()
    assert_value(e.execute("SELECT WIDTH_BUCKET(5.5, 0, 10, 5);"), 3)

def test_generate_series_fn():
    e = make_engine()
    r = e.execute("SELECT GENERATE_SERIES(1, 3);")
    # 函数形式返回字符串表示的数组
    assert r.rows[0][0] is not None

def test_null_propagation_functions():
    e = make_engine()
    r = e.execute("SELECT UPPER(NULL);")
    assert r.rows[0][0] is None
    r = e.execute("SELECT ABS(NULL);")
    assert r.rows[0][0] is None
    r = e.execute("SELECT LENGTH(NULL);")
    assert r.rows[0][0] is None
