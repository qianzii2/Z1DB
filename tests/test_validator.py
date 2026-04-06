from __future__ import annotations
from tests.conftest import *
from parser.lexer import Lexer
from parser.parser import Parser
from parser.resolver import Resolver
from parser.validator import Validator

def _validate(sql, engine):
    ast = Parser(Lexer(sql).tokenize()).parse()
    ast = Resolver().resolve(ast, engine._catalog)
    return Validator().validate(ast, engine._catalog)

def test_table_not_found():
    e = make_engine()
    assert_error(lambda: _validate("SELECT * FROM nonexistent;", e))

def test_valid_select():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    _validate("SELECT a FROM t;", e)  # 不应报错

def test_group_by_bare_column():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    assert_error(lambda: _validate("SELECT a, b FROM t GROUP BY a;", e))

def test_group_by_with_agg():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    _validate("SELECT a, SUM(b) FROM t GROUP BY a;", e)

def test_nested_aggregate():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    assert_error(lambda: _validate("SELECT SUM(COUNT(a)) FROM t;", e))

def test_insert_column_not_found():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    assert_error(lambda: _validate("INSERT INTO t (nonexistent) VALUES (1);", e))

def test_create_duplicate_columns():
    e = make_engine()
    assert_error(lambda: _validate("CREATE TABLE t (a INT, a INT);", e))
