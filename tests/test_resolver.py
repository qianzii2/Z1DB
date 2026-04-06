from __future__ import annotations
from tests.conftest import *
from parser.lexer import Lexer
from parser.parser import Parser
from parser.resolver import Resolver
from parser.ast import ColumnRef, AliasExpr

def _resolve(sql, engine):
    ast = Parser(Lexer(sql).tokenize()).parse()
    return Resolver().resolve(ast, engine._catalog)

def test_star_expansion():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT, c INT);")
    ast = _resolve("SELECT * FROM t;", e)
    assert len(ast.select_list) == 3
    assert all(isinstance(x, ColumnRef) for x in ast.select_list)

def test_table_star():
    e = make_engine()
    e.execute("CREATE TABLE t (x INT, y INT);")
    ast = _resolve("SELECT t.* FROM t;", e)
    assert len(ast.select_list) == 2

def test_join_qualified():
    e = make_engine()
    e.execute("CREATE TABLE a (id INT, x INT);")
    e.execute("CREATE TABLE b (id INT, y INT);")
    ast = _resolve("SELECT * FROM a JOIN b ON a.id = b.id;", e)
    assert len(ast.select_list) == 4  # a.id, a.x, b.id, b.y

def test_alias_normalize():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    ast = _resolve("SELECT a FROM t AS x;", e)
    # 单表别名应被标准化
    assert isinstance(ast.select_list[0], ColumnRef)

def test_order_by_alias():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    ast = _resolve("SELECT a AS x FROM t ORDER BY x;", e)
    # ORDER BY x 应被解析为对 a 列的引用
    assert ast.order_by[0].expr is not None

def test_order_by_position():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    ast = _resolve("SELECT a, b FROM t ORDER BY 2;", e)
    # ORDER BY 2 → b 列
    order_expr = ast.order_by[0].expr
    assert isinstance(order_expr, ColumnRef) and order_expr.column == 'b'
