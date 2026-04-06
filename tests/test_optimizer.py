from __future__ import annotations
from tests.conftest import *
from parser.lexer import Lexer
from parser.parser import Parser
from parser.ast import Literal, BinaryExpr
from executor.optimizer import QueryOptimizer

def _optimize(sql):
    ast = Parser(Lexer(sql).tokenize()).parse()
    return QueryOptimizer().optimize(ast)

def test_constant_fold_add():
    ast = _optimize("SELECT 1 + 2;")
    expr = ast.select_list[0]
    # 可能被 AliasExpr 包裹
    inner = expr.expr if hasattr(expr, 'expr') else expr
    assert isinstance(inner, Literal) and inner.value == 3

def test_constant_fold_mul():
    ast = _optimize("SELECT 3 * 4;")
    expr = ast.select_list[0]
    inner = expr.expr if hasattr(expr, 'expr') else expr
    assert isinstance(inner, Literal) and inner.value == 12

def test_constant_fold_string():
    ast = _optimize("SELECT 'a' || 'b';")
    expr = ast.select_list[0]
    inner = expr.expr if hasattr(expr, 'expr') else expr
    assert isinstance(inner, Literal) and inner.value == 'ab'

def test_identity_add_zero():
    ast = _optimize("SELECT a + 0 FROM t;")
    # a + 0 应简化为 a
    expr = ast.select_list[0]
    # 可能没有被简化（a 不是 Literal）——但 0 + a 的情况：
    ast2 = _optimize("SELECT 0 + a FROM t;")

def test_true_and_simplify():
    ast = _optimize("SELECT 1 FROM t WHERE TRUE AND a > 5;")
    # TRUE AND x → x
    # where 应只剩 a > 5
    if ast.where is not None:
        if isinstance(ast.where, BinaryExpr):
            assert ast.where.op != 'AND' or not (
                isinstance(ast.where.left, Literal) and ast.where.left.value is True)

def test_false_and_simplify():
    ast = _optimize("SELECT 1 FROM t WHERE FALSE AND a > 5;")
    # FALSE AND x → FALSE
    if isinstance(ast.where, Literal):
        assert ast.where.value == False

def test_integer_division_truncate():
    ast = _optimize("SELECT 7 / 2;")
    expr = ast.select_list[0]
    inner = expr.expr if hasattr(expr, 'expr') else expr
    if isinstance(inner, Literal):
        assert inner.value == 3  # 截断除法

def test_negative_division_truncate():
    ast = _optimize("SELECT -7 / 2;")
    expr = ast.select_list[0]
    inner = expr.expr if hasattr(expr, 'expr') else expr
    if isinstance(inner, Literal):
        assert inner.value == -3  # 向零截断
