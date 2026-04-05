from __future__ import annotations
"""查询优化器：常量折叠、谓词简化。
通过AliasExpr包装保留原始列名。"""
import dataclasses
from typing import Any, Optional
from parser.ast import (
    AliasExpr, BinaryExpr, CaseExpr, ColumnRef, InExpr, BetweenExpr,
    LikeExpr, Literal, SelectStmt, StarExpr, UnaryExpr, IsNullExpr,
    AggregateCall, FunctionCall,
)
from parser.formatter import Formatter
from storage.types import DataType


class QueryOptimizer:
    def optimize(self, ast: Any) -> Any:
        if not isinstance(ast, SelectStmt):
            return ast
        ast = self._constant_fold(ast)
        ast = self._simplify_predicates(ast)
        return ast

    def _constant_fold(self, ast: SelectStmt) -> SelectStmt:
        changes: dict = {}
        new_select = []
        for expr in ast.select_list:
            original_name = Formatter.expr_to_sql(expr)
            folded = self._fold(expr)
            if (isinstance(folded, Literal) and not isinstance(expr, Literal)
                    and not isinstance(expr, AliasExpr)):
                folded = AliasExpr(expr=folded, alias=original_name)
            elif isinstance(expr, AliasExpr):
                folded_inner = self._fold(expr.expr)
                folded = dataclasses.replace(expr, expr=folded_inner)
            new_select.append(folded)
        changes['select_list'] = new_select
        if ast.where:
            changes['where'] = self._fold(ast.where)
        if ast.having:
            changes['having'] = self._fold(ast.having)
        if ast.order_by:
            changes['order_by'] = [
                dataclasses.replace(sk, expr=self._fold(sk.expr))
                for sk in ast.order_by
            ]
        return dataclasses.replace(ast, **changes)

    def _fold(self, expr: Any) -> Any:
        if expr is None:
            return None
        if isinstance(expr, (Literal, ColumnRef, StarExpr)):
            return expr
        if isinstance(expr, AliasExpr):
            return dataclasses.replace(expr, expr=self._fold(expr.expr))
        if isinstance(expr, BinaryExpr):
            left = self._fold(expr.left)
            right = self._fold(expr.right)
            # 双方都是字面量才能折叠
            if isinstance(left, Literal) and isinstance(right, Literal):
                result = self._eval_const_binary(expr.op, left, right)
                if result is not None:
                    return result
            # 代数简化 — 只在另一侧也是Literal时才做，防止NULL语义错误
            if expr.op == '+':
                if isinstance(right, Literal) and right.value == 0:
                    return left
                if isinstance(left, Literal) and left.value == 0:
                    return right
            if expr.op == '*':
                if isinstance(right, Literal) and right.value == 1:
                    return left
                if isinstance(left, Literal) and left.value == 1:
                    return right
                # x * 0: 只有双方都是Literal才折叠（因为NULL*0=NULL）
                if (isinstance(right, Literal) and right.value == 0
                        and isinstance(left, Literal)):
                    return Literal(value=0, inferred_type=DataType.INT)
                if (isinstance(left, Literal) and left.value == 0
                        and isinstance(right, Literal)):
                    return Literal(value=0, inferred_type=DataType.INT)
            if expr.op == 'AND':
                if isinstance(left, Literal) and left.value is True:
                    return right
                if isinstance(right, Literal) and right.value is True:
                    return left
                if isinstance(left, Literal) and left.value is False:
                    return Literal(value=False, inferred_type=DataType.BOOLEAN)
                if isinstance(right, Literal) and right.value is False:
                    return Literal(value=False, inferred_type=DataType.BOOLEAN)
            if expr.op == 'OR':
                if isinstance(left, Literal) and left.value is False:
                    return right
                if isinstance(right, Literal) and right.value is False:
                    return left
                if isinstance(left, Literal) and left.value is True:
                    return Literal(value=True, inferred_type=DataType.BOOLEAN)
                if isinstance(right, Literal) and right.value is True:
                    return Literal(value=True, inferred_type=DataType.BOOLEAN)
            return dataclasses.replace(expr, left=left, right=right)
        if isinstance(expr, UnaryExpr):
            operand = self._fold(expr.operand)
            if isinstance(operand, Literal):
                if expr.op == '-' and isinstance(operand.value, (int, float)):
                    return Literal(value=-operand.value,
                                   inferred_type=operand.inferred_type)
                if expr.op == '+':
                    return operand
                if expr.op == 'NOT' and isinstance(operand.value, bool):
                    return Literal(value=not operand.value,
                                   inferred_type=DataType.BOOLEAN)
            return dataclasses.replace(expr, operand=operand)
        if isinstance(expr, IsNullExpr):
            inner = self._fold(expr.expr)
            if isinstance(inner, Literal):
                result = inner.value is None
                if expr.negated:
                    result = not result
                return Literal(value=result, inferred_type=DataType.BOOLEAN)
            return dataclasses.replace(expr, expr=inner)
        if isinstance(expr, CaseExpr):
            new_whens = [(self._fold(c), self._fold(r))
                         for c, r in expr.when_clauses]
            return dataclasses.replace(
                expr,
                operand=self._fold(expr.operand) if expr.operand else None,
                when_clauses=new_whens,
                else_expr=self._fold(expr.else_expr) if expr.else_expr else None)
        if isinstance(expr, InExpr):
            return dataclasses.replace(
                expr, expr=self._fold(expr.expr),
                values=[self._fold(v) for v in expr.values])
        if isinstance(expr, BetweenExpr):
            return dataclasses.replace(
                expr, expr=self._fold(expr.expr),
                low=self._fold(expr.low), high=self._fold(expr.high))
        if isinstance(expr, LikeExpr):
            return dataclasses.replace(
                expr, expr=self._fold(expr.expr),
                pattern=self._fold(expr.pattern))
        if isinstance(expr, (AggregateCall, FunctionCall)):
            return dataclasses.replace(
                expr, args=[self._fold(a) for a in expr.args])
        # 通用dataclass递归
        if dataclasses.is_dataclass(expr) and not isinstance(expr, type):
            changes: dict = {}
            for f in dataclasses.fields(expr):
                child = getattr(expr, f.name)
                if isinstance(child, list):
                    changes[f.name] = [self._fold(item) for item in child]
                elif isinstance(child, tuple):
                    changes[f.name] = tuple(self._fold(item) for item in child)
                elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                    changes[f.name] = self._fold(child)
            return dataclasses.replace(expr, **changes) if changes else expr
        return expr

    def _eval_const_binary(self, op: str, left: Literal,
                           right: Literal) -> Optional[Literal]:
        lv, rv = left.value, right.value
        if lv is None or rv is None:
            # NULL特殊规则
            if op == 'AND' and (lv is False or rv is False):
                return Literal(value=False, inferred_type=DataType.BOOLEAN)
            if op == 'OR' and (lv is True or rv is True):
                return Literal(value=True, inferred_type=DataType.BOOLEAN)
            return Literal(value=None, inferred_type=DataType.UNKNOWN)
        try:
            if op == '+':
                return Literal(value=lv + rv,
                               inferred_type=self._arith_type(left, right))
            if op == '-':
                return Literal(value=lv - rv,
                               inferred_type=self._arith_type(left, right))
            if op == '*':
                return Literal(value=lv * rv,
                               inferred_type=self._arith_type(left, right))
            if op == '/':
                if rv == 0:
                    return None
                if isinstance(lv, int) and isinstance(rv, int):
                    return Literal(value=int(lv / rv),
                                   inferred_type=DataType.INT)
                return Literal(value=lv / rv, inferred_type=DataType.DOUBLE)
            if op == '%':
                if rv == 0:
                    return None
                return Literal(value=lv % rv,
                               inferred_type=self._arith_type(left, right))
            if op == '||':
                return Literal(value=str(lv) + str(rv),
                               inferred_type=DataType.VARCHAR)
            if op == '=':
                return Literal(value=lv == rv, inferred_type=DataType.BOOLEAN)
            if op == '!=':
                return Literal(value=lv != rv, inferred_type=DataType.BOOLEAN)
            if op == '<':
                return Literal(value=lv < rv, inferred_type=DataType.BOOLEAN)
            if op == '>':
                return Literal(value=lv > rv, inferred_type=DataType.BOOLEAN)
            if op == '<=':
                return Literal(value=lv <= rv, inferred_type=DataType.BOOLEAN)
            if op == '>=':
                return Literal(value=lv >= rv, inferred_type=DataType.BOOLEAN)
            if op == 'AND':
                return Literal(value=bool(lv and rv),
                               inferred_type=DataType.BOOLEAN)
            if op == 'OR':
                return Literal(value=bool(lv or rv),
                               inferred_type=DataType.BOOLEAN)
        except Exception:
            return None
        return None

    def _arith_type(self, left: Literal, right: Literal) -> DataType:
        if isinstance(left.value, float) or isinstance(right.value, float):
            return DataType.DOUBLE
        return (left.inferred_type
                if left.inferred_type != DataType.UNKNOWN
                else DataType.INT)

    def _simplify_predicates(self, ast: SelectStmt) -> SelectStmt:
        if ast.where:
            if isinstance(ast.where, Literal) and ast.where.value is True:
                return dataclasses.replace(ast, where=None)
        return ast
