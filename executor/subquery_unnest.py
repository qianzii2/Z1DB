from __future__ import annotations
"""子查询去关联化 — 重写关联子查询为JOIN。
EXISTS → SEMI JOIN, NOT EXISTS → ANTI JOIN, IN → SEMI JOIN。
使用正确的SEMI/ANTI join类型（不再错误映射为INNER/LEFT）。"""
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple
from parser.ast import (
    AliasExpr, BinaryExpr, ColumnRef, ExistsExpr, FromClause, InExpr,
    JoinClause, Literal, SelectStmt, SubqueryExpr, TableRef,
)
from storage.types import DataType


class SubqueryUnnester:
    """尝试将子查询重写为JOIN。"""

    _counter = 0

    def unnest(self, ast: SelectStmt) -> SelectStmt:
        """对WHERE中的子查询进行去关联化。"""
        if ast.where is None:
            return ast
        new_where, extra_joins = self._process_predicate(ast.where)
        if not extra_joins:
            return ast
        new_from = ast.from_clause
        if new_from is None:
            return ast
        new_joins = list(new_from.joins) + extra_joins
        new_from = dataclasses.replace(new_from, joins=new_joins)
        return dataclasses.replace(
            ast, where=new_where, from_clause=new_from)

    def _process_predicate(
            self, pred: Any) -> Tuple[Any, List[JoinClause]]:
        extra_joins: List[JoinClause] = []

        if isinstance(pred, BinaryExpr) and pred.op == 'AND':
            left_pred, left_joins = self._process_predicate(pred.left)
            right_pred, right_joins = self._process_predicate(pred.right)
            extra_joins.extend(left_joins)
            extra_joins.extend(right_joins)
            new_pred = BinaryExpr(
                op='AND', left=left_pred, right=right_pred)
            return new_pred, extra_joins

        if isinstance(pred, ExistsExpr):
            join = self._unnest_exists(pred)
            if join:
                extra_joins.append(join)
                return Literal(value=True,
                               inferred_type=DataType.BOOLEAN), extra_joins

        if isinstance(pred, InExpr):
            result = self._unnest_in(pred)
            if result:
                join, new_pred = result
                extra_joins.append(join)
                return new_pred, extra_joins

        return pred, extra_joins

    def _unnest_exists(self, expr: ExistsExpr) -> Optional[JoinClause]:
        """EXISTS → SEMI JOIN, NOT EXISTS → ANTI JOIN。"""
        if not isinstance(expr.query, SelectStmt):
            return None
        subquery = expr.query
        if subquery.from_clause is None:
            return None
        corr_pred = subquery.where
        if corr_pred is None:
            return None

        sub_table = subquery.from_clause.table
        # 使用正确的SEMI/ANTI类型
        jt = 'ANTI' if expr.negated else 'SEMI'

        SubqueryUnnester._counter += 1
        alias = f'__unnest_{SubqueryUnnester._counter}'

        return JoinClause(
            join_type=jt,
            table=TableRef(
                name=sub_table.name, alias=alias,
                subquery=(subquery if sub_table.subquery else None)),
            on=self._remap_correlation(
                corr_pred,
                sub_table.alias or sub_table.name,
                alias))

    def _unnest_in(
            self, expr: InExpr
    ) -> Optional[Tuple[JoinClause, Any]]:
        """IN子查询 → SEMI JOIN, NOT IN → ANTI JOIN。"""
        if not expr.values:
            return None
        subquery_expr = None
        for v in expr.values:
            if isinstance(v, SubqueryExpr):
                subquery_expr = v
                break
        if (subquery_expr is None
                or not isinstance(subquery_expr.query, SelectStmt)):
            return None

        subquery = subquery_expr.query
        if not subquery.select_list or subquery.from_clause is None:
            return None

        SubqueryUnnester._counter += 1
        alias = f'__in_unnest_{SubqueryUnnester._counter}'
        sub_table = subquery.from_clause.table

        sub_col_name = self._get_output_col_name(subquery.select_list[0])
        on_pred = BinaryExpr(
            op='=',
            left=expr.expr,
            right=ColumnRef(table=alias, column=sub_col_name))

        # 使用正确的SEMI/ANTI类型
        jt = 'ANTI' if expr.negated else 'SEMI'
        join = JoinClause(
            join_type=jt,
            table=TableRef(name=sub_table.name, alias=alias),
            on=on_pred)

        # SEMI/ANTI JOIN自身控制输出，不需要额外WHERE条件
        new_pred = Literal(value=True, inferred_type=DataType.BOOLEAN)

        return join, new_pred

    @staticmethod
    def _remap_correlation(pred: Any, old_alias: str,
                           new_alias: str) -> Any:
        """重映射关联谓词中的表引用。"""
        if isinstance(pred, ColumnRef):
            if pred.table == old_alias:
                return ColumnRef(table=new_alias, column=pred.column)
            return pred
        if isinstance(pred, BinaryExpr):
            return BinaryExpr(
                op=pred.op,
                left=SubqueryUnnester._remap_correlation(
                    pred.left, old_alias, new_alias),
                right=SubqueryUnnester._remap_correlation(
                    pred.right, old_alias, new_alias))
        return pred

    @staticmethod
    def _get_output_col_name(expr: Any) -> str:
        if isinstance(expr, ColumnRef):
            return expr.column
        if isinstance(expr, AliasExpr):
            return expr.alias
        return '__col'
