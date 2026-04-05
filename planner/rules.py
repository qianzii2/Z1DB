from __future__ import annotations
"""优化规则 — 真正的谓词下推 + 谓词重排 + TopN推入。"""
import dataclasses
from typing import Any, List, Optional, Set
from parser.ast import (
    AliasExpr, BinaryExpr, ColumnRef, FromClause, JoinClause, Literal,
    SelectStmt, StarExpr, UnaryExpr, IsNullExpr, AggregateCall,
    TableRef,
)


class PredicatePushdown:
    """将WHERE中的单表谓词下推到JOIN的输入侧。

    SELECT * FROM A JOIN B ON A.id = B.id WHERE A.x > 10 AND B.y < 5
    →
    SELECT * FROM
      (SELECT * FROM A WHERE A.x > 10) A
      JOIN
      (SELECT * FROM B WHERE B.y < 5) B
      ON A.id = B.id

    减少JOIN输入规模 → 更快执行。"""

    @staticmethod
    def apply(ast: SelectStmt) -> SelectStmt:
        if ast.where is None or ast.from_clause is None:
            return ast
        if not ast.from_clause.joins:
            return ast

        # 收集所有表名/别名
        table_aliases: dict = {}  # alias → name
        base = ast.from_clause.table
        base_alias = base.alias or base.name
        table_aliases[base_alias] = base.name
        for jc in ast.from_clause.joins:
            if jc.table:
                ja = jc.table.alias or jc.table.name
                table_aliases[ja] = jc.table.name

        # 分解AND
        conjuncts = PredicatePushdown._split_and(ast.where)

        # 分类：单表谓词 vs 多表谓词
        single_table: dict = {}  # table_alias → [pred, ...]
        remaining: list = []

        for pred in conjuncts:
            pred_tables = PredicatePushdown._tables_referenced(pred)
            if len(pred_tables) == 1:
                t = next(iter(pred_tables))
                if t in table_aliases:
                    single_table.setdefault(t, []).append(pred)
                else:
                    remaining.append(pred)
            elif len(pred_tables) == 0:
                # 常量谓词，保留在WHERE
                remaining.append(pred)
            else:
                remaining.append(pred)

        if not single_table:
            return ast  # 没有可下推的

        # 对基表应用下推：包装为带WHERE的子查询TableRef
        new_base = base
        if base_alias in single_table:
            pushed = PredicatePushdown._combine_and(
                single_table[base_alias])
            if pushed and not base.subquery:
                # 构造 (SELECT * FROM table WHERE pushed) alias
                inner_select = SelectStmt(
                    select_list=[StarExpr()],
                    from_clause=FromClause(
                        table=TableRef(name=base.name), joins=[]),
                    where=PredicatePushdown._strip_table_prefix(
                        pushed, base_alias))
                new_base = TableRef(
                    name=base_alias, alias=base_alias,
                    subquery=inner_select)

        # 对JOIN表应用下推
        new_joins = []
        for jc in ast.from_clause.joins:
            if jc.table is None:
                new_joins.append(jc)
                continue
            ja = jc.table.alias or jc.table.name
            if ja in single_table and not jc.table.subquery:
                pushed = PredicatePushdown._combine_and(
                    single_table[ja])
                if pushed:
                    inner_select = SelectStmt(
                        select_list=[StarExpr()],
                        from_clause=FromClause(
                            table=TableRef(name=jc.table.name), joins=[]),
                        where=PredicatePushdown._strip_table_prefix(
                            pushed, ja))
                    new_tref = TableRef(
                        name=ja, alias=ja, subquery=inner_select)
                    new_joins.append(dataclasses.replace(
                        jc, table=new_tref))
                else:
                    new_joins.append(jc)
            else:
                new_joins.append(jc)

        new_where = (PredicatePushdown._combine_and(remaining)
                     if remaining else None)
        new_from = FromClause(table=new_base, joins=new_joins)
        return dataclasses.replace(
            ast, where=new_where, from_clause=new_from)

    @staticmethod
    def _strip_table_prefix(pred: Any, alias: str) -> Any:
        """移除谓词中的表前缀（下推后不再需要）。"""
        if isinstance(pred, ColumnRef):
            if pred.table == alias:
                return ColumnRef(table=None, column=pred.column)
            return pred
        if isinstance(pred, BinaryExpr):
            return BinaryExpr(
                op=pred.op,
                left=PredicatePushdown._strip_table_prefix(
                    pred.left, alias),
                right=PredicatePushdown._strip_table_prefix(
                    pred.right, alias))
        if isinstance(pred, UnaryExpr):
            return UnaryExpr(
                op=pred.op,
                operand=PredicatePushdown._strip_table_prefix(
                    pred.operand, alias))
        if isinstance(pred, IsNullExpr):
            return IsNullExpr(
                expr=PredicatePushdown._strip_table_prefix(
                    pred.expr, alias),
                negated=pred.negated)
        if dataclasses.is_dataclass(pred) and not isinstance(pred, type):
            changes = {}
            for f in dataclasses.fields(pred):
                child = getattr(pred, f.name)
                if isinstance(child, list):
                    changes[f.name] = [
                        PredicatePushdown._strip_table_prefix(i, alias)
                        for i in child]
                elif (dataclasses.is_dataclass(child)
                      and not isinstance(child, type)):
                    changes[f.name] = \
                        PredicatePushdown._strip_table_prefix(child, alias)
            return dataclasses.replace(pred, **changes) if changes else pred
        return pred

    @staticmethod
    def _split_and(expr: Any) -> list:
        if isinstance(expr, BinaryExpr) and expr.op == 'AND':
            return (PredicatePushdown._split_and(expr.left)
                    + PredicatePushdown._split_and(expr.right))
        return [expr]

    @staticmethod
    def _combine_and(exprs: list) -> Any:
        if not exprs:
            return None
        result = exprs[0]
        for e in exprs[1:]:
            result = BinaryExpr(op='AND', left=result, right=e)
        return result

    @staticmethod
    def _tables_referenced(expr: Any) -> Set[str]:
        if isinstance(expr, ColumnRef):
            return {expr.table} if expr.table else set()
        if isinstance(expr, AliasExpr):
            return PredicatePushdown._tables_referenced(expr.expr)
        if not dataclasses.is_dataclass(expr) or isinstance(expr, type):
            return set()
        result: Set[str] = set()
        for f in dataclasses.fields(expr):
            child = getattr(expr, f.name)
            if isinstance(child, list):
                for item in child:
                    result |= PredicatePushdown._tables_referenced(item)
            elif (dataclasses.is_dataclass(child)
                  and not isinstance(child, type)):
                result |= PredicatePushdown._tables_referenced(child)
        return result


class PredicateReorder:
    """按估计选择率重排AND子句。最具选择性的谓词先执行。"""

    @staticmethod
    def apply(ast: SelectStmt) -> SelectStmt:
        if ast.where is None:
            return ast
        conjuncts = PredicatePushdown._split_and(ast.where)
        if len(conjuncts) <= 1:
            return ast
        scored = [(PredicateReorder._score(c), c) for c in conjuncts]
        scored.sort(key=lambda x: x[0])
        reordered = [c for _, c in scored]
        new_where = PredicatePushdown._combine_and(reordered)
        return dataclasses.replace(ast, where=new_where)

    @staticmethod
    def _score(pred: Any) -> float:
        """分数越低越先执行（更具选择性或更便宜）。"""
        if isinstance(pred, BinaryExpr):
            if pred.op == '=' and isinstance(pred.right, Literal):
                return 0.1
            if pred.op == '!=' and isinstance(pred.right, Literal):
                return 0.9
            if pred.op in ('<', '>', '<=', '>='):
                return 0.33
        if isinstance(pred, IsNullExpr):
            return 0.05
        from parser.ast import LikeExpr, InExpr, BetweenExpr
        if isinstance(pred, LikeExpr):
            return 0.4
        if isinstance(pred, InExpr):
            return 0.2
        if isinstance(pred, BetweenExpr):
            return 0.25
        return 0.5


class TopNPushdown:
    """ORDER BY + LIMIT → TopN算子（堆排序O(n log K)）。"""

    @staticmethod
    def should_use_top_n(ast: SelectStmt) -> bool:
        if not ast.order_by or ast.limit is None:
            return False
        if (isinstance(ast.limit, Literal)
                and isinstance(ast.limit.value, int)):
            return ast.limit.value > 0
        return False

    @staticmethod
    def get_limit_value(ast: SelectStmt) -> int:
        if (isinstance(ast.limit, Literal)
                and isinstance(ast.limit.value, int)):
            return ast.limit.value
        return 0
