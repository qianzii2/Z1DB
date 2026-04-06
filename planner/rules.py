from __future__ import annotations
"""优化规则 — 谓词下推 + 谓词重排 + TopN 推入。
[FIX-B10] 不构造子查询，将单表谓词合并到 JOIN ON。
[P04] CROSS JOIN 不合并谓词到 ON（否则语义改变）。"""
import dataclasses
from typing import Any, List, Optional, Set
from parser.ast import (
    AliasExpr, BinaryExpr, ColumnRef, FromClause, JoinClause, Literal,
    SelectStmt, StarExpr, UnaryExpr, IsNullExpr, AggregateCall,
    TableRef,
)


class PredicatePushdown:
    """将 WHERE 中的单表谓词下推到 JOIN ON 条件中。
    减少 JOIN 输入规模 → 更快执行。"""

    @staticmethod
    def apply(ast: SelectStmt) -> SelectStmt:
        if ast.where is None or ast.from_clause is None:
            return ast
        if not ast.from_clause.joins:
            return ast

        # 收集所有表名/别名
        table_aliases: dict = {}
        base = ast.from_clause.table
        base_alias = base.alias or base.name
        table_aliases[base_alias] = base.name
        for jc in ast.from_clause.joins:
            if jc.table:
                ja = jc.table.alias or jc.table.name
                table_aliases[ja] = jc.table.name

        conjuncts = PredicatePushdown._split_and(ast.where)

        # 分类：单表谓词 vs 多表/无表谓词
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
            else:
                remaining.append(pred)

        if not single_table:
            return ast

        # 对 JOIN 表应用下推（合并到 ON 条件）
        new_joins = []
        for jc in ast.from_clause.joins:
            if jc.table is None:
                new_joins.append(jc)
                continue
            ja = jc.table.alias or jc.table.name
            if ja in single_table:
                # [P04] CROSS JOIN 不合并谓词到 ON（会改变语义为 INNER JOIN）
                if jc.join_type == 'CROSS':
                    remaining.extend(single_table[ja])
                    new_joins.append(jc)
                else:
                    pushed = PredicatePushdown._combine_and(
                        single_table[ja])
                    if pushed and jc.on:
                        combined_on = BinaryExpr(
                            op='AND', left=jc.on, right=pushed)
                        new_joins.append(
                            dataclasses.replace(jc, on=combined_on))
                    elif pushed:
                        new_joins.append(
                            dataclasses.replace(jc, on=pushed))
                    else:
                        new_joins.append(jc)
            else:
                new_joins.append(jc)

        # 基表的单表谓词保留在 WHERE
        base_preds = single_table.get(base_alias, [])
        all_remaining = base_preds + remaining
        new_where = (PredicatePushdown._combine_and(all_remaining)
                     if all_remaining else None)

        new_from = dataclasses.replace(
            ast.from_clause, joins=new_joins)
        return dataclasses.replace(
            ast, where=new_where, from_clause=new_from)

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
    """按估计选择率重排 AND 子句。最具选择性的谓词先执行。"""

    @staticmethod
    def apply(ast: SelectStmt) -> SelectStmt:
        if ast.where is None:
            return ast
        conjuncts = PredicatePushdown._split_and(ast.where)
        if len(conjuncts) <= 1:
            return ast
        scored = [(PredicateReorder._score(c), c)
                  for c in conjuncts]
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
    """ORDER BY + LIMIT → TopN 算子（堆排序 O(n log K)）。"""

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
