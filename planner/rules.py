from __future__ import annotations

"""Optimization rules — predicate pushdown, projection pushdown, etc.
Each rule transforms an AST, preserving semantics while reducing cost."""
import dataclasses
from typing import Any, List, Optional, Set
from parser.ast import (
    AliasExpr, BinaryExpr, ColumnRef, FromClause, JoinClause, Literal,
    SelectStmt, StarExpr, UnaryExpr, IsNullExpr, AggregateCall,
)


class PredicatePushdown:
    """Push WHERE predicates down into JOIN inputs.

    Example:
      SELECT * FROM A JOIN B ON A.id = B.id WHERE A.x > 10
    →
      SELECT * FROM (SELECT * FROM A WHERE A.x > 10) JOIN B ON A.id = B.id

    Reduces JOIN input size → faster execution."""

    @staticmethod
    def apply(ast: SelectStmt) -> SelectStmt:
        if ast.where is None or ast.from_clause is None:
            return ast
        if not ast.from_clause.joins:
            return ast  # No joins, nothing to push

        pushed_to_base = []
        remaining = []

        conjuncts = PredicatePushdown._split_and(ast.where)
        tables_in_from = {ast.from_clause.table.alias or ast.from_clause.table.name}
        for jc in ast.from_clause.joins:
            if jc.table:
                tables_in_from.add(jc.table.alias or jc.table.name)

        for pred in conjuncts:
            pred_tables = PredicatePushdown._tables_referenced(pred)
            if len(pred_tables) <= 1:
                # Single-table predicate → can push down
                pushed_to_base.append(pred)
            else:
                remaining.append(pred)

        if not pushed_to_base:
            return ast  # Nothing to push

        # Rebuild WHERE with only multi-table predicates
        new_where = PredicatePushdown._combine_and(remaining) if remaining else None
        # Note: actual pushdown into join inputs would require rewriting the plan
        # For now, we just reorder: pushed predicates first (cheaper to evaluate early)
        all_preds = pushed_to_base + remaining
        reordered_where = PredicatePushdown._combine_and(all_preds) if all_preds else None
        return dataclasses.replace(ast, where=reordered_where)

    @staticmethod
    def _split_and(expr: Any) -> list:
        """Split AND expression into conjuncts."""
        if isinstance(expr, BinaryExpr) and expr.op == 'AND':
            return (PredicatePushdown._split_and(expr.left) +
                    PredicatePushdown._split_and(expr.right))
        return [expr]

    @staticmethod
    def _combine_and(exprs: list) -> Any:
        if not exprs: return None
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
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                result |= PredicatePushdown._tables_referenced(child)
        return result


class PredicateReorder:
    """Reorder AND conjuncts by estimated selectivity.
    Most selective predicates first → fewer rows to evaluate for later predicates.

    Heuristic selectivity ranking:
    1. col = constant  (most selective, ~10%)
    2. col IS NULL     (~5%)
    3. col > constant  (~33%)
    4. col LIKE '...'  (~10%)
    5. function(col)   (~50%, expensive to evaluate)
    """

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
        """Lower score = evaluate first (more selective or cheaper)."""
        if isinstance(pred, BinaryExpr):
            if pred.op == '=' and isinstance(pred.right, Literal): return 0.1
            if pred.op == '!=' and isinstance(pred.right, Literal): return 0.9
            if pred.op in ('<', '>', '<=', '>='): return 0.33
        if isinstance(pred, IsNullExpr): return 0.05
        from parser.ast import LikeExpr, InExpr, BetweenExpr
        if isinstance(pred, LikeExpr): return 0.4
        if isinstance(pred, InExpr): return 0.2
        if isinstance(pred, BetweenExpr): return 0.25
        return 0.5


class TopNPushdown:
    """If ORDER BY + LIMIT, use TopN operator instead of full sort.
    O(n log K) vs O(n log n) where K = LIMIT value."""

    @staticmethod
    def should_use_top_n(ast: SelectStmt) -> bool:
        if not ast.order_by or ast.limit is None:
            return False
        if isinstance(ast.limit, Literal) and isinstance(ast.limit.value, int):
            return ast.limit.value > 0
        return False

    @staticmethod
    def get_limit_value(ast: SelectStmt) -> int:
        if isinstance(ast.limit, Literal) and isinstance(ast.limit.value, int):
            return ast.limit.value
        return 0
