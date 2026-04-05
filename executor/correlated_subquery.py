from __future__ import annotations
"""Correlated subquery — integrated into simple_planner resolve path.
Strategy: detect outer column refs → per-row substitute + execute with cache."""
import dataclasses
from typing import Any, Dict, List, Optional, Set
from metal.hash import murmur3_64
from parser.ast import (ColumnRef, ExistsExpr, InExpr, Literal, SelectStmt,
                         SubqueryExpr, AliasExpr)
from storage.types import DataType
from utils.errors import ExecutionError


class CorrelatedResolver:
    """Resolves correlated subqueries by substituting outer values."""

    def __init__(self, planner: Any, catalog: Any) -> None:
        self._planner = planner
        self._catalog = catalog
        self._cache: Dict[int, Any] = {}

    def resolve(self, node: Any, outer_row: Optional[Dict[str, Any]] = None,
                outer_cols: Optional[Set[str]] = None) -> Any:
        """Resolve subquery nodes. If outer_row provided, substitute correlated refs."""
        if node is None:
            return None

        if isinstance(node, SubqueryExpr):
            query = node.query
            if outer_row and self._is_correlated(query, outer_cols or set()):
                bound = self._bind(query, outer_row, outer_cols or set())
                ck = self._cache_key(bound, outer_row)
                if ck in self._cache:
                    return self._cache[ck]
                result = self._exec(bound)
                val = result.rows[0][0] if result.rows and result.columns else None
                dt = result.column_types[0] if result.column_types else DataType.INT
                lit = Literal(value=val, inferred_type=dt)
                self._cache[ck] = lit
                return lit
            else:
                result = self._exec(query)
                if result.rows and result.columns:
                    return Literal(value=result.rows[0][0],
                                   inferred_type=result.column_types[0] if result.column_types else DataType.INT)
                return Literal(value=None, inferred_type=DataType.UNKNOWN)

        if isinstance(node, ExistsExpr):
            query = node.query
            if outer_row and self._is_correlated(query, outer_cols or set()):
                bound = self._bind(query, outer_row, outer_cols or set())
                result = self._exec(bound)
                exists = len(result.rows) > 0
                val = not exists if node.negated else exists
                return Literal(value=val, inferred_type=DataType.BOOLEAN)
            else:
                result = self._exec(query)
                exists = len(result.rows) > 0
                val = not exists if node.negated else exists
                return Literal(value=val, inferred_type=DataType.BOOLEAN)

        if isinstance(node, InExpr):
            has_sq = any(isinstance(v, SubqueryExpr) for v in node.values)
            if has_sq:
                new_vals: list = []
                for v in node.values:
                    if isinstance(v, SubqueryExpr):
                        sq = v.query
                        if outer_row and self._is_correlated(sq, outer_cols or set()):
                            bound = self._bind(sq, outer_row, outer_cols or set())
                            result = self._exec(bound)
                        else:
                            result = self._exec(sq)
                        for row in result.rows:
                            dt = result.column_types[0] if result.column_types else DataType.INT
                            new_vals.append(Literal(value=row[0], inferred_type=dt))
                    else:
                        new_vals.append(v)
                return dataclasses.replace(node, values=new_vals,
                                           expr=self.resolve(node.expr, outer_row, outer_cols))

        # Recurse into dataclass children
        if isinstance(node, (list, tuple)):
            return type(node)(self.resolve(item, outer_row, outer_cols) for item in node)
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        changes: dict = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, (SubqueryExpr, ExistsExpr, InExpr)):
                changes[f.name] = self.resolve(child, outer_row, outer_cols)
            elif isinstance(child, list):
                changes[f.name] = [self.resolve(i, outer_row, outer_cols) for i in child]
            elif isinstance(child, tuple):
                changes[f.name] = tuple(self.resolve(i, outer_row, outer_cols) for i in child)
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self.resolve(child, outer_row, outer_cols)
        return dataclasses.replace(node, **changes) if changes else node

    def _is_correlated(self, query: Any, outer_cols: Set[str]) -> bool:
        """Does query reference any column from outer scope?"""
        refs = self._collect_refs(query)
        for ref in refs:
            if ref.table:
                q = f"{ref.table}.{ref.column}"
                if q in outer_cols or ref.table in outer_cols:
                    if not self._is_local(ref, query):
                        return True
            elif ref.column in outer_cols:
                if not self._is_local(ref, query):
                    return True
        return False

    def _bind(self, query: Any, vals: Dict[str, Any], outer_cols: Set[str]) -> Any:
        """Replace outer ColumnRefs with Literals."""
        if isinstance(query, ColumnRef):
            key = f"{query.table}.{query.column}" if query.table else query.column
            if key in vals:
                return self._to_lit(vals[key])
            if query.column in vals and not self._is_local(query, None):
                return self._to_lit(vals[query.column])
            return query
        if isinstance(query, Literal):
            return query
        if not dataclasses.is_dataclass(query) or isinstance(query, type):
            return query
        changes: dict = {}
        for f in dataclasses.fields(query):
            child = getattr(query, f.name)
            if isinstance(child, ColumnRef):
                changes[f.name] = self._bind(child, vals, outer_cols)
            elif isinstance(child, list):
                nl = []
                for item in child:
                    if isinstance(item, tuple):
                        nl.append(tuple(self._bind(x, vals, outer_cols) for x in item))
                    else:
                        nl.append(self._bind(item, vals, outer_cols))
                changes[f.name] = nl
            elif isinstance(child, tuple):
                changes[f.name] = tuple(self._bind(x, vals, outer_cols) for x in child)
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._bind(child, vals, outer_cols)
        return dataclasses.replace(query, **changes) if changes else query

    def _exec(self, query: Any) -> Any:
        from parser.resolver import Resolver
        from parser.validator import Validator
        from executor.core.result import ExecutionResult
        try:
            r = Resolver().resolve(query, self._catalog)
            v = Validator().validate(r, self._catalog)
            return self._planner.execute(v, self._catalog)
        except Exception:
            return ExecutionResult()

    def _collect_refs(self, node: Any) -> List[ColumnRef]:
        if isinstance(node, ColumnRef): return [node]
        if node is None or isinstance(node, (Literal, str, int, float, bool)): return []
        if isinstance(node, (list, tuple)):
            r: list = []
            for i in node: r.extend(self._collect_refs(i))
            return r
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            r = []
            for f in dataclasses.fields(node):
                r.extend(self._collect_refs(getattr(node, f.name)))
            return r
        return []

    def _is_local(self, ref: ColumnRef, query: Any) -> bool:
        if not isinstance(query, SelectStmt) or query is None: return False
        if query.from_clause is None: return False
        local = {query.from_clause.table.alias or query.from_clause.table.name}
        for jc in query.from_clause.joins:
            if jc.table: local.add(jc.table.alias or jc.table.name)
        if ref.table and ref.table in local: return True
        if not ref.table:
            for t in local:
                if self._catalog.table_exists(t):
                    if ref.column in self._catalog.get_table_columns(t): return True
        return False

    @staticmethod
    def _to_lit(val: Any) -> Literal:
        if val is None: return Literal(value=None, inferred_type=DataType.UNKNOWN)
        if isinstance(val, bool): return Literal(value=val, inferred_type=DataType.BOOLEAN)
        if isinstance(val, int): return Literal(value=val, inferred_type=DataType.INT)
        if isinstance(val, float): return Literal(value=val, inferred_type=DataType.DOUBLE)
        return Literal(value=str(val), inferred_type=DataType.VARCHAR)

    @staticmethod
    def _cache_key(query: Any, vals: Dict) -> int:
        return murmur3_64(f"{repr(query)[:100]}|{sorted(vals.items())}".encode())

    def clear_cache(self) -> None:
        self._cache.clear()
