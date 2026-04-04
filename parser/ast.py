from __future__ import annotations
"""Abstract Syntax Tree node definitions."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from storage.types import DataType


# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SelectStmt:
    distinct: bool = False
    select_list: list = field(default_factory=list)
    from_clause: Optional[FromClause] = None
    where: Optional[Any] = None
    group_by: Optional[GroupByClause] = None
    having: Optional[Any] = None
    window_defs: list = field(default_factory=list)
    order_by: list = field(default_factory=list)
    limit: Optional[Any] = None
    offset: Optional[Any] = None
    ctes: list = field(default_factory=list)
    sample: Optional[Any] = None


@dataclass
class InsertStmt:
    table: str
    columns: Optional[List[str]] = None
    values: list = field(default_factory=list)


@dataclass
class CreateTableStmt:
    table: str
    columns: List[ColumnDef] = field(default_factory=list)
    if_not_exists: bool = False


@dataclass
class DropTableStmt:
    table: str
    if_exists: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Clauses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FromClause:
    table: TableRef
    joins: list = field(default_factory=list)


@dataclass
class TableRef:
    name: str
    alias: Optional[str] = None


@dataclass
class SortKey:
    expr: Any
    direction: str = 'ASC'
    nulls: Optional[str] = None


@dataclass
class ColumnDef:
    name: str
    type_name: TypeName
    nullable: bool = True
    primary_key: bool = False


@dataclass
class TypeName:
    name: str
    params: List[int] = field(default_factory=list)


@dataclass
class GroupByClause:
    keys: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Literal:
    value: Any
    inferred_type: DataType = DataType.UNKNOWN


@dataclass
class ColumnRef:
    table: Optional[str] = None
    column: str = ''


@dataclass
class StarExpr:
    table: Optional[str] = None


@dataclass
class BinaryExpr:
    op: str = ''
    left: Any = None
    right: Any = None


@dataclass
class UnaryExpr:
    op: str = ''
    operand: Any = None


@dataclass
class IsNullExpr:
    expr: Any = None
    negated: bool = False


@dataclass
class AliasExpr:
    expr: Any = None
    alias: str = ''


@dataclass
class AggregateCall:
    name: str = ''
    args: list = field(default_factory=list)
    distinct: bool = False
    filter_clause: Optional[Any] = None


@dataclass
class FunctionCall:
    name: str = ''
    args: list = field(default_factory=list)
