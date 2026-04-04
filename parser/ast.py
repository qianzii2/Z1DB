from __future__ import annotations
"""AST node definitions."""
from dataclasses import dataclass, field
from typing import Any, List, Optional
from storage.types import DataType

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
class SetOperationStmt:
    op: str = 'UNION'
    all: bool = False
    left: Any = None
    right: Any = None

@dataclass
class InsertStmt:
    table: str = ''
    columns: Optional[List[str]] = None
    values: list = field(default_factory=list)

@dataclass
class UpdateStmt:
    table: str = ''
    assignments: list = field(default_factory=list)
    where: Optional[Any] = None

@dataclass
class DeleteStmt:
    table: str = ''
    where: Optional[Any] = None

@dataclass
class CreateTableStmt:
    table: str = ''
    columns: List[ColumnDef] = field(default_factory=list)
    if_not_exists: bool = False

@dataclass
class DropTableStmt:
    table: str = ''
    if_exists: bool = False

@dataclass
class ExplainStmt:
    statement: Any = None

@dataclass
class AlterTableStmt:
    table: str = ''
    action: str = ''       # ADD_COLUMN, DROP_COLUMN, RENAME_COLUMN
    column_def: Optional[ColumnDef] = None
    column_name: str = ''
    new_name: str = ''

@dataclass
class FromClause:
    table: TableRef = None  # type: ignore
    joins: list = field(default_factory=list)

@dataclass
class TableRef:
    name: str = ''
    alias: Optional[str] = None
    subquery: Optional[Any] = None

@dataclass
class JoinClause:
    join_type: str = 'INNER'
    table: Optional[TableRef] = None
    on: Optional[Any] = None

@dataclass
class SortKey:
    expr: Any = None
    direction: str = 'ASC'
    nulls: Optional[str] = None

@dataclass
class ColumnDef:
    name: str = ''
    type_name: Optional[TypeName] = None
    nullable: bool = True
    primary_key: bool = False

@dataclass
class TypeName:
    name: str = ''
    params: List[int] = field(default_factory=list)

@dataclass
class GroupByClause:
    keys: list = field(default_factory=list)

@dataclass
class Assignment:
    column: str = ''
    value: Any = None

@dataclass
class Literal:
    value: Any = None
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

@dataclass
class WindowCall:
    func: Any = None
    partition_by: list = field(default_factory=list)
    order_by: list = field(default_factory=list)
    frame: Optional[WindowFrame] = None

@dataclass
class WindowFrame:
    mode: str = 'ROWS'
    start: Optional[FrameBound] = None
    end: Optional[FrameBound] = None

@dataclass
class FrameBound:
    type: str = 'CURRENT_ROW'
    offset: Optional[int] = None

@dataclass
class CaseExpr:
    operand: Optional[Any] = None
    when_clauses: list = field(default_factory=list)
    else_expr: Optional[Any] = None

@dataclass
class CastExpr:
    expr: Any = None
    type_name: Optional[TypeName] = None

@dataclass
class InExpr:
    expr: Any = None
    values: list = field(default_factory=list)
    negated: bool = False

@dataclass
class BetweenExpr:
    expr: Any = None
    low: Any = None
    high: Any = None
    negated: bool = False

@dataclass
class LikeExpr:
    expr: Any = None
    pattern: Any = None
    negated: bool = False

@dataclass
class SubqueryExpr:
    query: Any = None

@dataclass
class ExistsExpr:
    query: Any = None
    negated: bool = False
