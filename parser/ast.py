from __future__ import annotations
"""AST 节点定义。每个 SQL 语句类型和表达式类型对应一个 dataclass。"""
from dataclasses import dataclass, field
from typing import Any, List, Optional
from storage.types import DataType


# ═══ 语句节点 ═══

@dataclass
class SelectStmt:
    """SELECT 语句。"""
    distinct: bool = False
    select_list: list = field(default_factory=list)
    from_clause: Optional['FromClause'] = None
    where: Optional[Any] = None
    group_by: Optional['GroupByClause'] = None
    having: Optional[Any] = None
    window_defs: list = field(default_factory=list)
    order_by: list = field(default_factory=list)
    limit: Optional[Any] = None
    offset: Optional[Any] = None
    ctes: list = field(default_factory=list)
    sample: Optional[Any] = None


@dataclass
class SetOperationStmt:
    """UNION / INTERSECT / EXCEPT 语句。"""
    op: str = 'UNION'
    all: bool = False
    left: Any = None
    right: Any = None


@dataclass
class InsertStmt:
    """INSERT 语句。"""
    table: str = ''
    columns: Optional[List[str]] = None
    values: list = field(default_factory=list)
    query: Optional[Any] = None  # INSERT ... SELECT


@dataclass
class UpdateStmt:
    """UPDATE 语句。"""
    table: str = ''
    assignments: list = field(default_factory=list)
    where: Optional[Any] = None


@dataclass
class DeleteStmt:
    """DELETE 语句。"""
    table: str = ''
    where: Optional[Any] = None


@dataclass
class CreateTableStmt:
    """CREATE TABLE 语句。"""
    table: str = ''
    columns: List['ColumnDef'] = field(default_factory=list)
    if_not_exists: bool = False


@dataclass
class DropTableStmt:
    """DROP TABLE 语句。"""
    table: str = ''
    if_exists: bool = False


@dataclass
class CreateIndexStmt:
    """CREATE INDEX 语句。"""
    index_name: str = ''
    table: str = ''
    columns: List[str] = field(default_factory=list)
    unique: bool = False
    if_not_exists: bool = False


@dataclass
class DropIndexStmt:
    """DROP INDEX 语句。"""
    index_name: str = ''
    if_exists: bool = False


@dataclass
class ExplainStmt:
    """EXPLAIN 语句。"""
    statement: Any = None


@dataclass
class AlterTableStmt:
    """ALTER TABLE 语句。"""
    table: str = ''
    action: str = ''  # ADD_COLUMN / DROP_COLUMN / RENAME_COLUMN
    column_def: Optional['ColumnDef'] = None
    column_name: str = ''
    new_name: str = ''


@dataclass
class CopyStmt:
    """COPY table FROM/TO 'filepath' [WITH (...)]"""
    table: str = ''
    file_path: str = ''
    direction: str = 'FROM'  # FROM / TO
    has_header: bool = True
    delimiter: str = ','


@dataclass
class VacuumStmt:
    """VACUUM [table] — 物理清除已删除行。[M05]"""
    table: Optional[str] = None  # None = 全库


# ═══ 预留语句节点 [M03] ═══

@dataclass
class PrepareStmt:
    """PREPARE 语句（预留，未来版本实现）。"""
    name: str = ''
    params: List[str] = field(default_factory=list)
    statement: Any = None


@dataclass
class ExecuteStmt:
    """EXECUTE 语句（预留，未来版本实现）。"""
    name: str = ''
    args: list = field(default_factory=list)


# ═══ 子句节点 ═══

@dataclass
class FromClause:
    """FROM 子句。"""
    table: 'TableRef' = None  # type: ignore
    joins: list = field(default_factory=list)


@dataclass
class TableRef:
    """表引用。可以是表名、子查询或表函数。"""
    name: str = ''
    alias: Optional[str] = None
    subquery: Optional[Any] = None
    func_args: Optional[list] = None  # 表函数参数


@dataclass
class JoinClause:
    """JOIN 子句。"""
    join_type: str = 'INNER'  # INNER/LEFT/RIGHT/FULL/CROSS/SEMI/ANTI
    table: Optional['TableRef'] = None
    on: Optional[Any] = None
    using: Optional[List[str]] = None
    natural: bool = False


@dataclass
class SortKey:
    """排序键。"""
    expr: Any = None
    direction: str = 'ASC'
    nulls: Optional[str] = None  # NULLS_FIRST / NULLS_LAST


@dataclass
class ColumnDef:
    """列定义（CREATE TABLE / ALTER TABLE 用）。"""
    name: str = ''
    type_name: Optional['TypeName'] = None
    nullable: bool = True
    primary_key: bool = False
    check: Optional[Any] = None  # [M08] CHECK 约束（预留）


@dataclass
class TypeName:
    """类型名称 + 参数（如 VARCHAR(100)）。"""
    name: str = ''
    params: List[int] = field(default_factory=list)


@dataclass
class GroupByClause:
    """GROUP BY 子句。"""
    keys: list = field(default_factory=list)


@dataclass
class Assignment:
    """UPDATE SET 赋值。"""
    column: str = ''
    value: Any = None


# ═══ 表达式节点 ═══

@dataclass
class Literal:
    """字面量。"""
    value: Any = None
    inferred_type: DataType = DataType.UNKNOWN


@dataclass
class ColumnRef:
    """列引用。"""
    table: Optional[str] = None
    column: str = ''


@dataclass
class StarExpr:
    """星号表达式。table 非 None 时表示 t.*。"""
    table: Optional[str] = None


@dataclass
class BinaryExpr:
    """二元表达式。"""
    op: str = ''
    left: Any = None
    right: Any = None


@dataclass
class UnaryExpr:
    """一元表达式（+、-、NOT）。"""
    op: str = ''
    operand: Any = None


@dataclass
class IsNullExpr:
    """IS NULL / IS NOT NULL。"""
    expr: Any = None
    negated: bool = False


@dataclass
class AliasExpr:
    """别名表达式（expr AS alias）。"""
    expr: Any = None
    alias: str = ''


@dataclass
class AggregateCall:
    """聚合函数调用。"""
    name: str = ''
    args: list = field(default_factory=list)
    distinct: bool = False
    filter_clause: Optional[Any] = None


@dataclass
class FunctionCall:
    """标量函数调用。"""
    name: str = ''
    args: list = field(default_factory=list)


@dataclass
class WindowCall:
    """窗口函数调用。"""
    func: Any = None
    partition_by: list = field(default_factory=list)
    order_by: list = field(default_factory=list)
    frame: Optional['WindowFrame'] = None


@dataclass
class WindowFrame:
    """窗口帧定义。"""
    mode: str = 'ROWS'  # ROWS / RANGE
    start: Optional['FrameBound'] = None
    end: Optional['FrameBound'] = None


@dataclass
class FrameBound:
    """帧边界。"""
    type: str = 'CURRENT_ROW'
    # UNBOUNDED_PRECEDING / UNBOUNDED_FOLLOWING /
    # CURRENT_ROW / N_PRECEDING / N_FOLLOWING
    offset: Optional[int] = None


@dataclass
class CaseExpr:
    """CASE 表达式。"""
    operand: Optional[Any] = None  # 简单 CASE 的比较对象
    when_clauses: list = field(default_factory=list)  # [(cond, result), ...]
    else_expr: Optional[Any] = None


@dataclass
class CastExpr:
    """CAST(expr AS type) 表达式。"""
    expr: Any = None
    type_name: Optional['TypeName'] = None


@dataclass
class InExpr:
    """IN / NOT IN 表达式。"""
    expr: Any = None
    values: list = field(default_factory=list)
    negated: bool = False


@dataclass
class BetweenExpr:
    """BETWEEN / NOT BETWEEN 表达式。"""
    expr: Any = None
    low: Any = None
    high: Any = None
    negated: bool = False


@dataclass
class LikeExpr:
    """LIKE / NOT LIKE 表达式。"""
    expr: Any = None
    pattern: Any = None
    negated: bool = False


@dataclass
class SubqueryExpr:
    """标量子查询。"""
    query: Any = None


@dataclass
class ExistsExpr:
    """EXISTS / NOT EXISTS。"""
    query: Any = None
    negated: bool = False
