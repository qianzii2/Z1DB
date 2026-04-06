from __future__ import annotations
"""Z1DB 统一异常体系。
所有用户可见错误继承自 Z1Error，保证上层统一捕获。"""


class Z1Error(Exception):
    """Z1DB 根异常。所有自定义异常的基类。"""

    def __init__(self, message: str, error_code: int = 0):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


# ═══ 解析阶段异常 ═══

class ParseError(Z1Error):
    """词法/语法分析错误。"""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(message)


class SemanticError(Z1Error):
    """语义分析错误（名称解析、类型检查、GROUP BY 验证等）。"""
    pass


# ═══ 类型系统异常 ═══

class TypeMismatchError(Z1Error):
    """类型不匹配（如 INT + VARCHAR）。"""

    def __init__(self, message: str, expected: str = '', actual: str = ''):
        self.expected = expected
        self.actual = actual
        super().__init__(message)


# ═══ 执行阶段异常 ═══

class ExecutionError(Z1Error):
    """通用执行错误。"""
    pass


class DivisionByZeroError(Z1Error):
    """除以零。"""

    def __init__(self, message: str = 'division by zero'):
        super().__init__(message)


class NumericOverflowError(Z1Error):
    """整数溢出（INT 超出 [-2^31, 2^31-1]）。"""

    def __init__(self, message: str = 'numeric overflow'):
        super().__init__(message)


class MemoryLimitError(Z1Error):
    """内存预算超限，需要溢写到磁盘。"""
    pass


class RecursionLimitError(Z1Error):
    """递归 CTE 超过最大迭代次数或行数限制。"""
    pass


# ═══ 目录/元数据异常 ═══

class TableNotFoundError(Z1Error):
    """表不存在。"""

    def __init__(self, table_name: str):
        self.table_name = table_name
        super().__init__(f"table '{table_name}' not found")


class ColumnNotFoundError(Z1Error):
    """列不存在。"""

    def __init__(self, column_name: str):
        self.column_name = column_name
        super().__init__(f"column '{column_name}' not found")


class DuplicateError(Z1Error):
    """重复定义（表名、列名、索引名等）。"""
    pass


class ConstraintError(Z1Error):
    """约束违反（NOT NULL、CHECK、UNIQUE 等）。"""
    pass


# ═══ 基础设施异常 ═══

class SessionError(Z1Error):
    """会话/连接错误。"""
    pass


class PreparedStmtError(Z1Error):
    """参数化查询错误（预留）。"""
    pass


class VacuumError(Z1Error):
    """VACUUM 操作错误。"""
    pass
