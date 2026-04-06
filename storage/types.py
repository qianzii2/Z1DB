from __future__ import annotations
"""数据类型定义、类型提升规则、类型工具函数。"""

import array as _array
from enum import Enum
from typing import Dict, Optional

from utils.errors import ExecutionError, TypeMismatchError


class DataType(Enum):
    """Z1DB 支持的数据类型。"""
    INT = 'INT'             # 32 位有符号整数
    BIGINT = 'BIGINT'       # 64 位有符号整数
    FLOAT = 'FLOAT'         # 64 位浮点（内部等同 DOUBLE）
    DOUBLE = 'DOUBLE'       # 64 位浮点
    BOOLEAN = 'BOOLEAN'     # 布尔值
    VARCHAR = 'VARCHAR'     # 变长字符串（可指定最大长度）
    TEXT = 'TEXT'            # 无限长字符串
    DATE = 'DATE'           # 日期（epoch 天数）
    TIMESTAMP = 'TIMESTAMP' # 时间戳（epoch 微秒数）
    ARRAY = 'ARRAY'         # 数组（以字符串形式存储）
    UNKNOWN = 'UNKNOWN'     # 未知类型（NULL 字面量等）


# 数据类型 → array.array 类型码映射
# None 表示该类型不使用 array.array 存储
DTYPE_TO_ARRAY_CODE: Dict[DataType, Optional[str]] = {
    DataType.INT: 'i',       # int32
    DataType.BIGINT: 'q',   # int64
    DataType.FLOAT: 'd',    # float64
    DataType.DOUBLE: 'd',   # float64
    DataType.DATE: 'i',     # epoch 天数 → int32
    DataType.TIMESTAMP: 'q', # epoch 微秒 → int64
    DataType.BOOLEAN: None,  # 用 Bitmap 存储
    DataType.VARCHAR: None,  # 用 list 或 InlineStringStore
    DataType.TEXT: None,
    DataType.ARRAY: None,
    DataType.UNKNOWN: None,
}

# 类型名称 → DataType 映射（SQL 解析用）
TYPE_NAME_TO_DATATYPE: Dict[str, DataType] = {
    'INT': DataType.INT,
    'INTEGER': DataType.INT,
    'BIGINT': DataType.BIGINT,
    'FLOAT': DataType.FLOAT,
    'REAL': DataType.FLOAT,
    'DOUBLE': DataType.DOUBLE,
    'BOOLEAN': DataType.BOOLEAN,
    'BOOL': DataType.BOOLEAN,
    'VARCHAR': DataType.VARCHAR,
    'TEXT': DataType.TEXT,
    'DATE': DataType.DATE,
    'TIMESTAMP': DataType.TIMESTAMP,
    # DECIMAL 暂映射到 DOUBLE（未来实现精确十进制）
    'DECIMAL': DataType.DOUBLE,
    'NUMERIC': DataType.DOUBLE,
}


def resolve_type_name(name: str,
                      params: Optional[list[int]] = None
                      ) -> tuple[DataType, Optional[int]]:
    """解析 SQL 类型名称为 (DataType, 最大长度)。
    示例: 'VARCHAR', [100] → (DataType.VARCHAR, 100)"""
    upper = name.upper()
    if upper not in TYPE_NAME_TO_DATATYPE:
        raise ExecutionError(f"unknown data type: {name}")
    dt = TYPE_NAME_TO_DATATYPE[upper]
    max_len = params[0] if params else None
    return dt, max_len


# ═══ 数值类型提升 ═══
# 等级越高精度越大，混合运算时提升到较高等级
_NUMERIC_RANK: Dict[DataType, int] = {
    DataType.BOOLEAN: 0,
    DataType.INT: 1,
    DataType.BIGINT: 2,
    DataType.FLOAT: 3,
    DataType.DOUBLE: 4,
}


def promote(left: DataType, right: DataType) -> DataType:
    """返回两个类型的公共超类型（类型提升）。
    规则：
      - UNKNOWN 让步于任何类型
      - 两个数值类型取较高等级
      - BIGINT + FLOAT → DOUBLE（避免精度丢失）
      - 两个字符串类型取 TEXT（更宽泛）
      - 不兼容类型抛出 TypeMismatchError
    """
    if left == DataType.UNKNOWN:
        return right if right != DataType.UNKNOWN else DataType.UNKNOWN
    if right == DataType.UNKNOWN:
        return left
    if left == right:
        return left

    # 数值提升
    l_rank = _NUMERIC_RANK.get(left)
    r_rank = _NUMERIC_RANK.get(right)
    if l_rank is not None and r_rank is not None:
        # 特殊规则：BIGINT + FLOAT → DOUBLE
        pair = frozenset((left, right))
        if pair == frozenset((DataType.BIGINT, DataType.FLOAT)):
            return DataType.DOUBLE
        return left if l_rank >= r_rank else right

    # 字符串提升
    if is_string(left) and is_string(right):
        return DataType.TEXT if DataType.TEXT in (left, right) else DataType.VARCHAR

    raise TypeMismatchError(
        f"cannot promote {left.name} and {right.name}",
        expected=left.name,
        actual=right.name,
    )


# ═══ 类型判断工具 ═══

def is_numeric(dt: DataType) -> bool:
    """是否为数值类型（含 BOOLEAN，因为 BOOLEAN 可参与算术运算）。"""
    return dt in _NUMERIC_RANK


def is_string(dt: DataType) -> bool:
    """是否为字符串类型。"""
    return dt in (DataType.VARCHAR, DataType.TEXT)


def is_temporal(dt: DataType) -> bool:
    """是否为时间类型。"""
    return dt in (DataType.DATE, DataType.TIMESTAMP)


def is_comparable(dt1: DataType, dt2: DataType) -> bool:
    """两个类型是否可以互相比较。"""
    if dt1 == dt2:
        return True
    if is_numeric(dt1) and is_numeric(dt2):
        return True
    if is_string(dt1) and is_string(dt2):
        return True
    if is_temporal(dt1) and is_temporal(dt2):
        return True
    return False


# ═══ 导入时校验平台字长 ═══
assert _array.array('i').itemsize == 4, "int32 必须为 4 字节"
assert _array.array('q').itemsize == 8, "int64 必须为 8 字节"
assert _array.array('d').itemsize == 8, "float64 必须为 8 字节"
