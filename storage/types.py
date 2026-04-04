from __future__ import annotations
"""Data types, promotion rules, and type utilities."""

import array as _array
from enum import Enum
from typing import Dict, Optional

from utils.errors import ExecutionError, TypeMismatchError


class DataType(Enum):
    INT = 'INT'
    BIGINT = 'BIGINT'
    FLOAT = 'FLOAT'
    DOUBLE = 'DOUBLE'
    BOOLEAN = 'BOOLEAN'
    VARCHAR = 'VARCHAR'
    TEXT = 'TEXT'
    DATE = 'DATE'
    TIMESTAMP = 'TIMESTAMP'
    ARRAY = 'ARRAY'
    UNKNOWN = 'UNKNOWN'


DTYPE_TO_ARRAY_CODE: Dict[DataType, Optional[str]] = {
    DataType.INT: 'i',
    DataType.BIGINT: 'q',
    DataType.FLOAT: 'd',
    DataType.DOUBLE: 'd',
    DataType.DATE: 'i',
    DataType.TIMESTAMP: 'q',
    DataType.BOOLEAN: None,
    DataType.VARCHAR: None,
    DataType.TEXT: None,
    DataType.ARRAY: None,
    DataType.UNKNOWN: None,
}

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
}


def resolve_type_name(name: str, params: Optional[list[int]] = None) -> tuple[DataType, Optional[int]]:
    """Resolve a type name string to ``(DataType, max_length | None)``."""
    upper = name.upper()
    if upper not in TYPE_NAME_TO_DATATYPE:
        raise ExecutionError(f"unknown data type: {name}")
    dt = TYPE_NAME_TO_DATATYPE[upper]
    max_len = params[0] if params else None
    return dt, max_len


# -- numeric rank for promotion ----------------------------------------
_NUMERIC_RANK: Dict[DataType, int] = {
    DataType.BOOLEAN: 0,
    DataType.INT: 1,
    DataType.BIGINT: 2,
    DataType.FLOAT: 3,
    DataType.DOUBLE: 4,
}


def promote(left: DataType, right: DataType) -> DataType:
    """Return the common super-type for *left* and *right*."""
    if left == DataType.UNKNOWN:
        return right if right != DataType.UNKNOWN else DataType.UNKNOWN
    if right == DataType.UNKNOWN:
        return left

    # Same type
    if left == right:
        return left

    # Both numeric
    l_rank = _NUMERIC_RANK.get(left)
    r_rank = _NUMERIC_RANK.get(right)
    if l_rank is not None and r_rank is not None:
        # Special: BIGINT + FLOAT → DOUBLE to avoid precision loss
        pair = frozenset((left, right))
        if pair == frozenset((DataType.BIGINT, DataType.FLOAT)):
            return DataType.DOUBLE
        return left if l_rank >= r_rank else right

    # String types
    if left in (DataType.VARCHAR, DataType.TEXT) and right in (DataType.VARCHAR, DataType.TEXT):
        if DataType.TEXT in (left, right):
            return DataType.TEXT
        return DataType.VARCHAR

    raise TypeMismatchError(
        f"cannot promote {left.name} and {right.name}",
        expected=left.name,
        actual=right.name,
    )


def is_numeric(dt: DataType) -> bool:
    return dt in _NUMERIC_RANK


def is_string(dt: DataType) -> bool:
    return dt in (DataType.VARCHAR, DataType.TEXT)


def is_temporal(dt: DataType) -> bool:
    return dt in (DataType.DATE, DataType.TIMESTAMP)


# -- sanity checks at import time -------------------------------------
assert _array.array('i').itemsize == 4, "int32 must be 4 bytes"
assert _array.array('q').itemsize == 8, "int64 must be 8 bytes"
assert _array.array('d').itemsize == 8, "float64 must be 8 bytes"
