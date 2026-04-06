from __future__ import annotations
"""向量化操作模板 — 消除 evaluator 中的逐元素循环重复。
每个模板封装：null 检测 → 计算 → 结果写入。
调用方只需传入计算 lambda。"""
from typing import Any, Callable, Optional
from executor.core.vector import DataVector
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType


def map_unary_numeric(vec: DataVector, n: int,
                      fn: Callable, out_dtype: DataType) -> DataVector:
    """一元数值映射：对每个非 NULL 值应用 fn。
    fn 签名: fn(value) → result_value | None"""
    code = DTYPE_TO_ARRAY_CODE.get(out_dtype)
    is_typed = code is not None
    rd = TypedVector(code) if is_typed else []
    rn = Bitmap(n)
    zero = 0 if is_typed else None
    for i in range(n):
        if vec.is_null(i):
            rn.set_bit(i)
            rd.append(zero) if is_typed else rd.append(None)
        else:
            val = fn(vec.get(i))
            if val is None:
                rn.set_bit(i)
                rd.append(zero) if is_typed else rd.append(None)
            else:
                rd.append(val)
    return DataVector(dtype=out_dtype, data=rd, nulls=rn, _length=n)


def map_binary_numeric(a: DataVector, b: DataVector, n: int,
                       fn: Callable, out_dtype: DataType) -> DataVector:
    """二元数值映射：对每对非 NULL 值应用 fn。
    fn 签名: fn(a_val, b_val) → result_value | None"""
    code = DTYPE_TO_ARRAY_CODE.get(out_dtype)
    is_typed = code is not None
    rd = TypedVector(code) if is_typed else []
    rn = Bitmap(n)
    zero = 0 if is_typed else None
    for i in range(n):
        if a.is_null(i) or b.is_null(i):
            rn.set_bit(i)
            rd.append(zero) if is_typed else rd.append(None)
        else:
            val = fn(a.get(i), b.get(i))
            if val is None:
                rn.set_bit(i)
                rd.append(zero) if is_typed else rd.append(None)
            else:
                rd.append(val)
    return DataVector(dtype=out_dtype, data=rd, nulls=rn, _length=n)


def map_unary_string(vec: DataVector, n: int,
                     fn: Callable) -> DataVector:
    """一元字符串映射：对每个非 NULL 值应用 fn(str) → str。"""
    rd = []
    rn = Bitmap(n)
    for i in range(n):
        if vec.is_null(i):
            rn.set_bit(i)
            rd.append('')
        else:
            rd.append(fn(str(vec.get(i))))
    return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)


def map_binary_string(a: DataVector, b: DataVector, n: int,
                      fn: Callable) -> DataVector:
    """二元字符串映射。"""
    rd = []
    rn = Bitmap(n)
    for i in range(n):
        if a.is_null(i) or b.is_null(i):
            rn.set_bit(i)
            rd.append('')
        else:
            rd.append(fn(str(a.get(i)), str(b.get(i))))
    return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)


def map_ternary_string(a: DataVector, b: DataVector,
                       c: DataVector, n: int,
                       fn: Callable) -> DataVector:
    """三元字符串映射。"""
    rd = []
    rn = Bitmap(n)
    for i in range(n):
        if a.is_null(i) or b.is_null(i) or c.is_null(i):
            rn.set_bit(i)
            rd.append('')
        else:
            rd.append(fn(str(a.get(i)), str(b.get(i)), str(c.get(i))))
    return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)


def map_str_int(sv: DataVector, iv: DataVector, n: int,
                fn: Callable) -> DataVector:
    """字符串 + 整数 → 字符串映射。"""
    rd = []
    rn = Bitmap(n)
    for i in range(n):
        if sv.is_null(i) or iv.is_null(i):
            rn.set_bit(i)
            rd.append('')
        else:
            rd.append(fn(str(sv.get(i)), int(iv.get(i))))
    return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)


def map_binary_bool(a: DataVector, b: DataVector, n: int,
                    fn: Callable) -> DataVector:
    """二元布尔映射：fn(a, b) → bool。"""
    rd = Bitmap(n)
    rn = Bitmap(n)
    for i in range(n):
        if a.is_null(i) or b.is_null(i):
            rn.set_bit(i)
        elif fn(a.get(i), b.get(i)):
            rd.set_bit(i)
    return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)


def values_to_vector(values: list, dtype: DataType,
                     n: int) -> DataVector:
    """Python list → DataVector。统一处理 NULL 标记。"""
    if dtype == DataType.UNKNOWN:
        dtype = DataType.INT
    code = DTYPE_TO_ARRAY_CODE.get(dtype)
    nulls = Bitmap(n)

    if dtype == DataType.BOOLEAN:
        data = Bitmap(n)
        for i in range(n):
            if values[i] is None:
                nulls.set_bit(i)
            elif values[i]:
                data.set_bit(i)
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)

    if dtype in (DataType.VARCHAR, DataType.TEXT):
        data = []
        for i in range(n):
            if values[i] is None:
                nulls.set_bit(i)
                data.append('')
            else:
                data.append(str(values[i]))
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)

    if code is not None:
        data = TypedVector(code)
        for i in range(n):
            if values[i] is None:
                nulls.set_bit(i)
                data.append(0)
            else:
                data.append(values[i])
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)

    # 回退
    data = TypedVector('q')
    for i in range(n):
        if values[i] is None:
            nulls.set_bit(i)
            data.append(0)
        else:
            data.append(int(values[i]))
    return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)
