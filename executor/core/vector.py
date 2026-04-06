from __future__ import annotations
"""DataVector — 列向量，NaN-Boxing 数值主存储。
INT/FLOAT/DOUBLE/DATE：NaN-Boxing（_packed array）
BIGINT/TIMESTAMP：直接 TypedVector('q')（避免 2^53 精度限制）
VARCHAR/TEXT：list 或 InlineStringStore
BOOLEAN：Bitmap

[P02] 新增 get_batch 批量读取。
[P07] get() 中 INT 列内联 TAG 判断避免函数调用。"""
import array as _array
from typing import Any, List, Optional

from metal.bitmap import Bitmap, BitmapLike
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType
from utils.errors import ExecutionError

try:
    from executor.core.pool import get_default_pool
    _HAS_POOL = True
except ImportError:
    _HAS_POOL = False

try:
    from metal.bitmagic import (
        nan_pack_float, nan_unpack_float,
        nan_pack_int, nan_pack_bool, nan_pack_null,
        nan_is_null, nan_unpack, NULL_TAG,
        INT_TAG, TAG_MASK, VALUE_MASK,
    )
    _HAS_NANBOX = True
except ImportError:
    _HAS_NANBOX = False

try:
    from metal.inline_string import InlineStringStore
    _HAS_INLINE = True
except ImportError:
    _HAS_INLINE = False
    InlineStringStore = None  # type: ignore

# NaN-Boxing 仅用于这些类型（INT 32位安全，FLOAT/DOUBLE 原生 float64）
_NANBOX_TYPES = frozenset({
    DataType.INT, DataType.FLOAT, DataType.DOUBLE, DataType.DATE,
})

# 这些类型直接用 TypedVector('q')，不走 NaN-Boxing
_DIRECT_INT64_TYPES = frozenset({
    DataType.BIGINT, DataType.TIMESTAMP,
})


class DataVector:
    """列向量。每列数据的核心容器。
    _packed: NaN-Boxing 编码的 array.array('Q')（INT/FLOAT/DOUBLE/DATE）
    data: 解码后的 TypedVector / list / Bitmap
    nulls: NULL 位图"""

    __slots__ = ('dtype', 'data', 'nulls', '_length',
                 'dict_encoded', '_packed')

    def __init__(self, dtype: DataType, data: Any,
                 nulls: Bitmap, _length: int,
                 dict_encoded: Any = None,
                 _packed: Any = None) -> None:
        self.dtype = dtype
        self.data = data
        self.nulls = nulls
        self._length = _length
        self.dict_encoded = dict_encoded
        self._packed = _packed

    def __len__(self) -> int:
        return self._length

    def get(self, i: int) -> Any:
        """读取第 i 行的值。NULL 返回 None。
        [P07] INT 列内联 TAG 判断，避免 nan_unpack 函数调用。"""
        # NaN-Boxing 路径
        if self._packed is not None:
            bits = self._packed[i]
            if bits == NULL_TAG:
                return None
            # [P07] INT/DATE 快速路径：直接检查 INT_TAG
            if _HAS_NANBOX and self.dtype in (DataType.INT, DataType.DATE):
                if (bits & TAG_MASK) == INT_TAG:
                    v = bits & VALUE_MASK
                    if v & 0x80000000:
                        v -= 0x100000000
                    return v
            # 通用解包
            _, val = nan_unpack(bits)
            if val is not None and self.dtype in (
                    DataType.INT, DataType.DATE):
                return int(val) if isinstance(val, float) else val
            if val is not None and self.dtype == DataType.BIGINT:
                return int(val) if isinstance(val, float) else val
            return val
        # 直接路径
        if self.nulls.get_bit(i):
            return None
        if self.dtype == DataType.BOOLEAN:
            return self.data.get_bit(i)
        if isinstance(self.data, TypedVector):
            return self.data[i]
        if isinstance(self.data, list):
            return self.data[i]
        return None

    def get_batch(self, start: int, count: int) -> list:
        """[P02] 批量读取 [start, start+count)。比逐个 get 快 2-3x。"""
        end = min(start + count, self._length)
        if self._packed is not None:
            result = []
            for i in range(start, end):
                bits = self._packed[i]
                if bits == NULL_TAG:
                    result.append(None)
                    continue
                if _HAS_NANBOX and self.dtype in (
                        DataType.INT, DataType.DATE):
                    if (bits & TAG_MASK) == INT_TAG:
                        v = bits & VALUE_MASK
                        if v & 0x80000000:
                            v -= 0x100000000
                        result.append(v)
                        continue
                _, val = nan_unpack(bits)
                if val is not None and self.dtype in (
                        DataType.INT, DataType.DATE, DataType.BIGINT):
                    result.append(
                        int(val) if isinstance(val, float) else val)
                else:
                    result.append(val)
            return result
        # 直接路径批量
        result = []
        for i in range(start, end):
            if self.nulls.get_bit(i):
                result.append(None)
            elif self.dtype == DataType.BOOLEAN:
                result.append(self.data.get_bit(i))
            elif isinstance(self.data, TypedVector):
                result.append(self.data[i])
            elif isinstance(self.data, list):
                result.append(self.data[i])
            else:
                result.append(None)
        return result

    def is_null(self, i: int) -> bool:
        if self._packed is not None:
            return self._packed[i] == NULL_TAG
        return self.nulls.get_bit(i)

    def get_raw(self, i: int) -> int:
        if self._packed is not None:
            return self._packed[i]
        return 0

    @property
    def packed_array(self) -> Optional[Any]:
        return self._packed

    # ═══ 过滤 ═══

    def filter_by_bitmap(self, mask: BitmapLike) -> 'DataVector':
        if isinstance(self.data, list) and hasattr(mask, 'gather_with_nulls'):
            new_data, new_nulls = mask.gather_with_nulls(
                self.data, self.nulls)
            n = len(new_data)
            return DataVector(
                dtype=self.dtype, data=new_data,
                nulls=new_nulls, _length=n)
        return self.filter_by_indices(mask.to_indices())

    def filter_by_indices(self,
                          indices: List[int]) -> 'DataVector':
        n = len(indices)
        # NaN-Boxing 快速路径
        if self._packed is not None:
            new_packed = _array.array(
                'Q', [self._packed[i] for i in indices])
            new_nulls = Bitmap(n)
            for j in range(n):
                if new_packed[j] == NULL_TAG:
                    new_nulls.set_bit(j)
            new_data = _unpack_to_typed(
                self.dtype, new_packed, n)
            return DataVector(
                dtype=self.dtype, data=new_data,
                nulls=new_nulls, _length=n,
                _packed=new_packed)
        # 标准路径
        new_nulls = Bitmap(n)
        if isinstance(self.data, TypedVector):
            new_data: Any = TypedVector(
                self.data.dtype_code,
                [self.data[i] for i in indices])
        elif isinstance(self.data, list):
            new_data = [self.data[i] for i in indices]
        elif isinstance(self.data, Bitmap):
            new_data = Bitmap(n)
            for j, i in enumerate(indices):
                if self.data.get_bit(i):
                    new_data.set_bit(j)
        else:
            raise ExecutionError(
                f"错误的数据类型: {type(self.data)}")
        for j, i in enumerate(indices):
            if self.nulls.get_bit(i):
                new_nulls.set_bit(j)
        return DataVector(
            dtype=self.dtype, data=new_data,
            nulls=new_nulls, _length=n)

    def to_python_list(self) -> list:
        return [self.get(i) for i in range(self._length)]

    # ═══ 工厂方法 ═══

    @staticmethod
    def from_column_chunk(chunk: Any) -> 'DataVector':
        """从 ColumnChunk 构建 DataVector。"""
        from storage.column_chunk import ColumnChunk
        n = chunk.row_count
        if _HAS_INLINE and isinstance(chunk.data, InlineStringStore):
            data = [chunk.data.get(i) for i in range(n)]
        elif isinstance(chunk.data, TypedVector):
            data = chunk.data.copy()
        elif isinstance(chunk.data, list):
            data = list(chunk.data[:n])
        elif isinstance(chunk.data, Bitmap):
            data = chunk.data.copy()
        elif chunk.data is None:
            data = [chunk.get(i) for i in range(n)]
        else:
            raise ExecutionError(
                f"错误的 chunk 数据: {type(chunk.data)}")
        nulls = Bitmap(n)
        for i in range(n):
            if chunk.null_bitmap.get_bit(i):
                nulls.set_bit(i)
        de = chunk.dict_encoded
        packed = _build_packed(chunk.dtype, data, nulls, n)
        return DataVector(
            dtype=chunk.dtype, data=data, nulls=nulls,
            _length=n, dict_encoded=de, _packed=packed)

    @staticmethod
    def from_scalar(value: Any,
                    dtype: DataType) -> 'DataVector':
        """从标量值构建单行 DataVector。"""
        if value is None:
            nulls = Bitmap(1)
            nulls.set_bit(0)
            dt = dtype if dtype != DataType.UNKNOWN else DataType.INT
            return DataVector.from_nulls(dt, 1, nulls)
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        packed = None
        if code is not None:
            data: Any = TypedVector(code, [value])
            packed = _pack_values(dtype, [value], [False])
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = [value]
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(1)
            if value:
                data.set_bit(0)
        else:
            raise ExecutionError(
                f"无法创建 {dtype.name} 的标量向量")
        return DataVector(
            dtype=dtype, data=data, nulls=Bitmap(1),
            _length=1, _packed=packed)

    @staticmethod
    def from_nulls(dtype: DataType, n: int,
                   nulls: Bitmap) -> 'DataVector':
        """构建全 NULL 的 DataVector。"""
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        packed = None
        if code is not None:
            data: Any = TypedVector(code, [0] * n)
            if _HAS_NANBOX and dtype in _NANBOX_TYPES:
                packed = _array.array('Q', [NULL_TAG] * n)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = [''] * n
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
        else:
            data = TypedVector('q', [0] * n)
        return DataVector(
            dtype=dtype, data=data, nulls=nulls,
            _length=n, _packed=packed)

    @staticmethod
    def concat(vectors: 'list[DataVector]') -> 'DataVector':
        """拼接多个 DataVector。"""
        if not vectors:
            raise ExecutionError("空向量列表")
        dtype = vectors[0].dtype
        total = sum(len(v) for v in vectors)
        # NaN-Boxing 拼接
        if all(v._packed is not None for v in vectors):
            new_packed = _array.array('Q')
            for v in vectors:
                new_packed.extend(v._packed)
            new_nulls = Bitmap(total)
            for i in range(total):
                if new_packed[i] == NULL_TAG:
                    new_nulls.set_bit(i)
            new_data = _unpack_to_typed(dtype, new_packed, total)
            return DataVector(
                dtype=dtype, data=new_data,
                nulls=new_nulls, _length=total,
                _packed=new_packed)
        # 标准拼接
        first = vectors[0]
        if isinstance(first.data, TypedVector):
            new_data: Any = TypedVector(first.data.dtype_code)
            for v in vectors:
                new_data.extend(v.data)
        elif isinstance(first.data, list):
            new_data = []
            for v in vectors:
                new_data.extend(v.data[:len(v)])
        elif isinstance(first.data, Bitmap):
            new_data = Bitmap(0)
            for v in vectors:
                new_data.append_from(v.data, len(v))
        else:
            raise ExecutionError(
                f"错误的数据: {type(first.data)}")
        new_nulls = Bitmap(0)
        for v in vectors:
            new_nulls.append_from(v.nulls, len(v))
        packed = _build_packed(dtype, new_data, new_nulls, total)
        return DataVector(
            dtype=dtype, data=new_data, nulls=new_nulls,
            _length=total, _packed=packed)


# ═══ NaN-Boxing 辅助 ═══

def _build_packed(dtype, data, nulls, n):
    """构建 NaN-Boxing packed 数组。"""
    if not _HAS_NANBOX or dtype not in _NANBOX_TYPES:
        return None
    if not isinstance(data, TypedVector):
        return None
    try:
        is_float = dtype in (DataType.FLOAT, DataType.DOUBLE)
        if _HAS_POOL:
            packed = get_default_pool().alloc_packed(n)
        else:
            packed = _array.array('Q', [0] * n)
        for i in range(n):
            if nulls.get_bit(i):
                packed[i] = NULL_TAG
            elif is_float:
                packed[i] = nan_pack_float(float(data[i]))
            else:
                packed[i] = nan_pack_int(int(data[i]))
        return packed
    except Exception:
        return None


def _pack_values(dtype, values, null_flags):
    if not _HAS_NANBOX or dtype not in _NANBOX_TYPES:
        return None
    try:
        n = len(values)
        is_float = dtype in (DataType.FLOAT, DataType.DOUBLE)
        packed = _array.array('Q', [0] * n)
        for i in range(n):
            if null_flags[i]:
                packed[i] = NULL_TAG
            elif is_float:
                packed[i] = nan_pack_float(float(values[i]))
            else:
                packed[i] = nan_pack_int(int(values[i]))
        return packed
    except Exception:
        return None


def _unpack_to_typed(dtype, packed, n):
    """从 packed 数组解码到 TypedVector。"""
    code = DTYPE_TO_ARRAY_CODE.get(dtype)
    if code is None:
        return TypedVector('q', [0] * n)
    td = TypedVector(code)
    is_float = dtype in (DataType.FLOAT, DataType.DOUBLE)
    for i in range(n):
        bits = packed[i]
        if bits == NULL_TAG:
            td.append(0.0 if is_float else 0)
        else:
            _, val = nan_unpack(bits)
            if val is None:
                td.append(0.0 if is_float else 0)
            else:
                td.append(
                    float(val) if is_float else int(val))
    return td
