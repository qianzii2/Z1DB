from __future__ import annotations
"""TypedVector — 类型化向量，双后端（array.array + RawMemoryBlock）。
小数据（<4096 元素）用 array.array，大数据自动迁移到 ctypes 连续内存。
to_array() 缓存结果避免重复转换。"""
import array as _array
from typing import List, Optional
from utils.errors import NumericOverflowError
from metal.config import RAW_THRESHOLD

try:
    from metal.memory import RawMemoryBlock
    _HAS_RAW = True
except ImportError:
    _HAS_RAW = False

# int64 安全边界
_MIN_INT64 = -9223372036854775808
_MAX_INT64 = 9223372036854775807


class TypedVector:
    """类型化向量，支持 array.array 和 RawMemoryBlock 双后端。
    写入超过 RAW_THRESHOLD 时自动迁移到 RawMemoryBlock（ctypes 内存）。"""

    __slots__ = ('_array', '_raw', '_raw_size',
                 'dtype_code', '_use_raw', '_array_cache')

    def __init__(self, dtype_code: str,
                 initial_data: list = None) -> None:
        self.dtype_code = dtype_code
        self._raw: Optional[RawMemoryBlock] = None
        self._raw_size: int = 0
        self._use_raw: bool = False
        self._array_cache: Optional[_array.array] = None

        if initial_data is None:
            self._array: Optional[_array.array] = _array.array(
                dtype_code)
        else:
            # int64 边界钳制，防止 OverflowError
            if dtype_code == 'q':
                safe = []
                for v in initial_data:
                    if isinstance(v, int):
                        if v < _MIN_INT64:
                            v = _MIN_INT64
                        elif v > _MAX_INT64:
                            v = _MAX_INT64
                    safe.append(v)
                self._array = _array.array(dtype_code, safe)
            else:
                self._array = _array.array(
                    dtype_code, initial_data)

    @property
    def raw_array(self) -> Optional[_array.array]:
        """返回底层 array.array（仅小数据模式可用）。"""
        return self._array

    def append(self, value: object) -> None:
        """追加单个值。"""
        self._array_cache = None  # 失效缓存
        if self._use_raw and self._raw is not None:
            try:
                self._raw.append(value)
                self._raw_size += 1
            except OverflowError as exc:
                raise NumericOverflowError(str(exc))
            return
        try:
            self._array.append(value)
        except OverflowError as exc:
            raise NumericOverflowError(str(exc))
        # 超过阈值时迁移到 RawMemoryBlock
        if (_HAS_RAW and not self._use_raw
                and len(self._array) >= RAW_THRESHOLD):
            self._migrate_to_raw()

    def _migrate_to_raw(self) -> None:
        """从 array.array 迁移到 RawMemoryBlock。"""
        data = self._array.tolist()
        self._raw = RawMemoryBlock(
            self.dtype_code, max(len(data) * 2, 64))
        self._raw.batch_append(data)
        self._raw_size = len(data)
        self._use_raw = True
        self._array = None
        self._array_cache = None

    def extend(self, other: 'TypedVector') -> None:
        """合并另一个 TypedVector。"""
        self._array_cache = None
        if self._use_raw and self._raw is not None:
            values = other.to_list()
            self._raw.batch_append(values)
            self._raw_size += len(values)
        elif other._use_raw and other._raw is not None:
            if not self._use_raw:
                self._migrate_to_raw()
            values = other.to_list()
            self._raw.batch_append(values)
            self._raw_size += len(values)
        else:
            self._array.extend(other._array)
            if _HAS_RAW and len(self._array) >= RAW_THRESHOLD:
                self._migrate_to_raw()

    def batch_append(self, values: list) -> None:
        """批量追加。"""
        self._array_cache = None
        if self._use_raw and self._raw is not None:
            self._raw.batch_append(values)
            self._raw_size += len(values)
            return
        if (_HAS_RAW
                and len(values) >= RAW_THRESHOLD):
            if not self._use_raw and self._array is not None:
                self._migrate_to_raw()
            if self._raw is not None:
                self._raw.batch_append(values)
                self._raw_size += len(values)
                return
        for v in values:
            self.append(v)

    def __getitem__(self, index: int) -> object:
        if self._use_raw and self._raw is not None:
            return self._raw.get(index)
        return self._array[index]

    def __setitem__(self, index: int,
                    value: object) -> None:
        self._array_cache = None
        if self._use_raw and self._raw is not None:
            self._raw.set(index, value)
        else:
            self._array[index] = value

    def __len__(self) -> int:
        if self._use_raw:
            return self._raw_size
        return len(self._array)

    def to_list(self) -> list:
        """转为 Python list。"""
        if self._use_raw and self._raw is not None:
            return self._raw.get_batch(0, self._raw_size)
        return self._array.tolist()

    def raw_buffer(self) -> memoryview:
        """零拷贝访问底层缓冲区。"""
        if self._use_raw and self._raw is not None:
            return self._raw.get_slice(0, self._raw_size)
        return memoryview(self._array)

    def copy(self) -> 'TypedVector':
        """深拷贝。"""
        if self._use_raw and self._raw is not None:
            return TypedVector(
                self.dtype_code,
                self._raw.get_batch(0, self._raw_size))
        tv = TypedVector(self.dtype_code)
        tv._array = _array.array(
            self.dtype_code, self._array)
        return tv

    def filter_by_indices(self,
                          indices: list[int]) -> 'TypedVector':
        """按索引列表提取子集。"""
        if self._use_raw and self._raw is not None:
            return TypedVector(
                self.dtype_code,
                [self._raw.get(i) for i in indices])
        return TypedVector(
            self.dtype_code,
            [self._array[i] for i in indices])

    def to_array(self) -> _array.array:
        """返回 array.array。大向量缓存结果避免重复转换。"""
        if self._use_raw and self._raw is not None:
            if self._array_cache is not None:
                return self._array_cache
            self._array_cache = _array.array(
                self.dtype_code,
                self._raw.get_batch(0, self._raw_size))
            return self._array_cache
        return self._array
