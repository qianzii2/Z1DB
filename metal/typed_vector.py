from __future__ import annotations
"""TypedVector — RawMemoryBlock 主后端。
[性能] to_array 结果缓存，避免大向量反复全量复制。"""
import array as _array
from typing import List, Optional
from utils.errors import NumericOverflowError

try:
    from metal.memory import RawMemoryBlock
    _HAS_RAW = True
except ImportError:
    _HAS_RAW = False

_RAW_THRESHOLD = 4096


class TypedVector:
    __slots__ = ('_array', '_raw', '_raw_size',
                 'dtype_code', '_use_raw', '_array_cache')

    def __init__(self, dtype_code: str,
                 initial_data: Optional[List] = None) -> None:
        self.dtype_code = dtype_code
        self._raw: Optional[RawMemoryBlock] = None
        self._raw_size = 0
        self._use_raw = False
        self._array_cache: Optional[_array.array] = None  # [性能] to_array 缓存

        if initial_data is not None:
            n = len(initial_data)
            if _HAS_RAW and n >= _RAW_THRESHOLD:
                self._use_raw = True
                self._raw = RawMemoryBlock(dtype_code, max(n, 64))
                self._raw.batch_append(initial_data)
                self._raw_size = n
                self._array = None
            else:
                self._array = _array.array(dtype_code, initial_data)
        else:
            self._array = _array.array(dtype_code)

    @property
    def raw_array(self) -> Optional[_array.array]:
        return self._array

    def append(self, value: object) -> None:
        self._array_cache = None  # 写入后失效缓存
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
        if _HAS_RAW and not self._use_raw and len(self._array) >= _RAW_THRESHOLD:
            self._migrate_to_raw()

    def _migrate_to_raw(self) -> None:
        data = self._array.tolist()
        self._raw = RawMemoryBlock(self.dtype_code, max(len(data) * 2, 64))
        self._raw.batch_append(data)
        self._raw_size = len(data)
        self._use_raw = True
        self._array = None
        self._array_cache = None

    def extend(self, other: 'TypedVector') -> None:
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
            if _HAS_RAW and len(self._array) >= _RAW_THRESHOLD:
                self._migrate_to_raw()

    def batch_append(self, values: list) -> None:
        self._array_cache = None
        if self._use_raw and self._raw is not None:
            self._raw.batch_append(values)
            self._raw_size += len(values)
            return
        if _HAS_RAW and len(values) >= _RAW_THRESHOLD:
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

    def __setitem__(self, index: int, value: object) -> None:
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
        if self._use_raw and self._raw is not None:
            return self._raw.get_batch(0, self._raw_size)
        return self._array.tolist()

    def raw_buffer(self) -> memoryview:
        if self._use_raw and self._raw is not None:
            return self._raw.get_slice(0, self._raw_size)
        return memoryview(self._array)

    def copy(self) -> 'TypedVector':
        if self._use_raw and self._raw is not None:
            return TypedVector(
                self.dtype_code,
                self._raw.get_batch(0, self._raw_size))
        tv = TypedVector(self.dtype_code)
        tv._array = _array.array(self.dtype_code, self._array)
        return tv

    def filter_by_indices(self, indices: 'list[int]') -> 'TypedVector':
        if self._use_raw and self._raw is not None:
            return TypedVector(
                self.dtype_code,
                [self._raw.get(i) for i in indices])
        return TypedVector(
            self.dtype_code,
            [self._array[i] for i in indices])

    def to_array(self) -> _array.array:
        """[性能] 返回 array.array。大向量缓存结果避免重复转换。"""
        if self._use_raw and self._raw is not None:
            if self._array_cache is not None:
                return self._array_cache
            self._array_cache = _array.array(
                self.dtype_code,
                self._raw.get_batch(0, self._raw_size))
            return self._array_cache
        return self._array
