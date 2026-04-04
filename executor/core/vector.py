from __future__ import annotations
"""DataVector — typed column vector with null bitmap."""

from typing import Any, List, Optional

from metal.bitmap import Bitmap, BitmapLike
from metal.typed_vector import TypedVector
from storage.column_chunk import ColumnChunk
from storage.types import DTYPE_TO_ARRAY_CODE, DataType
from utils.errors import ExecutionError


class DataVector:
    """A typed array of values with an associated null bitmap."""

    __slots__ = ('dtype', 'data', 'nulls', '_length')

    def __init__(self, dtype: DataType,
                 data: Any,
                 nulls: Bitmap,
                 _length: int) -> None:
        self.dtype = dtype
        self.data = data      # TypedVector | Bitmap | list[str]
        self.nulls = nulls
        self._length = _length

    def __len__(self) -> int:
        return self._length

    def get(self, i: int) -> Any:
        if self.nulls.get_bit(i):
            return None
        if self.dtype == DataType.BOOLEAN:
            assert isinstance(self.data, Bitmap)
            return self.data.get_bit(i)
        if isinstance(self.data, TypedVector):
            return self.data[i]
        if isinstance(self.data, list):
            return self.data[i]
        return None  # pragma: no cover

    def is_null(self, i: int) -> bool:
        return self.nulls.get_bit(i)

    # -- filtering -----------------------------------------------------
    def filter_by_bitmap(self, mask: BitmapLike) -> DataVector:
        return self.filter_by_indices(mask.to_indices())

    def filter_by_indices(self, indices: List[int]) -> DataVector:
        n = len(indices)
        new_nulls = Bitmap(n)

        if isinstance(self.data, TypedVector):
            new_data: Any = TypedVector(self.data.dtype_code, [self.data[i] for i in indices])
        elif isinstance(self.data, list):
            new_data = [self.data[i] for i in indices]
        elif isinstance(self.data, Bitmap):
            new_data = Bitmap(n)
            for j, i in enumerate(indices):
                if self.data.get_bit(i):
                    new_data.set_bit(j)
        else:
            raise ExecutionError(f"unexpected data type in filter: {type(self.data)}")

        for j, i in enumerate(indices):
            if self.nulls.get_bit(i):
                new_nulls.set_bit(j)

        return DataVector(dtype=self.dtype, data=new_data, nulls=new_nulls, _length=n)

    # -- conversion ----------------------------------------------------
    def to_python_list(self) -> list:
        return [self.get(i) for i in range(self._length)]

    # -- factories -----------------------------------------------------
    @staticmethod
    def from_column_chunk(chunk: ColumnChunk) -> DataVector:
        n = chunk.row_count
        if isinstance(chunk.data, TypedVector):
            data: Any = chunk.data.copy()
        elif isinstance(chunk.data, list):
            data = list(chunk.data[:n])
        elif isinstance(chunk.data, Bitmap):
            data = chunk.data.copy()
        else:
            raise ExecutionError(f"unexpected chunk data type: {type(chunk.data)}")

        nulls = Bitmap(n)
        for i in range(n):
            if chunk.null_bitmap.get_bit(i):
                nulls.set_bit(i)
        return DataVector(dtype=chunk.dtype, data=data, nulls=nulls, _length=n)

    @staticmethod
    def from_scalar(value: Any, dtype: DataType) -> DataVector:
        if value is None:
            nulls = Bitmap(1)
            nulls.set_bit(0)
            dt = dtype if dtype != DataType.UNKNOWN else DataType.INT
            return DataVector.from_nulls(dt, 1, nulls)

        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            data: Any = TypedVector(code, [value])
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = [value]
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(1)
            if value:
                data.set_bit(0)
        else:
            raise ExecutionError(f"cannot create scalar vector for type {dtype.name}")
        return DataVector(dtype=dtype, data=data, nulls=Bitmap(1), _length=1)

    @staticmethod
    def from_nulls(dtype: DataType, n: int, nulls: Bitmap) -> DataVector:
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            data: Any = TypedVector(code, [0] * n)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = [''] * n
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
        else:
            data = TypedVector('q', [0] * n)
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)

    @staticmethod
    def concat(vectors: list[DataVector]) -> DataVector:
        if not vectors:
            raise ExecutionError("cannot concat empty vector list")
        dtype = vectors[0].dtype
        total = sum(len(v) for v in vectors)

        if isinstance(vectors[0].data, TypedVector):
            new_data: Any = TypedVector(vectors[0].data.dtype_code)
            for v in vectors:
                new_data.extend(v.data)
        elif isinstance(vectors[0].data, list):
            new_data = []
            for v in vectors:
                new_data.extend(v.data[:len(v)])
        elif isinstance(vectors[0].data, Bitmap):
            new_data = Bitmap(0)
            for v in vectors:
                new_data.append_from(v.data, len(v))
        else:
            raise ExecutionError(f"unexpected data type in concat: {type(vectors[0].data)}")

        new_nulls = Bitmap(0)
        for v in vectors:
            new_nulls.append_from(v.nulls, len(v))

        return DataVector(dtype=dtype, data=new_data, nulls=new_nulls, _length=total)
