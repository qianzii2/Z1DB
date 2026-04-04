from __future__ import annotations
"""Type-safe wrapper around array.array."""

import array as _array
from typing import List, Optional

from utils.errors import NumericOverflowError


class TypedVector:
    """Thin wrapper over ``array.array`` providing type safety and helpers."""

    __slots__ = ('_array', 'dtype_code')

    def __init__(self, dtype_code: str, initial_data: Optional[List] = None) -> None:
        self.dtype_code = dtype_code
        if initial_data is not None:
            self._array: _array.array = _array.array(dtype_code, initial_data)  # type: ignore[arg-type]
        else:
            self._array = _array.array(dtype_code)

    # ------------------------------------------------------------------
    def append(self, value: object) -> None:
        try:
            self._array.append(value)  # type: ignore[arg-type]
        except OverflowError as exc:
            raise NumericOverflowError(str(exc)) from exc

    def extend(self, other: TypedVector) -> None:
        self._array.extend(other._array)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> object:
        return self._array[index]

    def __setitem__(self, index: int, value: object) -> None:
        self._array[index] = value  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self._array)

    # ------------------------------------------------------------------
    def to_list(self) -> list:
        return self._array.tolist()

    def raw_buffer(self) -> memoryview:
        return memoryview(self._array)

    def copy(self) -> TypedVector:
        tv = TypedVector(self.dtype_code)
        tv._array = _array.array(self.dtype_code, self._array)
        return tv

    def filter_by_indices(self, indices: list[int]) -> TypedVector:
        return TypedVector(self.dtype_code, [self._array[i] for i in indices])
