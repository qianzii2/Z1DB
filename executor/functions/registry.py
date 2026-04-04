from __future__ import annotations
"""Aggregate function registry and built-in implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from executor.core.vector import DataVector
from storage.types import DataType
from utils.errors import ExecutionError


class AggregateFunction(ABC):
    name: str = ''

    @abstractmethod
    def init(self) -> Any: ...

    @abstractmethod
    def update(self, state: Any, values: Optional[DataVector],
               row_count: int) -> Any: ...

    @abstractmethod
    def finalize(self, state: Any) -> Any: ...

    @abstractmethod
    def return_type(self, input_types: List[DataType]) -> DataType: ...

    def merge(self, s1: Any, s2: Any) -> Any:
        raise NotImplementedError


# ======================================================================
# Built-in aggregates
# ======================================================================

class CountAgg(AggregateFunction):
    name = 'COUNT'

    def init(self) -> int:
        return 0

    def update(self, state: int, values: Optional[DataVector],
               row_count: int) -> int:
        if values is None:
            # COUNT(*)
            return state + row_count
        # COUNT(expr) — skip NULLs
        non_null = sum(1 for i in range(len(values)) if not values.is_null(i))
        return state + non_null

    def finalize(self, state: int) -> int:
        return state

    def return_type(self, input_types: List[DataType]) -> DataType:
        return DataType.BIGINT


class SumAgg(AggregateFunction):
    name = 'SUM'

    def init(self) -> tuple:
        return (0, False)  # (accumulator, has_value)

    def update(self, state: tuple, values: Optional[DataVector],
               row_count: int) -> tuple:
        acc, has = state
        if values is None:
            return state
        for i in range(len(values)):
            if not values.is_null(i):
                acc += values.get(i)
                has = True
        return (acc, has)

    def finalize(self, state: tuple) -> Any:
        acc, has = state
        return acc if has else None

    def return_type(self, input_types: List[DataType]) -> DataType:
        if input_types and input_types[0] in (DataType.FLOAT, DataType.DOUBLE):
            return DataType.DOUBLE
        return DataType.BIGINT


class AvgAgg(AggregateFunction):
    name = 'AVG'

    def init(self) -> tuple:
        return (0.0, 0)  # (sum, count)

    def update(self, state: tuple, values: Optional[DataVector],
               row_count: int) -> tuple:
        s, c = state
        if values is None:
            return state
        for i in range(len(values)):
            if not values.is_null(i):
                s += values.get(i)
                c += 1
        return (s, c)

    def finalize(self, state: tuple) -> Any:
        s, c = state
        if c == 0:
            return None
        return s / c

    def return_type(self, input_types: List[DataType]) -> DataType:
        return DataType.DOUBLE


class MinAgg(AggregateFunction):
    name = 'MIN'

    def init(self) -> tuple:
        return (None, False)

    def update(self, state: tuple, values: Optional[DataVector],
               row_count: int) -> tuple:
        cur, has = state
        if values is None:
            return state
        for i in range(len(values)):
            if not values.is_null(i):
                v = values.get(i)
                if not has or v < cur:
                    cur = v
                    has = True
        return (cur, has)

    def finalize(self, state: tuple) -> Any:
        cur, has = state
        return cur if has else None

    def return_type(self, input_types: List[DataType]) -> DataType:
        return input_types[0] if input_types else DataType.UNKNOWN


class MaxAgg(AggregateFunction):
    name = 'MAX'

    def init(self) -> tuple:
        return (None, False)

    def update(self, state: tuple, values: Optional[DataVector],
               row_count: int) -> tuple:
        cur, has = state
        if values is None:
            return state
        for i in range(len(values)):
            if not values.is_null(i):
                v = values.get(i)
                if not has or v > cur:
                    cur = v
                    has = True
        return (cur, has)

    def finalize(self, state: tuple) -> Any:
        cur, has = state
        return cur if has else None

    def return_type(self, input_types: List[DataType]) -> DataType:
        return input_types[0] if input_types else DataType.UNKNOWN


# ======================================================================

class FunctionRegistry:
    """Central registry of scalar and aggregate functions."""

    def __init__(self) -> None:
        self._scalars: Dict[str, Any] = {}
        self._aggregates: Dict[str, AggregateFunction] = {}

    def register_aggregate(self, func: AggregateFunction) -> None:
        self._aggregates[func.name.upper()] = func

    def get_aggregate(self, name: str) -> AggregateFunction:
        upper = name.upper()
        if upper not in self._aggregates:
            raise ExecutionError(f"unknown aggregate function: {name}")
        return self._aggregates[upper]

    def register_defaults(self) -> None:
        for cls in (CountAgg, SumAgg, AvgAgg, MinAgg, MaxAgg):
            self.register_aggregate(cls())
