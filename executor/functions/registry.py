from __future__ import annotations
"""Aggregate function registry — Phase 5: MEDIAN, ARRAY_AGG, more stats."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import math
from executor.core.vector import DataVector
from storage.types import DataType
from utils.errors import ExecutionError


class AggregateFunction(ABC):
    name: str = ''
    @abstractmethod
    def init(self) -> Any: ...
    @abstractmethod
    def update(self, state: Any, values: Optional[DataVector], row_count: int) -> Any: ...
    @abstractmethod
    def finalize(self, state: Any) -> Any: ...
    @abstractmethod
    def return_type(self, input_types: List[DataType]) -> DataType: ...


class CountAgg(AggregateFunction):
    name = 'COUNT'
    def init(self): return 0
    def update(self, s, v, rc):
        if v is None: return s + rc
        return s + sum(1 for i in range(len(v)) if not v.is_null(i))
    def finalize(self, s): return s
    def return_type(self, it): return DataType.BIGINT

class CountDistinctAgg(AggregateFunction):
    name = 'COUNT_DISTINCT'
    def init(self): return set()
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): s.add(v.get(i))
        return s
    def finalize(self, s): return len(s)
    def return_type(self, it): return DataType.BIGINT

class SumAgg(AggregateFunction):
    name = 'SUM'
    def init(self): return (0, False)
    def update(self, s, v, rc):
        a, h = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): a += v.get(i); h = True
        return (a, h)
    def finalize(self, s): return s[0] if s[1] else None
    def return_type(self, it):
        if it and it[0] in (DataType.FLOAT, DataType.DOUBLE): return DataType.DOUBLE
        return DataType.BIGINT

class AvgAgg(AggregateFunction):
    name = 'AVG'
    def init(self): return (0.0, 0)
    def update(self, s, v, rc):
        sm, c = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): sm += v.get(i); c += 1
        return (sm, c)
    def finalize(self, s): return s[0]/s[1] if s[1] else None
    def return_type(self, it): return DataType.DOUBLE

class MinAgg(AggregateFunction):
    name = 'MIN'
    def init(self): return (None, False)
    def update(self, s, v, rc):
        c, h = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = v.get(i)
                if not h or val < c: c = val; h = True
        return (c, h)
    def finalize(self, s): return s[0] if s[1] else None
    def return_type(self, it): return it[0] if it else DataType.UNKNOWN

class MaxAgg(AggregateFunction):
    name = 'MAX'
    def init(self): return (None, False)
    def update(self, s, v, rc):
        c, h = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = v.get(i)
                if not h or val > c: c = val; h = True
        return (c, h)
    def finalize(self, s): return s[0] if s[1] else None
    def return_type(self, it): return it[0] if it else DataType.UNKNOWN

class StddevAgg(AggregateFunction):
    name = 'STDDEV'
    def init(self): return (0.0, 0.0, 0)
    def update(self, s, v, rc):
        sm, sq, c = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = float(v.get(i)); sm += val; sq += val*val; c += 1
        return (sm, sq, c)
    def finalize(self, s):
        sm, sq, c = s
        if c < 2: return None
        return math.sqrt(max((sq - sm*sm/c) / (c-1), 0))
    def return_type(self, it): return DataType.DOUBLE

class StddevPopAgg(StddevAgg):
    name = 'STDDEV_POP'
    def finalize(self, s):
        sm, sq, c = s
        if c == 0: return None
        return math.sqrt(max((sq - sm*sm/c) / c, 0))

class VarianceAgg(AggregateFunction):
    name = 'VARIANCE'
    def init(self): return (0.0, 0.0, 0)
    def update(self, s, v, rc):
        sm, sq, c = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = float(v.get(i)); sm += val; sq += val*val; c += 1
        return (sm, sq, c)
    def finalize(self, s):
        sm, sq, c = s
        if c < 2: return None
        return (sq - sm*sm/c) / (c-1)
    def return_type(self, it): return DataType.DOUBLE

class VarPopAgg(VarianceAgg):
    name = 'VAR_POP'
    def finalize(self, s):
        sm, sq, c = s
        if c == 0: return None
        return (sq - sm*sm/c) / c

class MedianAgg(AggregateFunction):
    name = 'MEDIAN'
    def init(self): return []
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): s.append(v.get(i))
        return s
    def finalize(self, s):
        if not s: return None
        s.sort()
        n = len(s)
        if n % 2 == 1: return s[n // 2]
        return (s[n//2 - 1] + s[n//2]) / 2
    def return_type(self, it): return DataType.DOUBLE

class ArrayAggFunc(AggregateFunction):
    name = 'ARRAY_AGG'
    def init(self): return []
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): s.append(v.get(i))
        return s
    def finalize(self, s): return s if s else None
    def return_type(self, it): return DataType.VARCHAR  # represented as string

class StringAggFunc(AggregateFunction):
    name = 'STRING_AGG'
    def init(self): return ([], None)
    def update(self, s, v, rc):
        parts, sep = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): parts.append(str(v.get(i)))
        return (parts, sep)
    def finalize(self, s):
        parts, sep = s
        if not parts: return None
        return (sep or ',').join(parts)
    def return_type(self, it): return DataType.VARCHAR

class ModeAgg(AggregateFunction):
    name = 'MODE'
    def init(self): return {}
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = v.get(i); s[val] = s.get(val, 0) + 1
        return s
    def finalize(self, s):
        if not s: return None
        return max(s, key=s.get)
    def return_type(self, it): return it[0] if it else DataType.UNKNOWN


class FunctionRegistry:
    def __init__(self) -> None:
        self._scalars: Dict[str, Any] = {}
        self._aggregates: Dict[str, AggregateFunction] = {}
    def register_aggregate(self, func: AggregateFunction) -> None:
        self._aggregates[func.name.upper()] = func
    def get_aggregate(self, name: str) -> AggregateFunction:
        upper = name.upper()
        if upper not in self._aggregates: raise ExecutionError(f"unknown aggregate: {name}")
        return self._aggregates[upper]
    def has_aggregate(self, name: str) -> bool:
        return name.upper() in self._aggregates
    def register_defaults(self) -> None:
        for cls in (CountAgg, CountDistinctAgg, SumAgg, AvgAgg, MinAgg, MaxAgg,
                    StddevAgg, StddevPopAgg, VarianceAgg, VarPopAgg,
                    MedianAgg, ArrayAggFunc, StringAggFunc, ModeAgg):
            self.register_aggregate(cls())
