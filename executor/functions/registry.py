from __future__ import annotations
"""聚合函数注册表 — 22 个聚合。
集成 kll_sketch.py 作为 APPROX_PERCENTILE 的可选后端。"""
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
    def update(self, state: Any, values: Optional[DataVector],
               row_count: int) -> Any: ...
    @abstractmethod
    def finalize(self, state: Any) -> Any: ...
    @abstractmethod
    def return_type(self, input_types: List[DataType]) -> DataType: ...


class CountAgg(AggregateFunction):
    name = 'COUNT'
    def init(self): return 0
    def update(self, s, v, rc):
        if v is None: return s + rc
        return s + sum(1 for i in range(len(v))
                       if not v.is_null(i))
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
        if it and it[0] in (DataType.FLOAT, DataType.DOUBLE):
            return DataType.DOUBLE
        return DataType.BIGINT


class SumDistinctAgg(AggregateFunction):
    name = 'SUM_DISTINCT'
    def init(self): return set()
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): s.add(v.get(i))
        return s
    def finalize(self, s): return sum(s) if s else None
    def return_type(self, it):
        if it and it[0] in (DataType.FLOAT, DataType.DOUBLE):
            return DataType.DOUBLE
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
    def finalize(self, s): return s[0] / s[1] if s[1] else None
    def return_type(self, it): return DataType.DOUBLE


class AvgDistinctAgg(AggregateFunction):
    name = 'AVG_DISTINCT'
    def init(self): return set()
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): s.add(v.get(i))
        return s
    def finalize(self, s):
        return sum(s) / len(s) if s else None
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
    def return_type(self, it):
        return it[0] if it else DataType.UNKNOWN


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
    def return_type(self, it):
        return it[0] if it else DataType.UNKNOWN


class StddevAgg(AggregateFunction):
    name = 'STDDEV'
    def init(self): return (0.0, 0.0, 0)
    def update(self, s, v, rc):
        sm, sq, c = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = float(v.get(i))
                sm += val; sq += val * val; c += 1
        return (sm, sq, c)
    def finalize(self, s):
        sm, sq, c = s
        if c < 2: return None
        return math.sqrt(max((sq - sm * sm / c) / (c - 1), 0))
    def return_type(self, it): return DataType.DOUBLE


class StddevPopAgg(StddevAgg):
    name = 'STDDEV_POP'
    def finalize(self, s):
        sm, sq, c = s
        if c == 0: return None
        return math.sqrt(max((sq - sm * sm / c) / c, 0))


class VarianceAgg(AggregateFunction):
    name = 'VARIANCE'
    def init(self): return (0.0, 0.0, 0)
    def update(self, s, v, rc):
        sm, sq, c = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = float(v.get(i))
                sm += val; sq += val * val; c += 1
        return (sm, sq, c)
    def finalize(self, s):
        sm, sq, c = s
        if c < 2: return None
        return (sq - sm * sm / c) / (c - 1)
    def return_type(self, it): return DataType.DOUBLE


class VarPopAgg(VarianceAgg):
    name = 'VAR_POP'
    def finalize(self, s):
        sm, sq, c = s
        if c == 0: return None
        return (sq - sm * sm / c) / c


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
        s.sort(); n = len(s)
        return (s[n // 2] if n % 2 == 1
                else (s[n // 2 - 1] + s[n // 2]) / 2)
    def return_type(self, it): return DataType.DOUBLE


class PercentileContAgg(AggregateFunction):
    name = 'PERCENTILE_CONT'
    def init(self): return ([], 0.5)
    def update(self, s, v, rc):
        vals, pct = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): vals.append(v.get(i))
        return (vals, pct)
    def finalize(self, s):
        vals, pct = s
        if not vals: return None
        vals.sort(); n = len(vals)
        idx = pct * (n - 1)
        lo = int(idx); hi = min(lo + 1, n - 1)
        frac = idx - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac
    def return_type(self, it): return DataType.DOUBLE


class PercentileDiscAgg(AggregateFunction):
    name = 'PERCENTILE_DISC'
    def init(self): return ([], 0.5)
    def update(self, s, v, rc):
        vals, pct = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): vals.append(v.get(i))
        return (vals, pct)
    def finalize(self, s):
        vals, pct = s
        if not vals: return None
        vals.sort()
        idx = int(math.ceil(pct * len(vals))) - 1
        return vals[max(0, min(idx, len(vals) - 1))]
    def return_type(self, it):
        return it[0] if it else DataType.UNKNOWN


class ModeAgg(AggregateFunction):
    name = 'MODE'
    def init(self): return {}
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = v.get(i)
                s[val] = s.get(val, 0) + 1
        return s
    def finalize(self, s):
        if not s: return None
        return max(s, key=s.get)
    def return_type(self, it):
        return it[0] if it else DataType.UNKNOWN


class ApproxCountDistinctAgg(AggregateFunction):
    name = 'APPROX_COUNT_DISTINCT'
    def init(self):
        from executor.sketch.hyperloglog import HyperLogLog
        return HyperLogLog(p=11)
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): s.add(v.get(i))
        return s
    def finalize(self, s): return s.estimate()
    def return_type(self, it): return DataType.BIGINT


class ApproxPercentileAgg(AggregateFunction):
    """近似分位数。集成：优先用 KLLSketch，回退到 TDigest。"""
    name = 'APPROX_PERCENTILE'

    def init(self):
        # 优先使用 KLL（理论最优空间）
        try:
            from executor.sketch.kll_sketch import KLLSketch
            return ('KLL', KLLSketch(k=200))
        except ImportError:
            pass
        # 回退到 TDigest
        try:
            from executor.sketch.t_digest import TDigest
            return ('TDIGEST', TDigest(compression=100))
        except ImportError:
            return ('LIST', [])

    def update(self, s, v, rc):
        if v is None:
            return s
        tag, sketch = s
        for i in range(len(v)):
            if not v.is_null(i):
                try:
                    val = float(v.get(i))
                    if tag == 'LIST':
                        sketch.append(val)
                    else:
                        sketch.add(val)
                except (ValueError, TypeError):
                    pass
        return (tag, sketch)

    def finalize(self, s):
        tag, sketch = s
        if tag == 'LIST':
            if not sketch:
                return None
            sketch.sort()
            n = len(sketch)
            idx = int(0.5 * (n - 1))
            return sketch[idx]
        try:
            return sketch.quantile(0.5)
        except Exception:
            return None

    def return_type(self, it): return DataType.DOUBLE


class ApproxTopKAgg(AggregateFunction):
    name = 'APPROX_TOP_K'
    def init(self):
        from executor.sketch.count_min_sketch import CountMinSketch
        return (CountMinSketch(width=2048, depth=5), {})
    def update(self, s, v, rc):
        cms, candidates = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i):
                val = v.get(i)
                cms.add(val)
                candidates[val] = candidates.get(val, 0) + 1
        return (cms, candidates)
    def finalize(self, s):
        cms, candidates = s
        if not candidates: return None
        top = sorted(candidates.items(), key=lambda x: -x[1])[:10]
        return str([item[0] for item in top])
    def return_type(self, it): return DataType.VARCHAR


class ArrayAggFunc(AggregateFunction):
    name = 'ARRAY_AGG'
    def init(self): return []
    def update(self, s, v, rc):
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): s.append(v.get(i))
        return s
    def finalize(self, s): return str(s) if s else None
    def return_type(self, it): return DataType.VARCHAR


class StringAggFunc(AggregateFunction):
    name = 'STRING_AGG'
    def init(self): return ([], ',')
    def update(self, s, v, rc):
        parts, sep = s
        if v is None: return s
        for i in range(len(v)):
            if not v.is_null(i): parts.append(str(v.get(i)))
        return (parts, sep)
    def finalize(self, s):
        parts, sep = s
        return sep.join(parts) if parts else None
    def return_type(self, it): return DataType.VARCHAR
    def init_with_sep(self, sep: str) -> tuple:
        return ([], sep)


class GroupingAgg(AggregateFunction):
    name = 'GROUPING'
    def init(self): return 0
    def update(self, s, v, rc): return s
    def finalize(self, s): return s
    def return_type(self, it): return DataType.INT


class FunctionRegistry:
    def __init__(self) -> None:
        self._scalars: Dict[str, Any] = {}
        self._aggregates: Dict[str, AggregateFunction] = {}

    def register_aggregate(self,
                           func: AggregateFunction) -> None:
        self._aggregates[func.name.upper()] = func

    def get_aggregate(self, name: str) -> AggregateFunction:
        upper = name.upper()
        if upper not in self._aggregates:
            raise ExecutionError(f"unknown aggregate: {name}")
        return self._aggregates[upper]

    def has_aggregate(self, name: str) -> bool:
        return name.upper() in self._aggregates

    def register_defaults(self) -> None:
        for cls in (CountAgg, CountDistinctAgg,
                    SumAgg, SumDistinctAgg,
                    AvgAgg, AvgDistinctAgg,
                    MinAgg, MaxAgg,
                    StddevAgg, StddevPopAgg,
                    VarianceAgg, VarPopAgg,
                    MedianAgg,
                    PercentileContAgg, PercentileDiscAgg,
                    ModeAgg,
                    ApproxCountDistinctAgg,
                    ApproxPercentileAgg,
                    ApproxTopKAgg,
                    ArrayAggFunc, StringAggFunc,
                    GroupingAgg):
            self.register_aggregate(cls())
