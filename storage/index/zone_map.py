from __future__ import annotations
"""ZoneMap — chunk 级 min/max/null_count，用于谓词裁剪。
对选择性谓词可跳过 80-95% 的 chunk。"""
from typing import Any, List, Optional
from enum import Enum


class PruneResult(Enum):
    """裁剪结果。"""
    ALWAYS_TRUE = 'ALWAYS_TRUE'    # 所有行满足 → 不需过滤
    ALWAYS_FALSE = 'ALWAYS_FALSE'  # 无行满足 → 跳过整个 chunk
    MAYBE = 'MAYBE'                # 需要逐行扫描


class ZoneMap:
    """单个 chunk 的 ZoneMap。"""

    __slots__ = ('min_val', 'max_val', 'null_count', 'row_count')

    def __init__(self, min_val: Any = None,
                 max_val: Any = None,
                 null_count: int = 0,
                 row_count: int = 0) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.null_count = null_count
        self.row_count = row_count

    def check_gt(self, value: Any) -> PruneResult:
        if self.min_val is None:
            return PruneResult.MAYBE
        try:
            if self.min_val > value:
                return PruneResult.ALWAYS_TRUE
            if self.max_val <= value:
                return PruneResult.ALWAYS_FALSE
        except TypeError:
            pass
        return PruneResult.MAYBE

    def check_gte(self, value: Any) -> PruneResult:
        if self.min_val is None:
            return PruneResult.MAYBE
        try:
            if self.min_val >= value:
                return PruneResult.ALWAYS_TRUE
            if self.max_val < value:
                return PruneResult.ALWAYS_FALSE
        except TypeError:
            pass
        return PruneResult.MAYBE

    def check_lt(self, value: Any) -> PruneResult:
        if self.min_val is None:
            return PruneResult.MAYBE
        try:
            if self.max_val < value:
                return PruneResult.ALWAYS_TRUE
            if self.min_val >= value:
                return PruneResult.ALWAYS_FALSE
        except TypeError:
            pass
        return PruneResult.MAYBE

    def check_lte(self, value: Any) -> PruneResult:
        if self.min_val is None:
            return PruneResult.MAYBE
        try:
            if self.max_val <= value:
                return PruneResult.ALWAYS_TRUE
            if self.min_val > value:
                return PruneResult.ALWAYS_FALSE
        except TypeError:
            pass
        return PruneResult.MAYBE

    def check_eq(self, value: Any) -> PruneResult:
        if self.min_val is None:
            return PruneResult.MAYBE
        try:
            if value < self.min_val or value > self.max_val:
                return PruneResult.ALWAYS_FALSE
        except TypeError:
            pass
        return PruneResult.MAYBE

    def check_ne(self, value: Any) -> PruneResult:
        if self.min_val is None:
            return PruneResult.MAYBE
        try:
            if self.min_val == self.max_val == value:
                return PruneResult.ALWAYS_FALSE
        except TypeError:
            pass
        return PruneResult.MAYBE

    def check_is_null(self) -> PruneResult:
        if self.null_count == 0:
            return PruneResult.ALWAYS_FALSE
        if self.null_count == self.row_count:
            return PruneResult.ALWAYS_TRUE
        return PruneResult.MAYBE

    def check_is_not_null(self) -> PruneResult:
        if self.null_count == 0:
            return PruneResult.ALWAYS_TRUE
        if self.null_count == self.row_count:
            return PruneResult.ALWAYS_FALSE
        return PruneResult.MAYBE

    def check(self, op: str, value: Any) -> PruneResult:
        """统一检查接口。"""
        dispatch = {
            '>': self.check_gt, '>=': self.check_gte,
            '<': self.check_lt, '<=': self.check_lte,
            '=': self.check_eq, '!=': self.check_ne,
        }
        fn = dispatch.get(op)
        if fn:
            return fn(value)
        if op == 'IS_NULL':
            return self.check_is_null()
        if op == 'IS_NOT_NULL':
            return self.check_is_not_null()
        return PruneResult.MAYBE

    @staticmethod
    def from_values(values: list) -> 'ZoneMap':
        """从值列表构建 ZoneMap。"""
        min_v = max_v = None
        null_count = 0
        for v in values:
            if v is None:
                null_count += 1
                continue
            try:
                if min_v is None or v < min_v:
                    min_v = v
                if max_v is None or v > max_v:
                    max_v = v
            except TypeError:
                pass
        return ZoneMap(min_val=min_v, max_val=max_v,
                       null_count=null_count,
                       row_count=len(values))


def prune_chunks(zone_maps: List[ZoneMap], op: str,
                 value: Any) -> List[int]:
    """返回可能包含匹配行的 chunk 索引。ALWAYS_FALSE 的 chunk 被跳过。"""
    return [i for i, zm in enumerate(zone_maps)
            if zm.check(op, value) != PruneResult.ALWAYS_FALSE]
