from __future__ import annotations
"""LazyBatch — 延迟物化。
论文: Abadi et al., 2007 "Materialization Strategies"

Filter 产出 (original_batch, selection_bitmap) 而非立即复制数据。
下游算子按需物化需要的列，减少不必要的数据拷贝。"""
from typing import Any, Dict, List, Optional
from executor.core.batch import VectorBatch
from executor.core.vector import DataVector
from metal.bitmap import Bitmap
from storage.types import DataType


class LazyBatch:
    """延迟物化批处理。持有原始 batch 的引用 + 选择位图。
    在 materialize() 或直接列访问时才真正复制数据。"""

    __slots__ = ('_batch', '_mask', '_indices',
                 '_materialized')

    def __init__(self, batch: VectorBatch,
                 mask: Bitmap) -> None:
        self._batch = batch
        self._mask = mask
        self._indices: Optional[list] = None
        self._materialized: Optional[VectorBatch] = None

    @property
    def row_count(self) -> int:
        return self._mask.popcount()

    @property
    def column_names(self) -> List[str]:
        return self._batch.column_names

    @property
    def columns(self) -> Dict[str, DataVector]:
        """直接列访问时强制物化。"""
        return self.materialize().columns

    @property
    def original(self) -> VectorBatch:
        return self._batch

    @property
    def mask(self) -> Bitmap:
        return self._mask

    def get_indices(self) -> list:
        """惰性计算存活行索引。"""
        if self._indices is None:
            self._indices = self._mask.to_indices()
        return self._indices

    def get_column(self, name: str) -> DataVector:
        """获取单列（按 mask 过滤）。比全量 materialize 更高效。"""
        if self._materialized is not None:
            return self._materialized.columns[name]
        vec = self._batch.columns[name]
        return vec.filter_by_indices(self.get_indices())

    def materialize(self) -> VectorBatch:
        """强制全量物化。只执行一次。"""
        if self._materialized is None:
            self._materialized = self._batch.filter_by_bitmap(
                self._mask)
        return self._materialized

    def to_rows(self) -> List[list]:
        return self.materialize().to_rows()

    def add_column(self, name: str,
                   vec: DataVector) -> None:
        """添加列时强制物化。"""
        m = self.materialize()
        m.add_column(name, vec)
        self._materialized = m


def is_lazy(batch: Any) -> bool:
    """检查 batch 是否为延迟物化。"""
    return isinstance(batch, LazyBatch)


def ensure_batch(batch: Any) -> VectorBatch:
    """确保获得具体 VectorBatch，必要时物化。"""
    if isinstance(batch, LazyBatch):
        return batch.materialize()
    return batch
