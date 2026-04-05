from __future__ import annotations
"""Bloom Filter probe wrapper — pre-filters rows before hash join."""
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from storage.types import DataType
from structures.bloom_filter import BloomFilter


class BloomProbeOperator(Operator):
    """Pre-filter probe side using Bloom Filter from build side."""

    def __init__(self, child: Operator, bloom: BloomFilter,
                 key_expr: Any) -> None:
        super().__init__()
        self.child = child; self.children = [child]
        self._bloom = bloom
        self._key_expr = key_expr
        self._evaluator = ExpressionEvaluator()
        self._closed = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self._closed = False; self.child.open()

    def close(self) -> None:
        if not self._closed: self.child.close(); self._closed = True

    def next_batch(self) -> Optional[VectorBatch]:
        while True:
            batch = self.child.next_batch()
            if batch is None: return None
            key_vec = self._evaluator.evaluate(self._key_expr, batch)
            mask = Bitmap(batch.row_count)
            for i in range(batch.row_count):
                if key_vec.is_null(i): continue
                val = key_vec.get(i)
                if self._bloom.contains(val):
                    mask.set_bit(i)
            filtered = batch.filter_by_bitmap(mask)
            if filtered.row_count > 0: return filtered
