from __future__ import annotations
"""Top-N operator — ORDER BY ... LIMIT N using a heap.
O(n log N) instead of O(n log n). Huge win when N << n."""
import heapq
import functools
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class TopNOperator(Operator):
    """Maintains a heap of size N. Only materializes top N rows."""

    def __init__(self, child: Operator,
                 sort_keys: List[Tuple[Any, str, Optional[str]]],
                 n: int) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._sort_keys = sort_keys
        self._n = n
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self.child.open()
        evaluator = ExpressionEvaluator()

        # Collect all rows with their sort keys
        heap: list = []  # list of (sort_key_tuple, row_index, row_data)
        all_rows: list = []
        row_idx = 0

        while True:
            batch = self.child.next_batch()
            if batch is None:
                break

            # Evaluate sort key expressions
            key_vecs = []
            for expr, direction, nulls_pos in self._sort_keys:
                key_vecs.append(evaluator.evaluate(expr, batch).to_python_list())

            col_names = batch.column_names
            for i in range(batch.row_count):
                row = [batch.columns[n].get(i) for n in col_names]
                # Build comparable key
                key_parts = []
                for ki, (_, direction, nulls_pos) in enumerate(self._sort_keys):
                    val = key_vecs[ki][i]
                    key_parts.append(_make_comparable(val, direction, nulls_pos))

                entry = (tuple(key_parts), row_idx, row)
                if len(heap) < self._n:
                    heapq.heappush(heap, _NegEntry(entry))
                else:
                    # Replace if current is better than worst in heap
                    heapq.heappushpop(heap, _NegEntry(entry))
                row_idx += 1

        self.child.close()

        if not heap:
            schema = self.output_schema()
            self._result = VectorBatch.empty(
                [n for n, _ in schema], [t for _, t in schema])
        else:
            # Extract and sort the heap
            entries = sorted([e.entry for e in heap], key=lambda x: x[0])
            rows = [e[2] for e in entries]
            schema = self.output_schema()
            self._result = VectorBatch.from_rows(
                rows,
                [n for n, _ in schema],
                [t for _, t in schema])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass


class _NegEntry:
    """Wrapper for max-heap behavior with heapq (min-heap).
    We want to keep the N smallest entries, so the heap should
    be a max-heap that evicts the largest."""
    __slots__ = ('entry',)

    def __init__(self, entry: tuple) -> None:
        self.entry = entry

    def __lt__(self, other: _NegEntry) -> bool:
        # Reverse comparison for max-heap behavior
        return self.entry[0] > other.entry[0]


def _make_comparable(val: Any, direction: str,
                     nulls_pos: Optional[str]) -> tuple:
    """Create a comparable key tuple for sorting."""
    if nulls_pos is None:
        nulls_pos = 'NULLS_LAST' if direction == 'ASC' else 'NULLS_FIRST'

    if val is None:
        null_priority = 1 if nulls_pos == 'NULLS_LAST' else -1
        return (null_priority, 0)

    if direction == 'DESC':
        # For DESC, we negate numeric values
        if isinstance(val, (int, float)):
            return (0, -val)
        # For strings, we can't negate, use wrapper
        return (0, _Reversed(val))

    return (0, val)


class _Reversed:
    """Wrapper for reverse comparison of non-numeric values."""
    __slots__ = ('val',)

    def __init__(self, val: Any) -> None:
        self.val = val

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, _Reversed):
            return self.val > other.val
        return False

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, _Reversed):
            return self.val < other.val
        return False

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _Reversed):
            return self.val == other.val
        return False

    def __le__(self, other: Any) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other: Any) -> bool:
        return self.__gt__(other) or self.__eq__(other)
