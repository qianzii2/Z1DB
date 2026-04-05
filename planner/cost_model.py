from __future__ import annotations
"""Cost model — estimates CPU and I/O cost for physical operators.
All costs in abstract units proportional to nanoseconds.

Key insight: the cost model doesn't need to be perfectly accurate,
it just needs to correctly RANK alternative plans."""
import math
from typing import Any, Dict, Optional
from storage.types import DataType


# Hardware cost constants (calibrated for Python interpreter)
CPU_TUPLE_COST = 0.01        # Process one tuple
CPU_OPERATOR_COST = 0.0025   # Per-tuple operator overhead
CPU_HASH_COST = 0.05         # One hash computation
CPU_COMPARE_COST = 0.02      # One comparison
CPU_SORT_COST_FACTOR = 1.5   # log factor for sorting
SEQ_SCAN_COST = 1.0          # Read one tuple sequentially
RANDOM_IO_COST = 4.0         # Random I/O (4x sequential)
MEMORY_COST = 0.001          # Per-byte memory usage cost


class CostEstimate:
    """Estimated cost of a physical plan node."""
    __slots__ = ('cpu', 'io', 'memory', 'rows', 'width', 'startup')

    def __init__(self, cpu: float = 0, io: float = 0, memory: float = 0,
                 rows: float = 0, width: float = 100, startup: float = 0) -> None:
        self.cpu = cpu
        self.io = io
        self.memory = memory
        self.rows = rows
        self.width = width
        self.startup = startup  # Cost before first row emitted

    @property
    def total(self) -> float:
        return self.cpu + self.io + self.memory + self.startup

    def __repr__(self) -> str:
        return (f"Cost(cpu={self.cpu:.1f}, io={self.io:.1f}, "
                f"mem={self.memory:.1f}, rows={self.rows:.0f}, total={self.total:.1f})")

    def __lt__(self, other: CostEstimate) -> bool:
        return self.total < other.total

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            cpu=self.cpu + other.cpu,
            io=self.io + other.io,
            memory=self.memory + other.memory,
            rows=self.rows,  # Keep this node's output rows
            width=self.width,
            startup=self.startup + other.startup)


class CostModel:
    """Estimates execution cost for different physical operators."""

    @staticmethod
    def seq_scan(rows: int, width: int = 100) -> CostEstimate:
        return CostEstimate(
            cpu=rows * CPU_TUPLE_COST,
            io=rows * SEQ_SCAN_COST,
            rows=rows, width=width)

    @staticmethod
    def filter(input_cost: CostEstimate, selectivity: float) -> CostEstimate:
        output_rows = max(1, input_cost.rows * selectivity)
        return CostEstimate(
            cpu=input_cost.cpu + input_cost.rows * CPU_COMPARE_COST,
            io=input_cost.io,
            rows=output_rows, width=input_cost.width)

    @staticmethod
    def hash_join(left: CostEstimate, right: CostEstimate,
                  selectivity: float = 0.1) -> CostEstimate:
        """Hash Join: build on right, probe with left.
        Cost = scan both + build HT + probe HT."""
        build_cost = right.rows * CPU_HASH_COST  # Hash all right rows
        probe_cost = left.rows * CPU_HASH_COST   # Hash all left rows
        output_rows = max(1, left.rows * right.rows * selectivity)
        ht_memory = right.rows * right.width * MEMORY_COST
        return CostEstimate(
            cpu=left.cpu + right.cpu + build_cost + probe_cost + output_rows * CPU_TUPLE_COST,
            io=left.io + right.io,
            memory=ht_memory,
            rows=output_rows,
            width=left.width + right.width,
            startup=right.cpu + right.io + build_cost)  # Must build entire HT first

    @staticmethod
    def nested_loop_join(left: CostEstimate, right: CostEstimate,
                         selectivity: float = 0.1) -> CostEstimate:
        """Nested Loop: for each left row, scan all right rows."""
        output_rows = max(1, left.rows * right.rows * selectivity)
        return CostEstimate(
            cpu=left.cpu + left.rows * (right.cpu + right.rows * CPU_COMPARE_COST),
            io=left.io + left.rows * right.io,
            rows=output_rows,
            width=left.width + right.width)

    @staticmethod
    def sort_merge_join(left: CostEstimate, right: CostEstimate,
                        selectivity: float = 0.1, already_sorted: bool = False) -> CostEstimate:
        """Sort Merge Join: sort both, then merge."""
        sort_left = 0 if already_sorted else left.rows * math.log2(max(left.rows, 2)) * CPU_COMPARE_COST
        sort_right = 0 if already_sorted else right.rows * math.log2(max(right.rows, 2)) * CPU_COMPARE_COST
        merge_cost = (left.rows + right.rows) * CPU_COMPARE_COST
        output_rows = max(1, left.rows * right.rows * selectivity)
        return CostEstimate(
            cpu=left.cpu + right.cpu + sort_left + sort_right + merge_cost,
            io=left.io + right.io,
            rows=output_rows,
            width=left.width + right.width,
            startup=sort_left + sort_right)

    @staticmethod
    def hash_agg(input_cost: CostEstimate, ndv: int) -> CostEstimate:
        """Hash aggregation: hash all rows, aggregate per group."""
        return CostEstimate(
            cpu=input_cost.cpu + input_cost.rows * CPU_HASH_COST + ndv * CPU_TUPLE_COST,
            io=input_cost.io,
            memory=ndv * input_cost.width * MEMORY_COST,
            rows=ndv, width=input_cost.width,
            startup=input_cost.cpu + input_cost.io)  # Must consume all input

    @staticmethod
    def sort(input_cost: CostEstimate) -> CostEstimate:
        n = max(input_cost.rows, 2)
        sort_cpu = n * math.log2(n) * CPU_COMPARE_COST * CPU_SORT_COST_FACTOR
        return CostEstimate(
            cpu=input_cost.cpu + sort_cpu,
            io=input_cost.io,
            memory=input_cost.rows * input_cost.width * MEMORY_COST,
            rows=input_cost.rows, width=input_cost.width,
            startup=input_cost.cpu + input_cost.io + sort_cpu)

    @staticmethod
    def top_n(input_cost: CostEstimate, n: int) -> CostEstimate:
        """Top-N via heap. O(rows * log N) instead of O(rows * log rows)."""
        heap_cost = input_cost.rows * math.log2(max(n, 2)) * CPU_COMPARE_COST
        return CostEstimate(
            cpu=input_cost.cpu + heap_cost,
            io=input_cost.io,
            memory=n * input_cost.width * MEMORY_COST,
            rows=min(n, input_cost.rows), width=input_cost.width,
            startup=input_cost.cpu + input_cost.io + heap_cost)

    @staticmethod
    def project(input_cost: CostEstimate, num_cols: int, total_cols: int) -> CostEstimate:
        ratio = num_cols / max(total_cols, 1)
        return CostEstimate(
            cpu=input_cost.cpu + input_cost.rows * CPU_TUPLE_COST,
            io=input_cost.io,
            rows=input_cost.rows,
            width=int(input_cost.width * ratio))

    @staticmethod
    def select_join_algorithm(left: CostEstimate, right: CostEstimate,
                              selectivity: float = 0.1,
                              already_sorted: bool = False) -> tuple:
        """Select cheapest join algorithm. Returns (algorithm_name, cost)."""
        candidates = [
            ('HASH_JOIN', CostModel.hash_join(left, right, selectivity)),
            ('NESTED_LOOP', CostModel.nested_loop_join(left, right, selectivity)),
            ('SORT_MERGE', CostModel.sort_merge_join(left, right, selectivity, already_sorted)),
        ]
        # NL only viable for small right side
        if right.rows > 100:
            candidates = [c for c in candidates if c[0] != 'NESTED_LOOP']
        best = min(candidates, key=lambda c: c[1].total)
        return best
