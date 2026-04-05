from __future__ import annotations
"""哈希连接 — 真正哈希表 + BloomFilter 预过滤 + 批量条件评估。
支持 INNER/LEFT/RIGHT/FULL/SEMI/ANTI。
修复：SEMI/ANTI 的 output_schema 只返回左表列。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from parser.ast import BinaryExpr, ColumnRef
from storage.types import DataType

try:
    from executor.core.lazy_batch import LazyBatch
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False

try:
    from structures.bloom_filter import BloomFilter
    _HAS_BLOOM = True
except ImportError:
    _HAS_BLOOM = False


def _ensure(batch: Any) -> Any:
    if _HAS_LAZY and isinstance(batch, LazyBatch):
        return batch.materialize()
    return batch


def extract_equi_keys(on_expr: Any) -> Tuple[Optional[str], Optional[str]]:
    """从ON条件提取等值 join key 的限定列名。"""
    if on_expr is None:
        return None, None
    if isinstance(on_expr, BinaryExpr) and on_expr.op == '=':
        left_col = _get_col_qualified(on_expr.left)
        right_col = _get_col_qualified(on_expr.right)
        if left_col and right_col:
            return left_col, right_col
    if isinstance(on_expr, BinaryExpr) and on_expr.op == 'AND':
        lk, rk = extract_equi_keys(on_expr.left)
        if lk and rk:
            return lk, rk
        return extract_equi_keys(on_expr.right)
    return None, None


def _get_col_qualified(expr: Any) -> Optional[str]:
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.column}" if expr.table else expr.column
    return None


def _has_extra_conditions(on_expr: Any) -> bool:
    """ON条件是否包含非等值条件。"""
    if on_expr is None:
        return False
    if isinstance(on_expr, BinaryExpr):
        if on_expr.op == 'AND':
            return True
        if on_expr.op == '=':
            l = _get_col_qualified(on_expr.left)
            r = _get_col_qualified(on_expr.right)
            return not (l and r)
    return True


class HashJoinOperator(Operator):
    """哈希连接：build 右表哈希表 + BloomFilter，probe 左表探测。"""

    def __init__(self, left: Operator, right: Operator, join_type: str,
                 on_expr: Any) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self.join_type = join_type
        self._on_expr = on_expr
        self._evaluator = ExpressionEvaluator()
        self._result_rows: List[list] = []
        self._out_names: List[str] = []
        self._out_types: List[DataType] = []
        self._emitted = False
        self._left_key, self._right_key = extract_equi_keys(on_expr)
        self._has_extra = _has_extra_conditions(on_expr)

    def output_schema(self) -> List[Tuple[str, DataType]]:
        # 修复：SEMI/ANTI 只输出左表列
        if self.join_type in ('SEMI', 'ANTI'):
            return self.left.output_schema()
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        out_schema = self.output_schema()
        self._out_names = [n for n, _ in out_schema]
        self._out_types = [t for _, t in out_schema]
        left_schema = self.left.output_schema()
        left_col_count = len(left_schema)
        # 完整 schema（含右表）用于条件评估
        full_schema = self.left.output_schema() + self.right.output_schema()

        # Build阶段：物化右表
        right_rows: List[Dict[str, Any]] = []
        while True:
            b = _ensure(self.right.next_batch())
            if b is None:
                break
            for i in range(b.row_count):
                right_rows.append(
                    {n: b.columns[n].get(i) for n in b.column_names})
        self.right.close()

        # 构建哈希表 + BloomFilter
        ht: Dict[Any, List[Tuple[int, Dict[str, Any]]]] = {}
        bloom: Optional[Any] = None
        use_ht = self._right_key is not None

        if use_ht:
            if _HAS_BLOOM and len(right_rows) > 100:
                bloom = BloomFilter(max(len(right_rows), 1), 0.01)
            for ri, r_row in enumerate(right_rows):
                key = r_row.get(self._right_key)
                ht.setdefault(key, []).append((ri, r_row))
                if bloom is not None and key is not None:
                    bloom.add(key)

        # Probe阶段
        self._result_rows = []
        right_matched: set = set()

        while True:
            lb = _ensure(self.left.next_batch())
            if lb is None:
                break
            for li in range(lb.row_count):
                l_row = {n: lb.columns[n].get(li) for n in lb.column_names}

                if use_ht:
                    probe_key = l_row.get(self._left_key)
                    if (bloom is not None and probe_key is not None
                            and not bloom.contains(probe_key)):
                        if self.join_type in ('LEFT', 'FULL'):
                            self._emit_left_null(l_row, left_col_count)
                        elif self.join_type == 'ANTI':
                            self._emit_left_only(l_row, left_col_count)
                        continue
                    candidates = ht.get(probe_key, [])
                else:
                    candidates = list(enumerate(right_rows))

                found = False
                for ri, r_row in candidates:
                    if self._has_extra and self._on_expr is not None:
                        combined = {**l_row, **r_row}
                        if not self._eval_cond(combined, full_schema):
                            continue

                    if self.join_type == 'SEMI':
                        self._emit_left_only(l_row, left_col_count)
                        found = True
                        break
                    elif self.join_type == 'ANTI':
                        found = True
                        break
                    else:
                        combined = {**l_row, **r_row}
                        self._result_rows.append(
                            [combined.get(n) for n in self._out_names])
                        found = True
                        right_matched.add(ri)

                if not found:
                    if self.join_type in ('LEFT', 'FULL'):
                        self._emit_left_null(l_row, left_col_count)
                    elif self.join_type == 'ANTI':
                        self._emit_left_only(l_row, left_col_count)

        self.left.close()

        if self.join_type in ('RIGHT', 'FULL'):
            for ri, r_row in enumerate(right_rows):
                if ri not in right_matched:
                    row = [None] * left_col_count
                    row += [r_row.get(n)
                            for n in self._out_names[left_col_count:]]
                    self._result_rows.append(row)

        self._emitted = False

    def _emit_left_null(self, l_row: dict, left_col_count: int) -> None:
        row = [l_row.get(n) for n in self._out_names[:left_col_count]]
        row += [None] * (len(self._out_names) - left_col_count)
        self._result_rows.append(row)

    def _emit_left_only(self, l_row: dict, left_col_count: int) -> None:
        """SEMI/ANTI：只输出左表列（与 output_schema 一致）。"""
        row = [l_row.get(n) for n in self._out_names[:left_col_count]]
        self._result_rows.append(row)

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(
            self._result_rows, self._out_names, self._out_types)

    def close(self) -> None:
        pass

    def _eval_cond(self, combined: Dict[str, Any],
                   schema: List[Tuple[str, DataType]]) -> bool:
        cols = {}
        for cn, ct in schema:
            val = combined.get(cn)
            cols[cn] = DataVector.from_scalar(
                val, ct if val is not None else DataType.INT)
        batch = VectorBatch(
            columns=cols,
            _column_order=[n for n, _ in schema], _row_count=1)
        try:
            return self._evaluator.evaluate_predicate(
                self._on_expr, batch).get_bit(0)
        except Exception:
            return False
