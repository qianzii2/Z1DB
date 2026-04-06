from __future__ import annotations
"""哈希连接 — 流水线式 probe。
[RF6] build 阶段在 open() 物化右表，probe 阶段在 next_batch() 流式处理左表。
不再一次性物化全部结果。"""
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

try:
    from metal.advanced_hash import CuckooHashMap
    _HAS_CUCKOO_HM = True
except ImportError:
    _HAS_CUCKOO_HM = False


def _ensure(batch):
    if _HAS_LAZY and isinstance(batch, LazyBatch):
        return batch.materialize()
    return batch


def extract_equi_keys(on_expr):
    if on_expr is None: return None, None
    if isinstance(on_expr, BinaryExpr) and on_expr.op == '=':
        lc = _get_col_qualified(on_expr.left)
        rc = _get_col_qualified(on_expr.right)
        if lc and rc: return lc, rc
    if isinstance(on_expr, BinaryExpr) and on_expr.op == 'AND':
        lk, rk = extract_equi_keys(on_expr.left)
        if lk and rk: return lk, rk
        return extract_equi_keys(on_expr.right)
    return None, None


def _get_col_qualified(expr):
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.column}" if expr.table else expr.column
    return None


def _has_extra_conditions(on_expr):
    if on_expr is None: return False
    if isinstance(on_expr, BinaryExpr):
        if on_expr.op == 'AND': return True
        if on_expr.op == '=':
            return not (_get_col_qualified(on_expr.left)
                        and _get_col_qualified(on_expr.right))
    return True


class HashJoinOperator(Operator):
    """[RF6] 流水线哈希连接。
    open():      物化右表 → 构建哈希表 + BloomFilter
    next_batch(): 逐 batch 读取左表 → probe → 输出匹配行"""

    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any) -> None:
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self.join_type = join_type
        self._on_expr = on_expr
        self._evaluator = ExpressionEvaluator()
        self._out_names: List[str] = []
        self._out_types: List[DataType] = []
        self._left_key, self._right_key = extract_equi_keys(on_expr)
        self._has_extra = _has_extra_conditions(on_expr)
        # build 阶段产物
        self._ht: Dict[Any, List[Tuple[int, Dict[str, Any]]]] = {}
        self._right_rows: List[Dict[str, Any]] = []
        self._bloom: Optional[BloomFilter] = None
        self._right_matched: set = set()
        self._use_ht = False
        self._left_col_count = 0
        self._full_schema: List[Tuple[str, DataType]] = []
        # probe 阶段状态
        self._left_exhausted = False
        self._pending_right: Optional[VectorBatch] = None

    def output_schema(self):
        if self.join_type in ('SEMI', 'ANTI'):
            return self.left.output_schema()
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        out_schema = self.output_schema()
        self._out_names = [n for n, _ in out_schema]
        self._out_types = [t for _, t in out_schema]
        self._left_col_count = len(self.left.output_schema())
        self._full_schema = self.left.output_schema() + self.right.output_schema()
        self._right_matched = set()
        self._left_exhausted = False
        self._pending_right = None

        # ═══ Build 阶段：物化右表 ═══
        self._right_rows = []
        while True:
            b = _ensure(self.right.next_batch())
            if b is None: break
            for i in range(b.row_count):
                self._right_rows.append(
                    {n: b.columns[n].get(i) for n in b.column_names})
        self.right.close()

        # 构建哈希表
        self._ht = {}
        self._bloom = None
        self._use_ht = self._right_key is not None

        if self._use_ht:
            if _HAS_BLOOM and len(self._right_rows) > 100:
                self._bloom = BloomFilter(max(len(self._right_rows), 1), 0.01)
            for ri, r_row in enumerate(self._right_rows):
                key = r_row.get(self._right_key)
                self._ht.setdefault(key, []).append((ri, r_row))
                if self._bloom is not None and key is not None:
                    self._bloom.add(key)

    def next_batch(self) -> Optional[VectorBatch]:
        """[RF6] 流水线 probe：每次从左表读一个 batch，产出匹配行。"""
        while True:
            # 左表耗尽后输出 RIGHT/FULL 的未匹配右行
            if self._left_exhausted:
                return self._emit_unmatched_right()

            lb = _ensure(self.left.next_batch())
            if lb is None:
                self._left_exhausted = True
                self.left.close()
                # 如果需要输出未匹配右行
                if self.join_type in ('RIGHT', 'FULL'):
                    return self._emit_unmatched_right()
                return None

            result_rows = self._probe_batch(lb)
            if result_rows:
                return VectorBatch.from_rows(
                    result_rows, self._out_names, self._out_types)
            # 本 batch 无匹配行，继续读下一个

    def _probe_batch(self, lb: VectorBatch) -> List[list]:
        """[P03] 批量 probe：预计算 key 列索引，避免逐行构建 dict。"""
        result_rows: List[list] = []
        left_col_count = self._left_col_count
        left_names = lb.column_names
        n = lb.row_count

        # [P03] 预提取列值为 Python list（一次性）
        left_cols = {name: lb.columns[name].to_python_list()
                     for name in left_names}

        # [P03] 预计算 probe key 列索引（open 阶段已知但这里保底）
        probe_keys = None
        if self._use_ht and self._left_key:
            if self._left_key in left_cols:
                probe_keys = left_cols[self._left_key]
            else:
                for name in left_names:
                    if name.endswith(
                            '.' + self._left_key.split('.')[-1]):
                        probe_keys = left_cols[name]
                        break

        # [P03] 预计算左表列名索引（避免逐行 dict 构建）
        left_name_list = list(left_names)
        out_names = self._out_names

        for li in range(n):
            # [P03] 用列表索引代替 dict.get（热路径优化）
            l_vals = [left_cols[name][li] for name in left_name_list]

            if self._use_ht:
                probe_key = (probe_keys[li]
                             if probe_keys is not None
                             else None)
                # BloomFilter 预过滤
                if (self._bloom is not None
                        and probe_key is not None
                        and not self._bloom.contains(probe_key)):
                    if self.join_type in ('LEFT', 'FULL'):
                        result_rows.append(
                            l_vals + [None] * (
                                len(out_names) - left_col_count))
                    elif self.join_type == 'ANTI':
                        result_rows.append(l_vals[:len(out_names)])
                    continue
                candidates = self._ht.get(probe_key, [])
            else:
                candidates = list(enumerate(self._right_rows))

            if not candidates:
                if self.join_type in ('LEFT', 'FULL'):
                    result_rows.append(
                        l_vals + [None] * (
                            len(out_names) - left_col_count))
                elif self.join_type == 'ANTI':
                    result_rows.append(l_vals[:len(out_names)])
                continue

            # 构建 l_row dict（仅用于条件评估，非热路径）
            l_row = dict(zip(left_name_list, l_vals))

            if self._has_extra and self._on_expr is not None:
                self._probe_with_eval(
                    l_row, candidates, left_col_count,
                    result_rows)
            else:
                self._probe_direct(
                    l_row, candidates, left_col_count,
                    result_rows)

        return result_rows

    def _probe_direct(self, l_row, candidates, left_col_count, result_rows):
        found = False
        for ri, r_row in candidates:
            if self.join_type == 'SEMI':
                result_rows.append(self._left_only_row(l_row))
                found = True; break
            elif self.join_type == 'ANTI':
                found = True; break
            else:
                combined = {**l_row, **r_row}
                result_rows.append([combined.get(n) for n in self._out_names])
                found = True; self._right_matched.add(ri)
        if not found:
            if self.join_type in ('LEFT', 'FULL'):
                result_rows.append(self._left_null_row(l_row, left_col_count))
            elif self.join_type == 'ANTI':
                result_rows.append(self._left_only_row(l_row))

    def _probe_with_eval(self, l_row, candidates, left_col_count, result_rows):
        combined_rows = [{**l_row, **r_row} for _, r_row in candidates]
        candidate_indices = [ri for ri, _ in candidates]
        passing = self._batch_eval(combined_rows)
        found = False
        for idx, passes in enumerate(passing):
            if not passes: continue
            ri = candidate_indices[idx]
            if self.join_type == 'SEMI':
                result_rows.append(self._left_only_row(l_row))
                found = True; break
            elif self.join_type == 'ANTI':
                found = True; break
            else:
                result_rows.append(
                    [combined_rows[idx].get(n) for n in self._out_names])
                found = True; self._right_matched.add(ri)
        if not found:
            if self.join_type in ('LEFT', 'FULL'):
                result_rows.append(self._left_null_row(l_row, left_col_count))
            elif self.join_type == 'ANTI':
                result_rows.append(self._left_only_row(l_row))

    def _batch_eval(self, combined_rows):
        n = len(combined_rows)
        if n == 0: return []
        col_names = [cn for cn, _ in self._full_schema]
        col_types = [ct for _, ct in self._full_schema]
        rows = [[row.get(cn) for cn in col_names] for row in combined_rows]
        batch = VectorBatch.from_rows(rows, col_names, col_types)
        try:
            mask = self._evaluator.evaluate_predicate(self._on_expr, batch)
            return [mask.get_bit(i) for i in range(n)]
        except Exception:
            return [False] * n

    def _emit_unmatched_right(self) -> Optional[VectorBatch]:
        """输出 RIGHT/FULL JOIN 的未匹配右行。"""
        if self._pending_right is not None:
            return None  # 已输出过
        rows = []
        left_col_count = self._left_col_count
        right_names = [n for n, _ in self.right.output_schema()]
        for ri, r_row in enumerate(self._right_rows):
            if ri not in self._right_matched:
                row = [None] * left_col_count
                row += [r_row.get(n) for n in right_names]
                rows.append(row)
        self._pending_right = True  # 标记已处理
        if rows:
            return VectorBatch.from_rows(rows, self._out_names, self._out_types)
        return None

    def _left_null_row(self, l_row, left_col_count):
        left_names = [n for n, _ in self.left.output_schema()]
        row = [l_row.get(n) for n in left_names]
        row += [None] * (len(self._out_names) - left_col_count)
        return row

    def _left_only_row(self, l_row):
        return [l_row.get(n) for n in self._out_names]

    def close(self):
        pass
