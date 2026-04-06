from __future__ import annotations
"""投影算子 — JIT 编译投影 + SWAR + LazyBatch 按需物化。
[集成] ExprCompiler.compile_projection + CompileCache 缓存编译结果。
热查询的投影表达式编译为 Python 函数，消除逐行 AST 遍历。"""
from typing import Any, Dict, List, Optional, Set, Tuple
import dataclasses
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from parser.ast import FunctionCall as _FunctionCall, ColumnRef, AliasExpr
from storage.types import DataType

try:
    from metal.swar import batch_to_upper, batch_to_lower
    _HAS_SWAR = True
except ImportError:
    _HAS_SWAR = False

try:
    from executor.core.lazy_batch import LazyBatch
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False

try:
    from executor.codegen.compiler import ExprCompiler
    from executor.codegen.cache import CompileCache
    _HAS_JIT = True
except ImportError:
    _HAS_JIT = False

# 全局编译缓存（跨算子实例复用）
_PROJ_COMPILE_CACHE: Optional['CompileCache'] = None
_JIT_THRESHOLD = 256  # 行数超过此阈值才尝试 JIT


def _get_proj_cache() -> Optional['CompileCache']:
    global _PROJ_COMPILE_CACHE
    if _HAS_JIT and _PROJ_COMPILE_CACHE is None:
        _PROJ_COMPILE_CACHE = CompileCache(max_size=512)
    return _PROJ_COMPILE_CACHE


class ProjectOperator(Operator):
    def __init__(self, child: Operator,
                 projections: List[Tuple[str, Any]]) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._projections = projections
        self._evaluator = ExpressionEvaluator()
        self._closed = False
        self._needed_cols: Optional[Set[str]] = None
        # JIT 状态
        self._jit_fn: Optional[Any] = None
        self._jit_attempted = False
        self._jit_names: List[str] = []

    def output_schema(self):
        child_schema: Dict[str, DataType] = dict(self.child.output_schema())
        return [(name, ExpressionEvaluator.infer_type(expr, child_schema))
                for name, expr in self._projections]

    def open(self):
        self._closed = False
        self._jit_attempted = False
        self._jit_fn = None
        self.child.open()
        self._needed_cols = set()
        for _, expr in self._projections:
            self._collect_columns(expr, self._needed_cols)

    def close(self):
        if not self._closed:
            self.child.close()
            self._closed = True

    def next_batch(self):
        raw = self.child.next_batch()
        if raw is None:
            return None

        # LazyBatch 按需物化
        if _HAS_LAZY and isinstance(raw, LazyBatch):
            batch = self._lazy_project(raw)
        else:
            batch = self._ensure_batch(raw)
            if batch is None:
                return None

        # 尝试 JIT 编译投影（首个 batch 时）
        if (not self._jit_attempted and _HAS_JIT
                and batch.row_count >= _JIT_THRESHOLD):
            self._jit_attempted = True
            self._jit_fn = self._try_compile_projection()

        # JIT 批量投影路径
        if self._jit_fn is not None:
            jit_result = self._jit_project(batch)
            if jit_result is not None:
                return jit_result

        # 标准向量化路径
        result_cols: Dict[str, Any] = {}
        names: List[str] = []
        for name, expr in self._projections:
            vec = self._evaluate_with_swar(expr, batch)
            result_cols[name] = vec
            names.append(name)
        return VectorBatch(columns=result_cols, _column_order=names)

    def _jit_project(self, batch: VectorBatch) -> Optional[VectorBatch]:
        """JIT 投影：编译后的函数一次性计算所有列。"""
        try:
            col_names = batch.column_names
            n = batch.row_count
            result_rows = []
            for i in range(n):
                row_dict = {cn: batch.columns[cn].get(i)
                            for cn in col_names}
                projected = self._jit_fn(row_dict)
                result_rows.append(projected)
            if not result_rows:
                schema = self.output_schema()
                return VectorBatch.empty(
                    [nm for nm, _ in schema],
                    [dt for _, dt in schema])
            schema = self.output_schema()
            return VectorBatch.from_rows(
                result_rows,
                [nm for nm, _ in schema],
                [dt for _, dt in schema])
        except Exception:
            # JIT 失败，禁用后续尝试
            self._jit_fn = None
            return None

    def _try_compile_projection(self) -> Optional[Any]:
        """尝试编译投影列表。使用全局 CompileCache。"""
        cache = _get_proj_cache()
        if cache is None:
            return None
        # 用所有投影表达式的组合哈希作为缓存 key
        proj_repr = repr([(name, expr) for name, expr in self._projections])
        expr_hash = cache.hash_expr(proj_repr)
        cached = cache.get(expr_hash)
        if cached is not None:
            return cached
        # 编译
        exprs = [expr for _, expr in self._projections]
        names = [name for name, _ in self._projections]
        fn = ExprCompiler.compile_projection(exprs, names)
        if fn is not None:
            cache.put(expr_hash, fn)
        return fn

    def _lazy_project(self, lazy: LazyBatch) -> VectorBatch:
        """LazyBatch 按需物化需要的列。"""
        needed = self._needed_cols or set()
        indices = lazy.get_indices()
        cols: Dict[str, DataVector] = {}
        for col_name in lazy.column_names:
            if col_name in needed or not needed:
                cols[col_name] = lazy.get_column(col_name)
        return VectorBatch(columns=cols, _column_order=lazy.column_names,
                           _row_count=lazy.row_count)

    def _evaluate_with_swar(self, expr, batch):
        """SWAR 加速 UPPER/LOWER。"""
        if _HAS_SWAR and isinstance(expr, _FunctionCall):
            name = expr.name.upper()
            if name in ('UPPER', 'LOWER') and len(expr.args) == 1:
                return self._swar_string_fn(name, expr.args[0], batch)
        return self._evaluator.evaluate(expr, batch)

    def _swar_string_fn(self, fn_name, arg_expr, batch):
        arg_vec = self._evaluator.evaluate(arg_expr, batch)
        n = len(arg_vec); rd: list = []; rn = Bitmap(n)
        swar_fn = batch_to_upper if fn_name == 'UPPER' else batch_to_lower
        for i in range(n):
            if arg_vec.is_null(i):
                rn.set_bit(i); rd.append('')
            else:
                s = str(arg_vec.get(i))
                try:
                    raw = bytearray(s.encode('ascii'))
                    result = swar_fn(raw)
                    rd.append(result.decode('ascii'))
                except (UnicodeEncodeError, UnicodeDecodeError):
                    rd.append(s.upper() if fn_name == 'UPPER' else s.lower())
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    @staticmethod
    def _collect_columns(expr, cols: Set[str]):
        if isinstance(expr, ColumnRef):
            cols.add(expr.column)
            if expr.table: cols.add(f"{expr.table}.{expr.column}")
            return
        if isinstance(expr, AliasExpr):
            ProjectOperator._collect_columns(expr.expr, cols); return
        if expr is None or not dataclasses.is_dataclass(expr) or isinstance(expr, type):
            return
        for f in dataclasses.fields(expr):
            child = getattr(expr, f.name)
            if isinstance(child, list):
                for item in child: ProjectOperator._collect_columns(item, cols)
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                ProjectOperator._collect_columns(child, cols)
