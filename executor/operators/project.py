from __future__ import annotations
"""投影算子 — 支持JIT编译和SWAR加速。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType

try:
    from executor.codegen.compiler import ExprCompiler
    _HAS_JIT = True
except ImportError:
    _HAS_JIT = False

try:
    from metal.swar import batch_to_upper, batch_to_lower
    _HAS_SWAR = True
except ImportError:
    _HAS_SWAR = False


class ProjectOperator(Operator):
    """评估表达式生成输出列。
    优化1: JIT编译投影列表
    优化2: SWAR加速ASCII字符串UPPER/LOWER"""

    def __init__(self, child: Operator, projections: List[Tuple[str, Any]],
                 enable_jit: bool = True) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._projections = projections
        self._evaluator = ExpressionEvaluator()
        self._closed = False
        self._enable_jit = enable_jit and _HAS_JIT
        self._jit_fn: Optional[Any] = None
        self._jit_attempted = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        child_schema: Dict[str, DataType] = dict(self.child.output_schema())
        result = []
        for name, expr in self._projections:
            dt = ExpressionEvaluator.infer_type(expr, child_schema)
            result.append((name, dt))
        return result

    def open(self) -> None:
        self._closed = False
        self.child.open()
        if self._enable_jit and not self._jit_attempted:
            self._jit_attempted = True
            exprs = [expr for _, expr in self._projections]
            names = [name for name, _ in self._projections]
            self._jit_fn = ExprCompiler.compile_projection(exprs, names)

    def close(self) -> None:
        if not self._closed:
            self.child.close()
            self._closed = True

    def next_batch(self) -> Optional[VectorBatch]:
        raw = self.child.next_batch()
        batch = self._ensure_batch(raw)
        if batch is None:
            return None

        # JIT路径
        if self._jit_fn is not None:
            return self._project_jit(batch)

        # 标准路径（可选SWAR加速）
        result_cols: Dict[str, Any] = {}
        names: List[str] = []
        for name, expr in self._projections:
            vec = self._evaluate_with_swar(expr, batch)
            result_cols[name] = vec
            names.append(name)
        return VectorBatch(columns=result_cols, _column_order=names)

    def _project_jit(self, batch: VectorBatch) -> VectorBatch:
        """JIT编译路径。"""
        from executor.core.vector import DataVector
        from metal.bitmap import Bitmap
        from metal.typed_vector import TypedVector
        from storage.types import DTYPE_TO_ARRAY_CODE

        n = batch.row_count
        col_names = batch.column_names
        out_schema = self.output_schema()

        all_results: list = []
        for i in range(n):
            row_dict = {cn: batch.columns[cn].get(i) for cn in col_names}
            projected = self._jit_fn(row_dict)
            all_results.append(projected)

        result_cols: Dict[str, DataVector] = {}
        for ci, (cname, ctype) in enumerate(out_schema):
            col_values = [row[ci] if ci < len(row) else None
                          for row in all_results]
            result_cols[cname] = _list_to_vector(col_values, ctype, n)

        return VectorBatch(
            columns=result_cols,
            _column_order=[n for n, _ in out_schema])

    def _evaluate_with_swar(self, expr: Any, batch: VectorBatch) -> Any:
        """评估表达式，UPPER/LOWER时尝试SWAR加速。"""
        from parser.ast import FunctionCall
        if _HAS_SWAR and isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name in ('UPPER', 'LOWER') and len(expr.args) == 1:
                return self._swar_string_fn(name, expr.args[0], batch)
        return self._evaluator.evaluate(expr, batch)

    def _swar_string_fn(self, fn_name: str, arg_expr: Any,
                        batch: VectorBatch) -> Any:
        """SWAR加速的UPPER/LOWER。"""
        from executor.core.vector import DataVector
        from metal.bitmap import Bitmap

        arg_vec = self._evaluator.evaluate(arg_expr, batch)
        n = len(arg_vec)
        rd: list = []
        rn = Bitmap(n)
        swar_fn = batch_to_upper if fn_name == 'UPPER' else batch_to_lower

        for i in range(n):
            if arg_vec.is_null(i):
                rn.set_bit(i)
                rd.append('')
            else:
                s = str(arg_vec.get(i))
                try:
                    raw = bytearray(s.encode('ascii'))
                    result = swar_fn(raw)
                    rd.append(result.decode('ascii'))
                except (UnicodeEncodeError, UnicodeDecodeError):
                    rd.append(
                        s.upper() if fn_name == 'UPPER' else s.lower())

        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn,
                          _length=n)


def _list_to_vector(values: list, dtype: DataType, n: int) -> Any:
    """通用list→DataVector转换（供多处复用）。"""
    from executor.core.vector import DataVector
    from metal.bitmap import Bitmap
    from metal.typed_vector import TypedVector
    from storage.types import DTYPE_TO_ARRAY_CODE

    if dtype == DataType.UNKNOWN:
        dtype = DataType.INT
    code = DTYPE_TO_ARRAY_CODE.get(dtype)
    nulls = Bitmap(n)
    if dtype in (DataType.VARCHAR, DataType.TEXT):
        data: Any = []
        for i in range(n):
            if values[i] is None:
                nulls.set_bit(i)
                data.append('')
            else:
                data.append(str(values[i]))
    elif dtype == DataType.BOOLEAN:
        data = Bitmap(n)
        for i in range(n):
            if values[i] is None:
                nulls.set_bit(i)
            elif values[i]:
                data.set_bit(i)
    elif code:
        data = TypedVector(code)
        for i in range(n):
            if values[i] is None:
                nulls.set_bit(i)
                data.append(0)
            else:
                data.append(values[i])
    else:
        data = TypedVector('q')
        for i in range(n):
            if values[i] is None:
                nulls.set_bit(i)
                data.append(0)
            else:
                data.append(int(values[i]))
    return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)
