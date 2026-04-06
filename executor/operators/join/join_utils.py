from __future__ import annotations
"""JOIN 条件评估共享实现。所有 JOIN 算子委托到此。"""
from typing import Any, Dict, List, Tuple
from executor.core.batch import VectorBatch
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType

_SHARED_EVALUATOR = ExpressionEvaluator()


def eval_join_condition(combined_dict: Dict[str, Any],
                        schema: List[Tuple[str, DataType]],
                        on_expr: Any) -> bool:
    """评估 JOIN 条件（共享实现）。"""
    if on_expr is None:
        return True
    cols = {}
    for cn, ct in schema:
        val = combined_dict.get(cn)
        cols[cn] = DataVector.from_scalar(
            val, ct if val is not None else DataType.INT)
    batch = VectorBatch(
        columns=cols,
        _column_order=[n for n, _ in schema],
        _row_count=1)
    try:
        return _SHARED_EVALUATOR.evaluate_predicate(
            on_expr, batch).get_bit(0)
    except Exception:
        return False


def batch_eval_join_condition(combined_rows: List[Dict[str, Any]],
                              schema: List[Tuple[str, DataType]],
                              on_expr: Any) -> List[bool]:
    """批量评估 JOIN 条件。比逐行快 5-10x。"""
    n = len(combined_rows)
    if n == 0:
        return []
    if n == 1:
        return [eval_join_condition(combined_rows[0], schema, on_expr)]
    col_names = [cn for cn, _ in schema]
    col_types = [ct for _, ct in schema]
    rows = [[row.get(cn) for cn in col_names] for row in combined_rows]
    batch = VectorBatch.from_rows(rows, col_names, col_types)
    try:
        mask = _SHARED_EVALUATOR.evaluate_predicate(on_expr, batch)
        return [mask.get_bit(i) for i in range(n)]
    except Exception:
        return [False] * n
