from __future__ import annotations
"""窗口排名/导航函数 — ROW_NUMBER, RANK, LAG, LEAD, FIRST_VALUE 等。"""
from typing import Any, List, Optional
from executor.expression.evaluator import ExpressionEvaluator


def compute_ranking(fn_name: str, fn: Any, idx: List[int],
                    se: list, batch: Any, results: list,
                    evaluator: ExpressionEvaluator) -> None:
    """计算排名类窗口函数。"""
    ps = len(idx)
    if fn_name == 'ROW_NUMBER':
        for r, i in enumerate(idx, 1):
            results[i] = r
    elif fn_name == 'RANK':
        _compute_rank(idx, se, results, dense=False)
    elif fn_name == 'DENSE_RANK':
        _compute_rank(idx, se, results, dense=True)
    elif fn_name == 'NTILE':
        nt = int(fn.args[0].value) if fn.args else 1
        for p, i in enumerate(idx):
            results[i] = (p * nt) // ps + 1
    elif fn_name == 'PERCENT_RANK':
        _compute_rank(idx, se, results, dense=False)
        for i in idx:
            results[i] = (results[i] - 1) / max(ps - 1, 1)
    elif fn_name == 'CUME_DIST':
        for p, i in enumerate(idx):
            le = p
            while le + 1 < ps and _rows_equal(idx[le + 1], idx[p], se):
                le += 1
            results[i] = (le + 1) / ps


def compute_navigation(fn_name: str, fn: Any, idx: List[int],
                       batch: Any, results: list, frame: Any,
                       ho: bool, evaluator: ExpressionEvaluator) -> None:
    """计算导航类窗口函数：LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE。"""
    ps = len(idx)
    if fn_name in ('LAG', 'LEAD'):
        off = 1; default = None
        if len(fn.args) >= 2:
            off = int(evaluator.evaluate(fn.args[1], batch).get(0) or 1)
        if len(fn.args) >= 3:
            default = evaluator.evaluate(fn.args[2], batch).get(0)
        av = evaluator.evaluate(fn.args[0], batch).to_python_list() if fn.args else None
        for p, i in enumerate(idx):
            s = p - off if fn_name == 'LAG' else p + off
            results[i] = av[idx[s]] if 0 <= s < ps and av else default

    elif fn_name in ('FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE'):
        av = (evaluator.evaluate(fn.args[0], batch).to_python_list()
              if fn.args else [None] * batch.row_count)
        for p, i in enumerate(idx):
            st, en = frame_bounds(frame, p, ps, ho)
            if fn_name == 'FIRST_VALUE':
                results[i] = av[idx[st]]
            elif fn_name == 'LAST_VALUE':
                results[i] = av[idx[en]]
            elif fn_name == 'NTH_VALUE':
                nth = int(fn.args[1].value) if len(fn.args) > 1 else 1
                t = st + nth - 1
                results[i] = av[idx[t]] if t <= en else None


def _compute_rank(idx, se, results, dense):
    if not idx: return
    r = d = 1; results[idx[0]] = 1
    for p in range(1, len(idx)):
        if not _rows_equal(idx[p], idx[p - 1], se):
            if dense: d += 1; r = d
            else: r = p + 1
        results[idx[p]] = r


def _rows_equal(i, j, se):
    for v, _, _ in se:
        if v[i] != v[j]: return False
    return True


def frame_bounds(frame, pos, psize, ho=True):
    """计算窗口帧的起止位置。"""
    if frame is None:
        return (0, pos) if ho else (0, psize - 1)
    s, e = 0, pos
    if frame.start:
        if frame.start.type == 'UNBOUNDED_PRECEDING': s = 0
        elif frame.start.type == 'CURRENT_ROW': s = pos
        elif frame.start.type == 'N_PRECEDING': s = max(0, pos - (frame.start.offset or 0))
        elif frame.start.type == 'N_FOLLOWING': s = min(psize - 1, pos + (frame.start.offset or 0))
    if frame.end:
        if frame.end.type == 'UNBOUNDED_FOLLOWING': e = psize - 1
        elif frame.end.type == 'CURRENT_ROW': e = pos
        elif frame.end.type == 'N_PRECEDING': e = max(0, pos - (frame.end.offset or 0))
        elif frame.end.type == 'N_FOLLOWING': e = min(psize - 1, pos + (frame.end.offset or 0))
    return s, e
