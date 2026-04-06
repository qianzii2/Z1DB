from __future__ import annotations
"""性能分析 + 基准测试 + 内存统计。"""
import struct
import time
from typing import Any, Dict, List, Optional
from utils.errors import Z1Error


class Profiler:
    """通过 Engine.execute() 执行，不跳过 WAL/事务/缓存。"""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def profile(self, sql: str) -> Dict[str, Any]:
        from parser.lexer import Lexer
        from parser.parser import Parser
        from parser.resolver import Resolver
        from parser.validator import Validator
        stages: Dict[str, Any] = {}
        t0 = time.perf_counter()
        Lexer(sql).tokenize()
        stages['parse'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        try:
            ast2 = Parser(Lexer(sql).tokenize()).parse()
            Resolver().resolve(ast2, self._engine._catalog)
            Validator().validate(ast2, self._engine._catalog)
        except Exception: pass
        stages['resolve'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = self._engine.execute(sql)
        stages['execute'] = time.perf_counter() - t0
        stages['optimize'] = 0.0
        stages['total'] = stages['parse'] + stages['resolve'] + stages['execute']
        stages['rows'] = result.row_count
        stages['result'] = result
        return stages


def run_benchmark(engine: Any, sql: str, iterations: int) -> Dict[str, float]:
    timings: List[float] = []; result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = engine.execute(sql)
        timings.append(time.perf_counter() - t0)
    timings.sort(); n = len(timings)
    return {
        'iterations': n, 'avg': sum(timings) / n,
        'min': timings[0], 'max': timings[-1],
        'p50': timings[n // 2],
        'p95': timings[int(n * 0.95)] if n >= 20 else timings[-1],
        'p99': timings[int(n * 0.99)] if n >= 100 else timings[-1],
        'total': sum(timings),
        'rows': result.row_count if result else 0,
    }


def memory_stats(engine: Any) -> List[tuple]:
    rows_data = []; total_bytes = 0
    for tname in engine.get_table_names():
        store = engine._catalog.get_store(tname)
        row_count = store.row_count
        schema = engine._catalog.get_table(tname)
        est_bytes = 0
        from storage.types import DTYPE_TO_ARRAY_CODE
        for col in schema.columns:
            code = DTYPE_TO_ARRAY_CODE.get(col.dtype)
            if code: est_bytes += row_count * struct.calcsize(code)
            else: est_bytes += row_count * 20
            est_bytes += (row_count + 7) // 8
        total_bytes += est_bytes
        rows_data.append((tname, str(row_count), _fmt_bytes(est_bytes)))
    rows_data.append(('─ TOTAL ─', '', _fmt_bytes(total_bytes)))
    return rows_data


def estimate_rows(engine: Any, sql: str) -> Optional[str]:
    try:
        from parser.lexer import Lexer
        from parser.parser import Parser
        from parser.ast import SelectStmt, Literal
        ast = Parser(Lexer(sql).tokenize()).parse()
        if not isinstance(ast, SelectStmt): return None
        if ast.from_clause is None: return "~1 row"
        tname = ast.from_clause.table.name
        if not engine._catalog.table_exists(tname): return None
        total = engine._catalog.get_store(tname).row_count
        est = int(total * 0.33) if ast.where else total
        if ast.limit and isinstance(ast.limit, Literal) and isinstance(ast.limit.value, int):
            est = min(est, ast.limit.value)
        return f"~{est:,} rows"
    except Exception: return None


def _fmt_bytes(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1024 * 1024: return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.1f} MB"
