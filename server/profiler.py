from __future__ import annotations
"""性能分析 + 基准测试 + 内存统计。
[D10] estimate_rows 使用公共接口而非私有属性。"""
import struct
import time
from typing import Any, Dict, List, Optional
from utils.errors import Z1Error


class Profiler:
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
            # [D10] 使用公共接口
            catalog = self._engine.get_catalog()
            Resolver().resolve(ast2, catalog)
            Validator().validate(ast2, catalog)
        except Exception:
            pass
        stages['resolve'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = self._engine.execute(sql)
        stages['execute'] = time.perf_counter() - t0
        stages['optimize'] = 0.0
        stages['total'] = (stages['parse']
                           + stages['resolve']
                           + stages['execute'])
        stages['rows'] = result.row_count
        stages['result'] = result
        return stages


def run_benchmark(engine, sql, iterations):
    timings = []
    result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = engine.execute(sql)
        timings.append(time.perf_counter() - t0)
    timings.sort()
    n = len(timings)
    return {
        'iterations': n,
        'avg': sum(timings) / n,
        'min': timings[0],
        'max': timings[-1],
        'p50': timings[n // 2],
        'p95': (timings[int(n * 0.95)]
                if n >= 20 else timings[-1]),
        'p99': (timings[int(n * 0.99)]
                if n >= 100 else timings[-1]),
        'total': sum(timings),
        'rows': result.row_count if result else 0,
    }


def memory_stats(engine):
    rows_data = []
    total_bytes = 0
    for tname in engine.get_table_names():
        store = engine.get_store(tname)  # [D10]
        row_count = store.row_count
        schema = engine.get_table_schema(tname)  # [D10]
        est_bytes = 0
        from storage.types import DTYPE_TO_ARRAY_CODE
        for col in schema.columns:
            code = DTYPE_TO_ARRAY_CODE.get(col.dtype)
            if code:
                est_bytes += row_count * struct.calcsize(code)
            else:
                est_bytes += row_count * 20
            est_bytes += (row_count + 7) // 8
        total_bytes += est_bytes
        rows_data.append(
            (tname, str(row_count), _fmt_bytes(est_bytes)))
    rows_data.append(('─ 合计 ─', '', _fmt_bytes(total_bytes)))
    return rows_data


def estimate_rows(engine, sql):
    """[D10] 使用公共接口。"""
    try:
        from parser.lexer import Lexer
        from parser.parser import Parser
        from parser.ast import SelectStmt, Literal
        ast = Parser(Lexer(sql).tokenize()).parse()
        if not isinstance(ast, SelectStmt):
            return None
        if ast.from_clause is None:
            return "~1 row"
        tname = ast.from_clause.table.name
        catalog = engine.get_catalog()  # [D10]
        if not catalog.table_exists(tname):
            return None
        total = catalog.get_store(tname).row_count
        est = int(total * 0.33) if ast.where else total
        if (ast.limit
                and isinstance(ast.limit, Literal)
                and isinstance(ast.limit.value, int)):
            est = min(est, ast.limit.value)
        return f"~{est:,} rows"
    except Exception:
        return None


def _fmt_bytes(b):
    if b < 1024: return f"{b} B"
    if b < 1024 * 1024: return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.1f} MB"
