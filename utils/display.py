from __future__ import annotations
"""结果格式化和表格渲染 — 委托到 utils/table_renderer.py。"""
import datetime
from typing import Any

from utils.table_renderer import print_table


def format_value(val: Any, dtype: Any) -> str:
    """将内部值格式化为可显示字符串。"""
    if val is None:
        return 'NULL'
    name = dtype.name if hasattr(dtype, 'name') else str(dtype)
    if name == 'BOOLEAN':
        return 'TRUE' if val else 'FALSE'
    if name == 'DATE':
        try:
            d = datetime.date.fromordinal(int(val) + 719163)
            return d.isoformat()
        except Exception:
            return str(val)
    if name == 'TIMESTAMP':
        try:
            dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(
                microseconds=int(val))
            return dt.isoformat()
        except Exception:
            return str(val)
    if name in ('FLOAT', 'DOUBLE'):
        s = str(val)
        if ('.' not in s and 'e' not in s and 'E' not in s
                and 'inf' not in s.lower() and 'nan' not in s.lower()):
            s += '.0'
        return s
    return str(val)


def print_result(result: Any) -> None:
    """打印查询结果（非 REPL 模式）。"""
    timing = getattr(result, 'timing', 0.0)
    time_str = f" ({timing:.3f} sec)" if timing > 0 else ""

    if getattr(result, 'message', '') and not getattr(result, 'columns', []):
        print(f"{result.message}{time_str}")
        return

    if getattr(result, 'affected_rows', 0) > 0 and not getattr(result, 'columns', []):
        n = result.affected_rows
        word = 'row' if n == 1 else 'rows'
        msg = result.message or f'Inserted {n} {word}'
        print(f"{msg}{time_str}")
        return

    columns = getattr(result, 'columns', [])
    if columns:
        col_types = getattr(result, 'column_types', [None] * len(columns))
        rows = getattr(result, 'rows', [])

        # 格式化为字符串
        str_rows: list = []
        for row in rows:
            str_row = []
            for ci, val in enumerate(row):
                dt = col_types[ci] if ci < len(col_types) else None
                str_row.append(
                    format_value(val, dt) if dt
                    else (str(val) if val is not None else 'NULL'))
            str_rows.append(str_row)

        print_table(columns, str_rows)

        n = len(rows)
        word = 'row' if n == 1 else 'rows'
        print(f"{n} {word}{time_str}")
    else:
        msg = getattr(result, 'message', '') or 'OK'
        print(f"{msg}{time_str}")
