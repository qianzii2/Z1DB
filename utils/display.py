from __future__ import annotations
"""Result formatting and table rendering.

Uses duck typing to avoid importing higher-layer types.  The *dtype*
parameter is expected to have a ``.name`` attribute (e.g. an ``Enum`` member).
"""

import datetime
from typing import Any


def format_value(val: Any, dtype: Any) -> str:
    """Format a single cell value for display."""
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
            dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=int(val))
            return dt.isoformat()
        except Exception:
            return str(val)
    if name in ('FLOAT', 'DOUBLE'):
        s = str(val)
        if '.' not in s and 'e' not in s and 'E' not in s and 'inf' not in s.lower() and 'nan' not in s.lower():
            s += '.0'
        return s
    return str(val)


def print_result(result: Any) -> None:
    """Pretty-print an ExecutionResult-like object."""
    # DDL / message results
    if getattr(result, 'message', '') and not getattr(result, 'columns', []):
        print(f"{result.message} ({_fmt_time(result.timing)})")
        return

    if getattr(result, 'affected_rows', 0) > 0 and not getattr(result, 'columns', []):
        n = result.affected_rows
        word = 'row' if n == 1 else 'rows'
        print(f"Inserted {n} {word} ({_fmt_time(result.timing)})")
        return

    columns = getattr(result, 'columns', [])
    if columns:
        col_types = getattr(result, 'column_types', [None] * len(columns))
        rows = getattr(result, 'rows', [])
        timing = getattr(result, 'timing', 0.0)

        # Build string matrix
        str_rows: list[list[str]] = []
        for row in rows:
            str_row: list[str] = []
            for ci, val in enumerate(row):
                dt = col_types[ci] if ci < len(col_types) else None
                str_row.append(format_value(val, dt) if dt else str(val) if val is not None else 'NULL')
            str_rows.append(str_row)

        # Column widths
        widths = [len(c) for c in columns]
        for sr in str_rows:
            for ci, s in enumerate(sr):
                if ci < len(widths):
                    widths[ci] = max(widths[ci], len(s))

        # Draw
        def line(left: str, mid: str, right: str, fill: str = '─') -> str:
            return left + mid.join(fill * (w + 2) for w in widths) + right

        print(line('┌', '┬', '┐'))
        header = '│' + '│'.join(f' {columns[i]:<{widths[i]}} ' for i in range(len(columns))) + '│'
        print(header)
        print(line('├', '┼', '┤'))
        for sr in str_rows:
            row_str = '│' + '│'.join(
                f' {sr[i]:<{widths[i]}} ' if i < len(sr) else ' ' * (widths[i] + 2)
                for i in range(len(columns))
            ) + '│'
            print(row_str)
        print(line('└', '┴', '┘'))

        n = len(rows)
        word = 'row' if n == 1 else 'rows'
        print(f"{n} {word} ({_fmt_time(timing)})")
    else:
        timing = getattr(result, 'timing', 0.0)
        msg = getattr(result, 'message', '') or 'OK'
        print(f"{msg} ({_fmt_time(timing)})")


def _fmt_time(t: float) -> str:
    return f"{t:.3f} sec"
