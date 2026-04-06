from __future__ import annotations
"""ANSI 颜色和格式化辅助。"""
from typing import Any


class Colors:
    """全局颜色开关。"""
    _enabled = False
    RESET = ''; BOLD = ''; DIM = ''
    RED = ''; GREEN = ''; YELLOW = ''; BLUE = ''
    CYAN = ''; MAGENTA = ''
    BG_RED = ''; BG_GREEN = ''

    @classmethod
    def enable(cls):
        cls._enabled = True
        cls.RESET = '\033[0m'; cls.BOLD = '\033[1m'; cls.DIM = '\033[2m'
        cls.RED = '\033[31m'; cls.GREEN = '\033[32m'; cls.YELLOW = '\033[33m'
        cls.BLUE = '\033[34m'; cls.CYAN = '\033[36m'; cls.MAGENTA = '\033[35m'
        cls.BG_RED = '\033[41m'; cls.BG_GREEN = '\033[42m'

    @classmethod
    def disable(cls):
        cls._enabled = False
        cls.RESET = cls.BOLD = cls.DIM = ''
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.CYAN = cls.MAGENTA = ''
        cls.BG_RED = cls.BG_GREEN = ''

    @classmethod
    def is_enabled(cls) -> bool: return cls._enabled


C = Colors


def c_keyword(s): return f"{C.BLUE}{C.BOLD}{s}{C.RESET}" if C._enabled else s
def c_number(s): return f"{C.CYAN}{s}{C.RESET}" if C._enabled else s
def c_string(s): return f"{C.GREEN}{s}{C.RESET}" if C._enabled else s
def c_error(s): return f"{C.RED}{C.BOLD}{s}{C.RESET}" if C._enabled else s
def c_ok(s): return f"{C.GREEN}{s}{C.RESET}" if C._enabled else s
def c_dim(s): return f"{C.DIM}{s}{C.RESET}" if C._enabled else s
def c_bold(s): return f"{C.BOLD}{s}{C.RESET}" if C._enabled else s
def c_header(s): return f"{C.YELLOW}{C.BOLD}{s}{C.RESET}" if C._enabled else s


def format_value_color(val: Any, dtype: Any) -> str:
    """带颜色的值格式化。"""
    from utils.display import format_value
    s = format_value(val, dtype)
    if not C._enabled: return s
    if s == 'NULL': return c_dim('NULL')
    name = dtype.name if hasattr(dtype, 'name') else ''
    if name == 'BOOLEAN': return c_keyword(s)
    if name in ('INT', 'BIGINT', 'FLOAT', 'DOUBLE'): return c_number(s)
    if name in ('VARCHAR', 'TEXT'): return c_string(s)
    if name in ('DATE', 'TIMESTAMP'): return c_number(s)
    return s


def print_result_enhanced(result: Any, show_timer: bool = True) -> None:
    """带颜色的结果打印。"""
    from utils.display import format_value
    from utils.table_renderer import render_table
    timing = getattr(result, 'timing', 0.0) if show_timer else 0.0

    if getattr(result, 'message', '') and not getattr(result, 'columns', []):
        msg = c_ok('OK') if result.message == 'OK' else result.message
        if timing > 0: print(f"{msg} {c_dim(f'({timing:.3f} sec)')}")
        else: print(msg)
        return

    if getattr(result, 'affected_rows', 0) > 0 and not getattr(result, 'columns', []):
        n = result.affected_rows; w = 'row' if n == 1 else 'rows'
        msg = result.message or f'Affected {n} {w}'
        if timing > 0: print(f"{msg} {c_dim(f'({timing:.3f} sec)')}")
        else: print(msg)
        return

    columns = getattr(result, 'columns', [])
    if columns:
        col_types = getattr(result, 'column_types', [None] * len(columns))
        rows = getattr(result, 'rows', [])
        raw_rows = []
        color_rows = []
        for row in rows:
            raw_row = []; color_row = []
            for ci, val in enumerate(row):
                dt = col_types[ci] if ci < len(col_types) else None
                raw = format_value(val, dt) if dt else (str(val) if val is not None else 'NULL')
                colored = format_value_color(val, dt) if dt else (str(val) if val is not None else c_dim('NULL'))
                raw_row.append(raw); color_row.append(colored)
            raw_rows.append(raw_row); color_rows.append(color_row)

        widths = [len(c) for c in columns]
        for sr in raw_rows:
            for ci, s in enumerate(sr):
                if ci < len(widths): widths[ci] = max(widths[ci], len(s))

        lines = render_table(
            columns, raw_rows, widths=widths,
            header_fmt=c_header,
            value_fmt=lambda raw, ri, ci: color_rows[ri][ci])
        for line in lines: print(line)
        n = len(rows); w = 'row' if n == 1 else 'rows'
        if timing > 0: print(f"{c_number(str(n))} {w} {c_dim(f'({timing:.3f} sec)')}")
        else: print(f"{n} {w}")
    else:
        msg = getattr(result, 'message', '') or 'OK'
        if timing > 0: print(f"{msg} {c_dim(f'({timing:.3f} sec)')}")
        else: print(msg)
