from __future__ import annotations

"""统一 Unicode 边框表格渲染器。
display.py 和 client.py 共用此模块。"""
from typing import Any, Callable, List, Optional


def render_table(columns: List[str],
                 rows: List[List[str]],
                 widths: Optional[List[int]] = None,
                 header_fmt: Callable[[str], str] = str,
                 value_fmt: Optional[Callable[[str, int, int], str]] = None
                 ) -> List[str]:
    """渲染 Unicode 边框表格，返回行字符串列表。

    参数:
        columns: 列名列表
        rows: 每行为字符串列表（原始宽度用于对齐计算）
        widths: 可选的列宽度列表（None 时自动计算）
        header_fmt: 列头格式化函数（如添加颜色）
        value_fmt: 值格式化函数 (raw_value, row_idx, col_idx) → display_str
    """
    nc = len(columns)
    if nc == 0:
        return []

    # 自动计算列宽
    if widths is None:
        widths = [len(c) for c in columns]
        for r in rows:
            for ci in range(min(nc, len(r))):
                widths[ci] = max(widths[ci], len(r[ci]))

    def line(l: str, m: str, r: str) -> str:
        return l + m.join('─' * (w + 2) for w in widths) + r

    lines: List[str] = []

    # 顶部边框
    lines.append(line('┌', '┬', '┐'))

    # 表头
    hdr_parts = []
    for i in range(nc):
        formatted = header_fmt(columns[i])
        padding = widths[i] - len(columns[i])
        hdr_parts.append(f' {formatted}{" " * padding} ')
    lines.append('│' + '│'.join(hdr_parts) + '│')

    # 表头分隔线
    lines.append(line('├', '┼', '┤'))

    # 数据行
    for ri, row in enumerate(rows):
        parts = []
        for ci in range(nc):
            raw = row[ci] if ci < len(row) else ''
            if value_fmt:
                display = value_fmt(raw, ri, ci)
            else:
                display = raw
            padding = widths[ci] - len(raw)
            parts.append(f' {display}{" " * padding} ')
        lines.append('│' + '│'.join(parts) + '│')

    # 底部边框
    lines.append(line('└', '┴', '┘'))

    return lines


def print_table(columns: List[str],
                rows: List[List[str]],
                **kwargs) -> None:
    """渲染并打印表格。"""
    for line in render_table(columns, rows, **kwargs):
        print(line)
