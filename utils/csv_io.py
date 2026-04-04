from __future__ import annotations
"""CSV import/export utilities."""
import io
from pathlib import Path
from typing import Any, List, Optional


def read_csv(path: str, delimiter: str = ',', has_header: bool = True,
             quote_char: str = '"') -> tuple[list[str], list[list[str]]]:
    """Read CSV file. Returns (headers, rows_of_strings)."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return parse_csv(text, delimiter, has_header, quote_char)


def parse_csv(text: str, delimiter: str = ',', has_header: bool = True,
              quote_char: str = '"') -> tuple[list[str], list[list[str]]]:
    """Parse CSV text. Returns (headers, rows)."""
    lines = _split_lines(text, quote_char)
    if not lines:
        return [], []
    rows: list[list[str]] = []
    for line in lines:
        if not line.strip():
            continue
        rows.append(_parse_row(line, delimiter, quote_char))
    if not rows:
        return [], []
    if has_header:
        headers = rows[0]
        return headers, rows[1:]
    else:
        headers = [f'col{i}' for i in range(len(rows[0]))]
        return headers, rows


def write_csv(path: str, headers: list[str], rows: list[list],
              delimiter: str = ',') -> int:
    """Write CSV file. Returns number of rows written."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(delimiter.join(headers) + '\n')
        for row in rows:
            vals = []
            for v in row:
                if v is None:
                    vals.append('')
                else:
                    s = str(v)
                    if delimiter in s or '"' in s or '\n' in s:
                        s = '"' + s.replace('"', '""') + '"'
                    vals.append(s)
            f.write(delimiter.join(vals) + '\n')
    return len(rows)


def _split_lines(text: str, quote_char: str) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    in_quote = False
    for ch in text:
        if ch == quote_char:
            in_quote = not in_quote
            current.append(ch)
        elif ch == '\n' and not in_quote:
            lines.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        lines.append(''.join(current))
    return lines


def _parse_row(line: str, delimiter: str, quote_char: str) -> list[str]:
    fields: list[str] = []
    current: list[str] = []
    in_quote = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quote:
            if ch == quote_char:
                if i + 1 < len(line) and line[i + 1] == quote_char:
                    current.append(quote_char)
                    i += 2
                    continue
                else:
                    in_quote = False
                    i += 1
                    continue
            current.append(ch)
        else:
            if ch == quote_char:
                in_quote = True
            elif ch == delimiter:
                fields.append(''.join(current))
                current = []
            else:
                current.append(ch)
        i += 1
    fields.append(''.join(current))
    return fields
