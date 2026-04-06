from __future__ import annotations
"""SQL 语句分割和完整性检测。
处理字符串内引号、行注释、块注释、括号平衡。"""


def split_statements(sql: str) -> list[str]:
    """按分号分割 SQL，正确处理字符串和注释。"""
    stmts = []; cur = []; in_str = False
    in_lc = False; in_bc = False; bcd = 0; i = 0
    while i < len(sql):
        ch = sql[i]
        if not in_str and not in_lc and not in_bc:
            if ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
                in_bc = True; bcd = 1
                cur.append(ch); cur.append(sql[i + 1])
                i += 2; continue
        if in_bc:
            if ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
                bcd += 1; cur.append(ch); cur.append(sql[i + 1])
                i += 2; continue
            if ch == '*' and i + 1 < len(sql) and sql[i + 1] == '/':
                bcd -= 1; cur.append(ch); cur.append(sql[i + 1])
                i += 2
                if bcd == 0: in_bc = False
                continue
            cur.append(ch); i += 1; continue
        if not in_str and not in_lc:
            if ch == '-' and i + 1 < len(sql) and sql[i + 1] == '-':
                in_lc = True; cur.append(ch); i += 1; continue
        if in_lc:
            if ch == '\n': in_lc = False
            cur.append(ch); i += 1; continue
        if in_str:
            cur.append(ch)
            if ch == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                cur.append(sql[i + 1]); i += 2; continue
            if ch == "'": in_str = False
            i += 1; continue
        if ch == "'":
            in_str = True; cur.append(ch); i += 1; continue
        if ch == ';':
            cur.append(ch); stmts.append(''.join(cur))
            cur = []; i += 1; continue
        cur.append(ch); i += 1
    rem = ''.join(cur).strip()
    if rem: stmts.append(rem)
    return stmts


def is_complete(buf: str) -> bool:
    """检查输入缓冲区是否为完整 SQL（以分号结尾且括号平衡）。"""
    s = buf.strip()
    if not s or not s.endswith(';'):
        return False
    depth = 0; in_str = False; in_lc = False
    in_bc = False; bcd = 0; i = 0
    while i < len(s):
        ch = s[i]
        if not in_str and not in_lc and not in_bc:
            if ch == '-' and i + 1 < len(s) and s[i + 1] == '-':
                in_lc = True; i += 2; continue
            if ch == '/' and i + 1 < len(s) and s[i + 1] == '*':
                in_bc = True; bcd = 1; i += 2; continue
        if in_bc:
            if ch == '/' and i + 1 < len(s) and s[i + 1] == '*':
                bcd += 1; i += 2; continue
            if ch == '*' and i + 1 < len(s) and s[i + 1] == '/':
                bcd -= 1; i += 2
            else: i += 1
            if bcd == 0: in_bc = False
            continue
        if in_lc:
            if ch == '\n': in_lc = False
            i += 1; continue
        if in_str:
            if ch == "'" and i + 1 < len(s) and s[i + 1] == "'":
                i += 2; continue
            if ch == "'": in_str = False
        else:
            if ch == "'": in_str = True
            elif ch == '(': depth += 1
            elif ch == ')': depth -= 1
        i += 1
    return depth == 0 and not in_str and not in_bc


def has_real_content(stmt: str) -> bool:
    """检查语句是否有实际内容（非空白/注释/分号）。"""
    i = 0
    while i < len(stmt):
        ch = stmt[i]
        if ch.isspace() or ch == ';':
            i += 1; continue
        if ch == '-' and i + 1 < len(stmt) and stmt[i + 1] == '-':
            while i < len(stmt) and stmt[i] != '\n': i += 1
            continue
        if ch == '/' and i + 1 < len(stmt) and stmt[i + 1] == '*':
            i += 2; d = 1
            while i < len(stmt) and d > 0:
                if stmt[i] == '/' and i + 1 < len(stmt) and stmt[i + 1] == '*':
                    d += 1; i += 1
                elif stmt[i] == '*' and i + 1 < len(stmt) and stmt[i + 1] == '/':
                    d -= 1; i += 1
                i += 1
            continue
        return True
    return False
