from __future__ import annotations
"""Interactive REPL for Z1DB."""
from utils.display import print_result
from utils.errors import Z1Error


class REPL:
    def __init__(self, engine: object) -> None:
        self._engine = engine

    def run(self) -> None:
        print("Z1DB v0.6 — Type .help for commands, .quit to exit")
        buffer = ''
        while True:
            prompt = 'z1db> ' if not buffer else '  ...> '
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print('\nBye'); break
            stripped = line.strip()
            if not stripped and not buffer: continue
            if not buffer and stripped.startswith('.'):
                self._handle_meta(stripped); continue
            buffer += line + '\n'
            if self._is_complete(buffer):
                sql = buffer.strip(); buffer = ''
                self._execute_statements(sql)

    def _execute_statements(self, sql: str) -> None:
        for stmt in self._split_statements(sql):
            stmt = stmt.strip()
            if not stmt or not self._has_real_content(stmt): continue
            if not stmt.endswith(';'): stmt += ';'
            try:
                result = self._engine.execute(stmt)  # type: ignore
                print_result(result)
            except Z1Error as e:
                print(f"Error: {e.message}")

    @staticmethod
    def _has_real_content(stmt: str) -> bool:
        i = 0
        while i < len(stmt):
            ch = stmt[i]
            if ch.isspace() or ch == ';': i += 1; continue
            if ch == '-' and i+1 < len(stmt) and stmt[i+1] == '-':
                while i < len(stmt) and stmt[i] != '\n': i += 1
                continue
            if ch == '/' and i+1 < len(stmt) and stmt[i+1] == '*':
                i += 2; d = 1
                while i < len(stmt) and d > 0:
                    if stmt[i] == '/' and i+1 < len(stmt) and stmt[i+1] == '*': d += 1; i += 1
                    elif stmt[i] == '*' and i+1 < len(stmt) and stmt[i+1] == '/': d -= 1; i += 1
                    i += 1
                continue
            return True
        return False

    def _split_statements(self, sql: str) -> list:
        stmts: list = []; cur: list = []; in_str = False; in_lc = False; in_bc = False; bcd = 0
        i = 0
        while i < len(sql):
            ch = sql[i]
            if not in_str and not in_lc and not in_bc:
                if ch == '/' and i+1 < len(sql) and sql[i+1] == '*':
                    in_bc = True; bcd = 1; cur.append(ch); cur.append(sql[i+1]); i += 2; continue
            if in_bc:
                if ch == '/' and i+1 < len(sql) and sql[i+1] == '*': bcd += 1; cur.append(ch); cur.append(sql[i+1]); i += 2; continue
                if ch == '*' and i+1 < len(sql) and sql[i+1] == '/':
                    bcd -= 1; cur.append(ch); cur.append(sql[i+1]); i += 2
                    if bcd == 0: in_bc = False
                    continue
                cur.append(ch); i += 1; continue
            if not in_str and not in_lc and ch == '-' and i+1 < len(sql) and sql[i+1] == '-':
                in_lc = True; cur.append(ch); i += 1; continue
            if in_lc:
                if ch == '\n': in_lc = False
                cur.append(ch); i += 1; continue
            if in_str:
                cur.append(ch)
                if ch == "'" and i+1 < len(sql) and sql[i+1] == "'": cur.append(sql[i+1]); i += 2; continue
                if ch == "'": in_str = False
                i += 1; continue
            if ch == "'": in_str = True; cur.append(ch); i += 1; continue
            if ch == ';': cur.append(ch); stmts.append(''.join(cur)); cur = []; i += 1; continue
            cur.append(ch); i += 1
        rem = ''.join(cur).strip()
        if rem: stmts.append(rem)
        return stmts

    def _is_complete(self, buf: str) -> bool:
        s = buf.strip()
        if not s or not s.endswith(';'): return False
        depth = 0; in_str = False; in_lc = False; in_bc = False; bcd = 0; i = 0
        while i < len(s):
            ch = s[i]
            if not in_str and not in_lc and not in_bc:
                if ch == '-' and i+1 < len(s) and s[i+1] == '-': in_lc = True; i += 2; continue
                if ch == '/' and i+1 < len(s) and s[i+1] == '*': in_bc = True; bcd = 1; i += 2; continue
            if in_bc:
                if ch == '/' and i+1 < len(s) and s[i+1] == '*': bcd += 1; i += 2; continue
                if ch == '*' and i+1 < len(s) and s[i+1] == '/':
                    bcd -= 1; i += 2
                    if bcd == 0: in_bc = False
                    continue
                i += 1; continue
            if in_lc:
                if ch == '\n': in_lc = False
                i += 1; continue
            if in_str:
                if ch == "'" and i+1 < len(s) and s[i+1] == "'": i += 2; continue
                if ch == "'": in_str = False
            else:
                if ch == "'": in_str = True
                elif ch == '(': depth += 1
                elif ch == ')': depth -= 1
            i += 1
        return depth == 0 and not in_str and not in_bc

    def _handle_meta(self, cmd: str) -> None:
        parts = cmd.split(); name = parts[0].lower()
        if name == '.quit': print('Bye'); raise SystemExit
        if name == '.help':
            print("  .tables          List tables")
            print("  .schema <table>  Show schema")
            print("  .analyze <table> Compute statistics")
            print("  .stats <table>   Show statistics")
            print("  .quit            Exit")
            print("  .help            Help"); return
        if name == '.tables':
            tables = self._engine.get_table_names()  # type: ignore
            if not tables: print("No tables."); return
            rows = [(t, str(self._engine.get_table_row_count(t))) for t in tables]  # type: ignore
            self._draw(['Table','Rows'], rows); return
        if name == '.schema':
            if len(parts) < 2: print("Usage: .schema <table>"); return
            try:
                schema = self._engine.get_table_schema(parts[1])  # type: ignore
            except Z1Error as e: print(f"Error: {e.message}"); return
            rows = []
            for c in schema.columns:
                ts = c.dtype.name + (f'({c.max_length})' if c.max_length else '')
                rows.append((c.name, ts, 'YES' if c.nullable else 'NO'))
            self._draw(['Column','Type','Nullable'], rows); return
        if name == '.analyze':
            if len(parts) < 2: print("Usage: .analyze <table>"); return
            try:
                stats = self._engine.analyze_table(parts[1])  # type: ignore
                print(f"Analyzed {parts[1]}: {stats.row_count} rows")
            except Z1Error as e: print(f"Error: {e.message}")
            except Exception as e: print(f"Error: {e}"); return
            return
        if name == '.stats':
            if len(parts) < 2: print("Usage: .stats <table>"); return
            try:
                stats = self._engine.get_table_stats(parts[1])  # type: ignore
                if stats is None:
                    print(f"No statistics for '{parts[1]}'. Run .analyze {parts[1]} first.")
                    return
                rows = []
                for cn, cs in stats.column_stats.items():
                    rows.append((cn, str(cs.ndv), str(cs.null_count),
                                 str(cs.min_val) if cs.min_val is not None else 'NULL',
                                 str(cs.max_val) if cs.max_val is not None else 'NULL'))
                self._draw(['Column','NDV','Nulls','Min','Max'], rows)
            except Z1Error as e: print(f"Error: {e.message}")
            except Exception as e: print(f"Error: {e}"); return
            return
        print(f"Unknown command: {name}")

    @staticmethod
    def _draw(headers: list, rows: list) -> None:
        nc = len(headers); w = [len(h) for h in headers]
        for r in rows:
            for ci in range(nc): w[ci] = max(w[ci], len(str(r[ci])))
        def ln(l,m,r): return l + m.join('─'*(x+2) for x in w) + r
        print(ln('┌','┬','┐'))
        print('│'+'│'.join(f' {headers[i]:<{w[i]}} ' for i in range(nc))+'│')
        print(ln('├','┼','┤'))
        for r in rows:
            print('│'+'│'.join(f' {str(r[i]):<{w[i]}} ' for i in range(nc))+'│')
        print(ln('└','┴','┘'))
