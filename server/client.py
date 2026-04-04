from __future__ import annotations
"""Interactive REPL for Z1DB."""
from utils.display import print_result
from utils.errors import Z1Error


class REPL:
    def __init__(self, engine: object) -> None:
        self._engine = engine

    def run(self) -> None:
        print("Z1DB v0.2 — Type .help for commands, .quit to exit")
        buffer = ''
        while True:
            prompt = 'z1db> ' if not buffer else '  ...> '
            try: line = input(prompt)
            except (EOFError, KeyboardInterrupt): print('\nBye'); break
            stripped = line.strip()
            if not stripped and not buffer: continue
            if not buffer and stripped.startswith('.'):
                self._handle_meta(stripped); continue
            buffer += line + '\n'
            if self._is_complete(buffer):
                sql = buffer.strip(); buffer = ''
                try:
                    result = self._engine.execute(sql)  # type: ignore
                    print_result(result)
                except Z1Error as e:
                    print(f"Error: {e.message}")

    def _is_complete(self, buf: str) -> bool:
        s = buf.strip()
        if not s or not s.endswith(';'): return False
        depth = 0; in_str = False; in_lc = False; i = 0
        while i < len(s):
            ch = s[i]
            if not in_str and not in_lc and ch == '-' and i+1 < len(s) and s[i+1] == '-':
                in_lc = True; i += 2; continue
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
        return depth == 0 and not in_str

    def _handle_meta(self, cmd: str) -> None:
        parts = cmd.split(); name = parts[0].lower()
        if name == '.quit': print('Bye'); raise SystemExit
        if name == '.help':
            print("  .tables          List tables")
            print("  .schema <table>  Show schema")
            print("  .quit            Exit")
            print("  .help            Help"); return
        if name == '.tables':
            tables = self._engine.get_table_names()  # type: ignore
            if not tables: print("No tables."); return
            rows = [(t, str(self._engine.get_table_row_count(t))) for t in tables]  # type: ignore
            self._draw(['Table','Rows'], rows); return
        if name == '.schema':
            if len(parts) < 2: print("Usage: .schema <table>"); return
            try: schema = self._engine.get_table_schema(parts[1])  # type: ignore
            except Z1Error as e: print(f"Error: {e.message}"); return
            rows = []
            for c in schema.columns:
                ts = c.dtype.name + (f'({c.max_length})' if c.max_length else '')
                rows.append((c.name, ts, 'YES' if c.nullable else 'NO'))
            self._draw(['Column','Type','Nullable'], rows); return
        print(f"Unknown command: {name}")

    @staticmethod
    def _draw(headers: list, rows: list) -> None:
        nc = len(headers)
        w = [len(h) for h in headers]
        for r in rows:
            for ci in range(nc): w[ci] = max(w[ci], len(str(r[ci])))
        def ln(l: str, m: str, r: str) -> str:
            return l + m.join('─'*(x+2) for x in w) + r
        print(ln('┌','┬','┐'))
        print('│'+'│'.join(f' {headers[i]:<{w[i]}} ' for i in range(nc))+'│')
        print(ln('├','┼','┤'))
        for r in rows:
            print('│'+'│'.join(f' {str(r[i]):<{w[i]}} ' for i in range(nc))+'│')
        print(ln('└','┴','┘'))
