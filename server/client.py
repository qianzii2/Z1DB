from __future__ import annotations
"""Interactive REPL for Z1DB."""
from utils.display import print_result
from utils.errors import Z1Error


class REPL:
    def __init__(self, engine: object) -> None:
        self._engine = engine

    def run(self) -> None:
        print("Z1DB v0.4 — Type .help for commands, .quit to exit")
        buffer = ''
        while True:
            prompt = 'z1db> ' if not buffer else '  ...> '
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print('\nBye')
                break
            stripped = line.strip()
            if not stripped and not buffer:
                continue
            if not buffer and stripped.startswith('.'):
                self._handle_meta(stripped)
                continue
            buffer += line + '\n'
            if self._is_complete(buffer):
                sql = buffer.strip()
                buffer = ''
                self._execute_statements(sql)

    def _execute_statements(self, sql: str) -> None:
        stmts = self._split_statements(sql)
        for stmt in stmts:
            stmt = stmt.strip()
            if not stmt:
                continue
            # Skip comment-only statements
            if not self._has_real_content(stmt):
                continue
            if not stmt.endswith(';'):
                stmt += ';'
            try:
                result = self._engine.execute(stmt)  # type: ignore
                print_result(result)
            except Z1Error as e:
                print(f"Error: {e.message}")

    @staticmethod
    def _has_real_content(stmt: str) -> bool:
        """Check if statement has content beyond comments, whitespace, and semicolons."""
        i = 0
        while i < len(stmt):
            ch = stmt[i]
            if ch.isspace() or ch == ';':
                i += 1
                continue
            if ch == '-' and i + 1 < len(stmt) and stmt[i + 1] == '-':
                # Skip to end of line
                while i < len(stmt) and stmt[i] != '\n':
                    i += 1
                continue
            if ch == '/' and i + 1 < len(stmt) and stmt[i + 1] == '*':
                # Skip block comment
                i += 2
                depth = 1
                while i < len(stmt) and depth > 0:
                    if stmt[i] == '/' and i + 1 < len(stmt) and stmt[i + 1] == '*':
                        depth += 1
                        i += 1
                    elif stmt[i] == '*' and i + 1 < len(stmt) and stmt[i + 1] == '/':
                        depth -= 1
                        i += 1
                    i += 1
                continue
            # Found a real character
            return True
        return False

    def _split_statements(self, sql: str) -> list:
        stmts: list = []
        current: list = []
        in_string = False
        in_line_comment = False
        in_block_comment = False
        block_depth = 0
        i = 0
        while i < len(sql):
            ch = sql[i]
            # Block comment
            if not in_string and not in_line_comment:
                if not in_block_comment and ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
                    in_block_comment = True
                    block_depth = 1
                    current.append(ch)
                    current.append(sql[i + 1])
                    i += 2
                    continue
                if in_block_comment:
                    if ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
                        block_depth += 1
                        current.append(ch)
                        current.append(sql[i + 1])
                        i += 2
                        continue
                    if ch == '*' and i + 1 < len(sql) and sql[i + 1] == '/':
                        block_depth -= 1
                        current.append(ch)
                        current.append(sql[i + 1])
                        i += 2
                        if block_depth == 0:
                            in_block_comment = False
                        continue
                    current.append(ch)
                    i += 1
                    continue
            # Line comment
            if not in_string and not in_block_comment and not in_line_comment:
                if ch == '-' and i + 1 < len(sql) and sql[i + 1] == '-':
                    in_line_comment = True
                    current.append(ch)
                    i += 1
                    continue
            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
                current.append(ch)
                i += 1
                continue
            # String
            if in_string:
                current.append(ch)
                if ch == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    current.append(sql[i + 1])
                    i += 2
                    continue
                if ch == "'":
                    in_string = False
                i += 1
                continue
            if ch == "'":
                in_string = True
                current.append(ch)
                i += 1
                continue
            # Semicolon — statement boundary
            if ch == ';':
                current.append(ch)
                stmts.append(''.join(current))
                current = []
                i += 1
                continue
            current.append(ch)
            i += 1
        remainder = ''.join(current).strip()
        if remainder:
            stmts.append(remainder)
        return stmts

    def _is_complete(self, buf: str) -> bool:
        s = buf.strip()
        if not s:
            return False
        if not s.endswith(';'):
            return False
        depth = 0
        in_str = False
        in_lc = False
        in_bc = False
        bc_depth = 0
        i = 0
        while i < len(s):
            ch = s[i]
            if not in_str and not in_lc and not in_bc:
                if ch == '-' and i + 1 < len(s) and s[i + 1] == '-':
                    in_lc = True
                    i += 2
                    continue
                if ch == '/' and i + 1 < len(s) and s[i + 1] == '*':
                    in_bc = True
                    bc_depth = 1
                    i += 2
                    continue
            if in_bc:
                if ch == '/' and i + 1 < len(s) and s[i + 1] == '*':
                    bc_depth += 1
                    i += 2
                    continue
                if ch == '*' and i + 1 < len(s) and s[i + 1] == '/':
                    bc_depth -= 1
                    i += 2
                    if bc_depth == 0:
                        in_bc = False
                    continue
                i += 1
                continue
            if in_lc:
                if ch == '\n':
                    in_lc = False
                i += 1
                continue
            if in_str:
                if ch == "'" and i + 1 < len(s) and s[i + 1] == "'":
                    i += 2
                    continue
                if ch == "'":
                    in_str = False
            else:
                if ch == "'":
                    in_str = True
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            i += 1
        return depth == 0 and not in_str and not in_bc

    def _handle_meta(self, cmd: str) -> None:
        parts = cmd.split()
        name = parts[0].lower()
        if name == '.quit':
            print('Bye')
            raise SystemExit
        if name == '.help':
            print("  .tables          List tables")
            print("  .schema <table>  Show schema")
            print("  .quit            Exit")
            print("  .help            Help")
            return
        if name == '.tables':
            tables = self._engine.get_table_names()  # type: ignore
            if not tables:
                print("No tables.")
                return
            rows = [(t, str(self._engine.get_table_row_count(t))) for t in tables]  # type: ignore
            self._draw(['Table', 'Rows'], rows)
            return
        if name == '.schema':
            if len(parts) < 2:
                print("Usage: .schema <table>")
                return
            try:
                schema = self._engine.get_table_schema(parts[1])  # type: ignore
            except Z1Error as e:
                print(f"Error: {e.message}")
                return
            rows = []
            for c in schema.columns:
                ts = c.dtype.name + (f'({c.max_length})' if c.max_length else '')
                rows.append((c.name, ts, 'YES' if c.nullable else 'NO'))
            self._draw(['Column', 'Type', 'Nullable'], rows)
            return
        print(f"Unknown command: {name}")

    @staticmethod
    def _draw(headers: list, rows: list) -> None:
        nc = len(headers)
        w = [len(h) for h in headers]
        for r in rows:
            for ci in range(nc):
                w[ci] = max(w[ci], len(str(r[ci])))
        def ln(l: str, m: str, r: str) -> str:
            return l + m.join('─' * (x + 2) for x in w) + r
        print(ln('┌', '┬', '┐'))
        print('│' + '│'.join(f' {headers[i]:<{w[i]}} ' for i in range(nc)) + '│')
        print(ln('├', '┼', '┤'))
        for r in rows:
            print('│' + '│'.join(f' {str(r[i]):<{w[i]}} ' for i in range(nc)) + '│')
        print(ln('└', '┴', '┘'))
