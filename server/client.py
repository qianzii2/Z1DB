from __future__ import annotations
"""Interactive REPL for Z1DB."""

from utils.display import print_result
from utils.errors import Z1Error


class REPL:
    """Read-Eval-Print Loop for the Z1DB engine."""

    def __init__(self, engine: object) -> None:
        self._engine = engine  # Engine instance (duck-typed)

    def run(self) -> None:
        print("Z1DB v0.1 — Type .help for commands, .quit to exit")
        buffer = ''
        while True:
            prompt = 'z1db> ' if not buffer else '  ...> '
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print('\nBye')
                break

            stripped = line.strip()

            # Empty input
            if not stripped and not buffer:
                continue

            # Meta-commands (only when no SQL is being accumulated)
            if not buffer and stripped.startswith('.'):
                self._handle_meta(stripped)
                continue

            buffer += line + '\n'

            if self._is_complete(buffer):
                sql = buffer.strip()
                buffer = ''
                try:
                    result = self._engine.execute(sql)  # type: ignore[union-attr]
                    print_result(result)
                except Z1Error as e:
                    print(f"Error: {e.message}")

    # ------------------------------------------------------------------
    def _is_complete(self, buf: str) -> bool:
        """Check whether the buffer contains a complete SQL statement."""
        stripped = buf.strip()
        if not stripped:
            return False
        if not stripped.endswith(';'):
            return False

        depth = 0
        in_string = False
        in_line_comment = False
        i = 0
        while i < len(stripped):
            ch = stripped[i]

            # Line comment
            if not in_string and not in_line_comment and ch == '-' and i + 1 < len(stripped) and stripped[i + 1] == '-':
                in_line_comment = True
                i += 2
                continue

            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
                i += 1
                continue

            if in_string:
                if ch == "'" and i + 1 < len(stripped) and stripped[i + 1] == "'":
                    i += 2
                    continue
                if ch == "'":
                    in_string = False
            else:
                if ch == "'":
                    in_string = True
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            i += 1

        return depth == 0 and not in_string

    # ------------------------------------------------------------------
    def _handle_meta(self, cmd: str) -> None:
        parts = cmd.split()
        name = parts[0].lower()

        if name == '.quit':
            print('Bye')
            raise SystemExit

        if name == '.help':
            print("Commands:")
            print("  .tables          List all tables")
            print("  .schema <table>  Show table schema")
            print("  .quit            Exit")
            print("  .help            Show this message")
            return

        if name == '.tables':
            tables = self._engine.get_table_names()  # type: ignore[union-attr]
            if not tables:
                print("No tables.")
                return
            # Draw table
            rows_data = []
            for t in tables:
                rc = self._engine.get_table_row_count(t)  # type: ignore[union-attr]
                rows_data.append((t, str(rc)))
            headers = ['Table', 'Rows']
            self._draw_table(headers, rows_data)
            return

        if name == '.schema':
            if len(parts) < 2:
                print("Usage: .schema <table_name>")
                return
            tname = parts[1]
            try:
                schema = self._engine.get_table_schema(tname)  # type: ignore[union-attr]
            except Z1Error as e:
                print(f"Error: {e.message}")
                return
            headers = ['Column', 'Type', 'Nullable']
            rows_data = []
            for c in schema.columns:
                type_str = c.dtype.name
                if c.max_length is not None:
                    type_str += f'({c.max_length})'
                rows_data.append((c.name, type_str, 'YES' if c.nullable else 'NO'))
            self._draw_table(headers, rows_data)
            return

        print(f"Unknown command: {name}. Type .help for a list.")

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_table(headers: list, rows: list) -> None:
        """Draw a simple box-drawing table."""
        ncols = len(headers)
        widths = [len(h) for h in headers]
        for row in rows:
            for ci in range(ncols):
                widths[ci] = max(widths[ci], len(str(row[ci])))

        def line(left: str, mid: str, right: str) -> str:
            return left + mid.join('─' * (w + 2) for w in widths) + right

        print(line('┌', '┬', '┐'))
        print('│' + '│'.join(f' {headers[i]:<{widths[i]}} ' for i in range(ncols)) + '│')
        print(line('├', '┼', '┤'))
        for row in rows:
            print('│' + '│'.join(f' {str(row[i]):<{widths[i]}} ' for i in range(ncols)) + '│')
        print(line('└', '┴', '┘'))
