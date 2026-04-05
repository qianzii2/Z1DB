from __future__ import annotations
"""Interactive REPL for Z1DB — Phase 7 with CSV, timer, CTE."""
from utils.display import print_result
from utils.errors import Z1Error


class REPL:
    def __init__(self, engine: object) -> None:
        self._engine = engine
        self._show_timer = True

    def run(self) -> None:
        print("Z1DB v0.8 — Type .help for commands, .quit to exit")
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
                if not self._show_timer:
                    result.timing = 0.0
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
            print("  .tables             List tables")
            print("  .schema <table>     Show schema")
            print("  .analyze <table>    Compute statistics")
            print("  .stats <table>      Show statistics")
            print("  .import <file> <t>  Import CSV into table")
            print("  .export <file> <t>  Export table to CSV")
            print("  .timer on|off       Toggle timing display")
            print("  .quit               Exit")
            print("  .help               Help"); return
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
            except (Z1Error, Exception) as e:
                print(f"Error: {e}"); return
            return
        if name == '.stats':
            if len(parts) < 2: print("Usage: .stats <table>"); return
            try:
                stats = self._engine.get_table_stats(parts[1])  # type: ignore
                if stats is None:
                    print(f"No statistics. Run .analyze {parts[1]} first."); return
                rows = [(cn, str(cs.ndv), str(cs.null_count),
                         str(cs.min_val) if cs.min_val is not None else 'NULL',
                         str(cs.max_val) if cs.max_val is not None else 'NULL')
                        for cn, cs in stats.column_stats.items()]
                self._draw(['Column','NDV','Nulls','Min','Max'], rows)
            except (Z1Error, Exception) as e:
                print(f"Error: {e}"); return
            return
        if name == '.timer':
            if len(parts) >= 2:
                self._show_timer = parts[1].lower() in ('on', '1', 'true')
                print(f"Timer {'on' if self._show_timer else 'off'}.")
            else:
                self._show_timer = not self._show_timer
                print(f"Timer {'on' if self._show_timer else 'off'}.")
            return
        if name == '.import':
            if len(parts) < 3: print("Usage: .import <file> <table>"); return
            self._import_csv(parts[1], parts[2]); return
        if name == '.export':
            if len(parts) < 3: print("Usage: .export <file> <table>"); return
            self._export_csv(parts[1], parts[2]); return
        print(f"Unknown command: {name}")

    def _import_csv(self, filepath: str, table: str) -> None:
        try:
            from utils.csv_io import read_csv
            from storage.types import DataType
            headers, rows = read_csv(filepath)
            if not headers: print("Empty CSV file."); return
            # Create table if not exists
            if not self._engine._catalog.table_exists(table):  # type: ignore
                from catalog.catalog import ColumnSchema, TableSchema
                cols = [ColumnSchema(name=h.lower().replace(' ', '_'), dtype=DataType.VARCHAR, nullable=True)
                        for h in headers]
                schema = TableSchema(name=table, columns=cols)
                self._engine._catalog.create_table(schema)  # type: ignore
            store = self._engine._catalog.get_store(table)  # type: ignore
            schema = self._engine._catalog.get_table(table)  # type: ignore
            count = 0
            for row in rows:
                # Pad or trim row to match schema
                while len(row) < len(schema.columns): row.append(None)
                row = row[:len(schema.columns)]
                # Type conversion attempt
                converted = []
                for val, col in zip(row, schema.columns):
                    if val is None or val == '':
                        converted.append(None)
                    elif col.dtype == DataType.INT:
                        try: converted.append(int(val))
                        except ValueError: converted.append(None)
                    elif col.dtype == DataType.BIGINT:
                        try: converted.append(int(val))
                        except ValueError: converted.append(None)
                    elif col.dtype in (DataType.FLOAT, DataType.DOUBLE):
                        try: converted.append(float(val))
                        except ValueError: converted.append(None)
                    elif col.dtype == DataType.BOOLEAN:
                        converted.append(val.lower() in ('true', '1', 'yes'))
                    else:
                        converted.append(str(val))
                store.append_row(converted)
                count += 1
            print(f"Imported {count} rows into '{table}'.")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error importing: {e}")

    def _export_csv(self, filepath: str, table: str) -> None:
        try:
            from utils.csv_io import write_csv
            schema = self._engine._catalog.get_table(table)  # type: ignore
            store = self._engine._catalog.get_store(table)  # type: ignore
            headers = schema.column_names
            rows = store.read_all_rows()
            count = write_csv(filepath, headers, rows)
            print(f"Exported {count} rows from '{table}' to '{filepath}'.")
        except Z1Error as e:
            print(f"Error: {e.message}")
        except Exception as e:
            print(f"Error exporting: {e}")

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
