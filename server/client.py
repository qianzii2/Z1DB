from __future__ import annotations
"""Z1DB REPL — 委托到 colors/profiler/table_renderer 模块。"""
import sys
from typing import Any, Dict, List, Optional
from server.colors import (
    C, Colors, c_bold, c_dim, c_error, c_header,
    c_number, c_ok, print_result_enhanced)
from server.profiler import Profiler, run_benchmark, memory_stats, estimate_rows
from server.query_history import QueryHistory
from utils.errors import Z1Error


class REPL:
    def __init__(self, engine: object) -> None:
        self._engine = engine
        self._show_timer = True
        self._show_estimate = False
        self._show_profile = False
        self._history = QueryHistory()
        self._profiler = Profiler(engine)

    def run(self) -> None:
        print(f"{c_bold('Z1DB')} v1.2 — {c_dim('Pure Python OLAP Engine')}")
        print(c_dim('Type .help for commands, .quit to exit'))
        buffer = ''
        while True:
            prompt = f"{c_bold('z1db>')} " if not buffer else f"{c_dim('  ...>')} "
            try: line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print(f'\n{c_dim("Bye")}'); break
            stripped = line.strip()
            if not stripped and not buffer: continue
            if not buffer and stripped.startswith('.'):
                self._handle_meta(stripped); continue
            buffer += line + '\n'
            if self._is_complete(buffer):
                sql = buffer.strip(); buffer = ''
                self._execute_statements(sql)

    def _execute_statements(self, sql):
        for stmt in self._split_statements(sql):
            stmt = stmt.strip()
            if not stmt or not self._has_real_content(stmt): continue
            if not stmt.endswith(';'): stmt += ';'
            self._execute_one(stmt)

    def _execute_one(self, sql):
        if self._show_estimate:
            est = estimate_rows(self._engine, sql)
            if est: print(c_dim(f"  ⏳ Estimated: {est}"))
        try:
            if self._show_profile:
                stages = self._profiler.profile(sql)
                result = stages['result']; result.timing = stages['total']
                print_result_enhanced(result, self._show_timer)
                self._print_profile_bar(stages)
                self._history.add(sql, stages['total'], stages.get('rows', 0))
            else:
                result = self._engine.execute(sql)
                if not self._show_timer: result.timing = 0.0
                print_result_enhanced(result, self._show_timer)
                self._history.add(sql, result.timing, result.row_count)
        except Z1Error as e:
            print(f"{c_error('Error')}: {e.message}")

    def _print_profile_bar(self, stages):
        total = stages['total']; bar_width = 40
        parts = [('parse', stages['parse'], C.BLUE),
                 ('resolve', stages['resolve'], C.CYAN),
                 ('optimize', stages['optimize'], C.YELLOW),
                 ('execute', stages['execute'], C.GREEN)]
        print(c_dim("  ┌─ Profile ─────────────────────────────────┐"))
        for pname, pt, pcolor in parts:
            pct = pt / total * 100 if total > 0 else 0
            filled = int(pct / 100 * bar_width)
            bar = '█' * filled + '░' * (bar_width - filled)
            bar_colored = f"{pcolor}{bar}{C.RESET}" if C._enabled else bar
            print(f"  │ {pname:>8s}: {bar_colored} {c_number(f'{pt*1000:.2f}ms')} ({pct:.1f}%)")
        print(f"  │ {'total':>8s}: {c_bold(f'{total*1000:.2f}ms')}")
        print(c_dim("  └─────────────────────────────────────────────┘"))

    # ═══ 元命令 ═══

    def _handle_meta(self, cmd):
        parts = cmd.split(); name = parts[0].lower()
        if name == '.quit': print(c_dim('Bye')); raise SystemExit
        if name == '.help': self._print_help(); return
        if name == '.tables': self._cmd_tables(); return
        if name == '.schema': self._cmd_schema(parts); return
        if name == '.analyze': self._cmd_analyze(parts); return
        if name == '.stats': self._cmd_stats(parts); return
        if name == '.timer': self._show_timer = self._toggle(parts, self._show_timer, 'Timer'); return
        if name == '.color':
            on = self._parse_toggle(parts, C.is_enabled())
            Colors.enable() if on else Colors.disable()
            print(f"Color {c_ok('on') if on else 'off'}."); return
        if name == '.profile': self._show_profile = self._toggle(parts, self._show_profile, 'Profile'); return
        if name == '.estimate': self._show_estimate = self._toggle(parts, self._show_estimate, 'Estimate'); return
        if name == '.history': self._cmd_history(parts); return
        if name == '.history_clear': self._history.clear(); print(c_ok("History cleared.")); return
        if name == '.benchmark': self._cmd_benchmark(parts); return
        if name == '.memory': self._cmd_memory(); return
        if name == '.import': self._cmd_import(parts); return
        if name == '.export': self._cmd_export(parts); return
        print(f"{c_error('Unknown command')}: {name}. Type .help")

    def _cmd_tables(self):
        tables = self._engine.get_table_names()
        if not tables: print(c_dim("No tables.")); return
        rows = [(t, str(self._engine.get_table_row_count(t))) for t in tables]
        self._draw(['Table', 'Rows'], rows)

    def _cmd_schema(self, parts):
        if len(parts) < 2: print("Usage: .schema <table>"); return
        try: schema = self._engine.get_table_schema(parts[1])
        except Z1Error as e: print(f"{c_error('Error')}: {e.message}"); return
        rows = [(c.name, c.dtype.name + (f'({c.max_length})' if c.max_length else ''),
                 'YES' if c.nullable else 'NO') for c in schema.columns]
        self._draw(['Column', 'Type', 'Nullable'], rows)

    def _cmd_analyze(self, parts):
        if len(parts) < 2: print("Usage: .analyze <table>"); return
        try:
            stats = self._engine.analyze_table(parts[1])
            print(c_ok(f"Analyzed {parts[1]}: {stats.row_count} rows"))
        except Exception as e: print(f"{c_error('Error')}: {e}")

    def _cmd_stats(self, parts):
        if len(parts) < 2: print("Usage: .stats <table>"); return
        stats = self._engine.get_table_stats(parts[1])
        if stats is None: print(c_dim(f"No stats. Run .analyze {parts[1]} first.")); return
        rows = [(cn, str(cs.ndv), str(cs.null_count),
                 str(cs.min_val) if cs.min_val is not None else 'NULL',
                 str(cs.max_val) if cs.max_val is not None else 'NULL')
                for cn, cs in stats.column_stats.items()]
        self._draw(['Column', 'NDV', 'Nulls', 'Min', 'Max'], rows)

    def _cmd_history(self, parts):
        if not self._history.entries: print(c_dim("No history.")); return
        n = int(parts[1]) if len(parts) > 1 else 20
        entries = self._history.entries[-n:]
        print(c_header(f"  Last {len(entries)} queries:"))
        for i, e in enumerate(entries, 1):
            print(f"  {c_dim(f'{i:3d}.')} {e['sql'][:60]} "
                  f"{c_dim(f'→ {e["rows"]} rows, {e["timing"]*1000:.1f}ms')}")

    def _cmd_benchmark(self, parts):
        if len(parts) < 2: print("Usage: .benchmark <iterations> <sql>"); return
        try: n = int(parts[1])
        except ValueError: print(c_error("First argument must be iteration count.")); return
        sql = ' '.join(parts[2:])
        if not sql: print(c_error("No SQL provided.")); return
        if not sql.endswith(';'): sql += ';'
        print(c_dim(f"Running {n} iterations..."))
        try:
            stats = run_benchmark(self._engine, sql, n)
            print(c_header("  ┌─ Benchmark Results ────────────────────┐"))
            for k in ('iterations', 'rows'):
                print(f"  │ {k:>12s}: {c_number(str(stats[k]))}")
            for k in ('avg', 'min', 'max', 'p50', 'p95', 'p99'):
                print(f"  │ {k:>12s}: {c_number(f'{stats[k]*1000:.3f}ms')}")
            print(f"  │ {'total':>12s}: {c_number(f'{stats["total"]:.3f}s')}")
            if stats['avg'] > 0:
                print(f'  │ {"throughput":>12s}: {c_bold(f"{1/stats['avg']:.0f} qps")}')
            print(c_header("  └─────────────────────────────────────────┘"))
        except Z1Error as e: print(f"{c_error('Error')}: {e.message}")

    def _cmd_memory(self):
        self._draw(['Table', 'Rows', 'Est. Memory'], memory_stats(self._engine))

    def _cmd_import(self, parts):
        if len(parts) < 3: print("Usage: .import <file> <table>"); return
        try:
            from utils.csv_io import read_csv
            from storage.types import DataType
            headers, rows = read_csv(parts[1])
            if not headers: print(c_dim("Empty CSV.")); return
            if not self._engine._catalog.table_exists(parts[2]):
                from catalog.catalog import ColumnSchema, TableSchema
                cols = [ColumnSchema(name=h.lower().replace(' ', '_'),
                                      dtype=DataType.VARCHAR, nullable=True) for h in headers]
                self._engine._catalog.create_table(TableSchema(name=parts[2], columns=cols))
            store = self._engine._catalog.get_store(parts[2])
            schema = self._engine._catalog.get_table(parts[2])
            count = 0
            for row in rows:
                while len(row) < len(schema.columns): row.append(None)
                row = row[:len(schema.columns)]
                converted = []
                for val, col in zip(row, schema.columns):
                    if val is None or val == '': converted.append(None)
                    elif col.dtype == DataType.INT:
                        try: converted.append(int(val))
                        except ValueError: converted.append(None)
                    elif col.dtype in (DataType.FLOAT, DataType.DOUBLE):
                        try: converted.append(float(val))
                        except ValueError: converted.append(None)
                    else: converted.append(str(val))
                store.append_row(converted); count += 1
            print(c_ok(f"Imported {count} rows into '{parts[2]}'."))
        except FileNotFoundError: print(c_error(f"File not found: {parts[1]}"))
        except Exception as e: print(f"{c_error('Error')}: {e}")

    def _cmd_export(self, parts):
        if len(parts) < 3: print("Usage: .export <file> <table>"); return
        try:
            from utils.csv_io import write_csv
            schema = self._engine._catalog.get_table(parts[2])
            store = self._engine._catalog.get_store(parts[2])
            count = write_csv(parts[1], schema.column_names, store.read_all_rows())
            print(c_ok(f"Exported {count} rows to '{parts[1]}'."))
        except Z1Error as e: print(f"{c_error('Error')}: {e.message}")
        except Exception as e: print(f"{c_error('Error')}: {e}")

    def _print_help(self):
        sections = [
            (c_header("  Data Commands:"), [
                (".tables", "List all tables"), (".schema <table>", "Show table schema"),
                (".import <file> <t>", "Import CSV"), (".export <file> <t>", "Export CSV")]),
            (c_header("  Analysis:"), [
                (".analyze <table>", "Compute statistics"), (".stats <table>", "Show statistics"),
                (".memory", "Show memory usage")]),
            (c_header("  Performance:"), [
                (".timer on|off", "Toggle timing"), (".profile on|off", "Toggle profiling"),
                (".estimate on|off", "Toggle estimation"), (".benchmark <n> <sql>", "Benchmark"),
                (".history [n]", "Show history"), (".history_clear", "Clear history")]),
            (c_header("  Display:"), [(".color on|off", "Toggle color")]),
            (c_header("  System:"), [(".help", "This message"), (".quit", "Exit")])]
        for header, cmds in sections:
            print(header)
            for cmd, desc in cmds:
                print(f"    {cmd:<25s} {c_dim(desc)}")

    # ═══ 辅助 ═══

    @staticmethod
    def _parse_toggle(parts, current):
        if len(parts) >= 2: return parts[1].lower() in ('on', '1', 'true')
        return not current

    def _toggle(self, parts, current, name):
        val = self._parse_toggle(parts, current)
        print(f"{name} {c_ok('on') if val else 'off'}."); return val

    @staticmethod
    def _has_real_content(stmt):
        i = 0
        while i < len(stmt):
            ch = stmt[i]
            if ch.isspace() or ch == ';': i += 1; continue
            if ch == '-' and i + 1 < len(stmt) and stmt[i + 1] == '-':
                while i < len(stmt) and stmt[i] != '\n': i += 1
                continue
            if ch == '/' and i + 1 < len(stmt) and stmt[i + 1] == '*':
                i += 2; d = 1
                while i < len(stmt) and d > 0:
                    if stmt[i] == '/' and i + 1 < len(stmt) and stmt[i + 1] == '*': d += 1; i += 1
                    elif stmt[i] == '*' and i + 1 < len(stmt) and stmt[i + 1] == '/': d -= 1; i += 1
                    i += 1
                continue
            return True
        return False

    def _split_statements(self, sql):
        stmts = []; cur = []; in_str = False; in_lc = False; in_bc = False; bcd = 0; i = 0
        while i < len(sql):
            ch = sql[i]
            if not in_str and not in_lc and not in_bc:
                if ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
                    in_bc = True; bcd = 1; cur.append(ch); cur.append(sql[i + 1]); i += 2; continue
            if in_bc:
                if ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*': bcd += 1; cur.append(ch); cur.append(sql[i + 1]); i += 2; continue
                if ch == '*' and i + 1 < len(sql) and sql[i + 1] == '/':
                    bcd -= 1; cur.append(ch); cur.append(sql[i + 1]); i += 2
                    if bcd == 0: in_bc = False
                    continue
                cur.append(ch); i += 1; continue
            if not in_str and not in_lc and ch == '-' and i + 1 < len(sql) and sql[i + 1] == '-':
                in_lc = True; cur.append(ch); i += 1; continue
            if in_lc:
                if ch == '\n': in_lc = False
                cur.append(ch); i += 1; continue
            if in_str:
                cur.append(ch)
                if ch == "'" and i + 1 < len(sql) and sql[i + 1] == "'": cur.append(sql[i + 1]); i += 2; continue
                if ch == "'": in_str = False
                i += 1; continue
            if ch == "'": in_str = True; cur.append(ch); i += 1; continue
            if ch == ';': cur.append(ch); stmts.append(''.join(cur)); cur = []; i += 1; continue
            cur.append(ch); i += 1
        rem = ''.join(cur).strip()
        if rem: stmts.append(rem)
        return stmts

    def _is_complete(self, buf):
        s = buf.strip()
        if not s or not s.endswith(';'): return False
        depth = 0; in_str = False; in_lc = False; in_bc = False; bcd = 0; i = 0
        while i < len(s):
            ch = s[i]
            if not in_str and not in_lc and not in_bc:
                if ch == '-' and i + 1 < len(s) and s[i + 1] == '-': in_lc = True; i += 2; continue
                if ch == '/' and i + 1 < len(s) and s[i + 1] == '*': in_bc = True; bcd = 1; i += 2; continue
            if in_bc:
                if ch == '/' and i + 1 < len(s) and s[i + 1] == '*': bcd += 1; i += 2; continue
                if ch == '*' and i + 1 < len(s) and s[i + 1] == '/': bcd -= 1; i += 2
                else: i += 1
                if bcd == 0: in_bc = False
                continue
            if in_lc:
                if ch == '\n': in_lc = False
                i += 1; continue
            if in_str:
                if ch == "'" and i + 1 < len(s) and s[i + 1] == "'": i += 2; continue
                if ch == "'": in_str = False
            else:
                if ch == "'": in_str = True
                elif ch == '(': depth += 1
                elif ch == ')': depth -= 1
            i += 1
        return depth == 0 and not in_str and not in_bc

    @staticmethod
    def _draw(headers, rows):
        from utils.table_renderer import render_table
        str_rows = [[str(v) for v in r] for r in rows]
        for line in render_table(headers, str_rows, header_fmt=c_header):
            print(line)
