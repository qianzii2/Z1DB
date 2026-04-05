from __future__ import annotations
"""Z1DB Interactive REPL — enhanced with color, profiling, benchmarks."""
import sys
import time
from typing import Any, Dict, List, Optional
from utils.errors import Z1Error


# ═══════════════════════════════════════════════════════════════
# ANSI Color (manually toggle with .color on/off)
# ═══════════════════════════════════════════════════════════════

class _Colors:
    """ANSI color codes. Disabled by default."""
    _enabled = False

    RESET = ''; BOLD = ''; DIM = ''
    RED = ''; GREEN = ''; YELLOW = ''; BLUE = ''; CYAN = ''; MAGENTA = ''
    BG_RED = ''; BG_GREEN = ''

    @classmethod
    def enable(cls) -> None:
        cls._enabled = True
        cls.RESET = '\033[0m'; cls.BOLD = '\033[1m'; cls.DIM = '\033[2m'
        cls.RED = '\033[31m'; cls.GREEN = '\033[32m'; cls.YELLOW = '\033[33m'
        cls.BLUE = '\033[34m'; cls.CYAN = '\033[36m'; cls.MAGENTA = '\033[35m'
        cls.BG_RED = '\033[41m'; cls.BG_GREEN = '\033[42m'

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False
        cls.RESET = cls.BOLD = cls.DIM = ''
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.CYAN = cls.MAGENTA = ''
        cls.BG_RED = cls.BG_GREEN = ''

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

C = _Colors


def _c_keyword(s: str) -> str:
    return f"{C.BLUE}{C.BOLD}{s}{C.RESET}" if C._enabled else s

def _c_number(s: str) -> str:
    return f"{C.CYAN}{s}{C.RESET}" if C._enabled else s

def _c_string(s: str) -> str:
    return f"{C.GREEN}{s}{C.RESET}" if C._enabled else s

def _c_error(s: str) -> str:
    return f"{C.RED}{C.BOLD}{s}{C.RESET}" if C._enabled else s

def _c_ok(s: str) -> str:
    return f"{C.GREEN}{s}{C.RESET}" if C._enabled else s

def _c_dim(s: str) -> str:
    return f"{C.DIM}{s}{C.RESET}" if C._enabled else s

def _c_bold(s: str) -> str:
    return f"{C.BOLD}{s}{C.RESET}" if C._enabled else s

def _c_header(s: str) -> str:
    return f"{C.YELLOW}{C.BOLD}{s}{C.RESET}" if C._enabled else s


# ═══════════════════════════════════════════════════════════════
# Enhanced Display
# ═══════════════════════════════════════════════════════════════

def _format_value_color(val: Any, dtype: Any) -> str:
    """Format value with optional color."""
    from utils.display import format_value
    s = format_value(val, dtype)
    if not C._enabled:
        return s
    if s == 'NULL':
        return _c_dim('NULL')
    name = dtype.name if hasattr(dtype, 'name') else ''
    if name == 'BOOLEAN':
        return _c_keyword(s)
    if name in ('INT', 'BIGINT', 'FLOAT', 'DOUBLE'):
        return _c_number(s)
    if name in ('VARCHAR', 'TEXT'):
        return _c_string(s)
    if name in ('DATE', 'TIMESTAMP'):
        return _c_number(s)
    return s


def _print_result_enhanced(result: Any, show_timer: bool = True) -> None:
    """Pretty-print with optional color support."""
    timing = getattr(result, 'timing', 0.0) if show_timer else 0.0

    if getattr(result, 'message', '') and not getattr(result, 'columns', []):
        msg = result.message
        if msg == 'OK':
            msg = _c_ok('OK')
        if timing > 0:
            print(f"{msg} {_c_dim(f'({timing:.3f} sec)')}")
        else:
            print(msg)
        return

    if getattr(result, 'affected_rows', 0) > 0 and not getattr(result, 'columns', []):
        n = result.affected_rows
        word = 'row' if n == 1 else 'rows'
        msg = f"{result.message or f'Affected {n} {word}'}"
        if timing > 0:
            print(f"{msg} {_c_dim(f'({timing:.3f} sec)')}")
        else:
            print(msg)
        return

    columns = getattr(result, 'columns', [])
    if columns:
        col_types = getattr(result, 'column_types', [None] * len(columns))
        rows = getattr(result, 'rows', [])

        # Build display strings (without color for width calculation)
        from utils.display import format_value
        raw_rows: list = []
        color_rows: list = []
        for row in rows:
            raw_row = []
            color_row = []
            for ci, val in enumerate(row):
                dt = col_types[ci] if ci < len(col_types) else None
                raw = format_value(val, dt) if dt else (str(val) if val is not None else 'NULL')
                colored = _format_value_color(val, dt) if dt else (str(val) if val is not None else _c_dim('NULL'))
                raw_row.append(raw)
                color_row.append(colored)
            raw_rows.append(raw_row)
            color_rows.append(color_row)

        # Column widths (based on raw text, not ANSI codes)
        widths = [len(c) for c in columns]
        for sr in raw_rows:
            for ci, s in enumerate(sr):
                if ci < len(widths):
                    widths[ci] = max(widths[ci], len(s))

        def line(l: str, m: str, r: str) -> str:
            return l + m.join('─' * (w + 2) for w in widths) + r

        # Header with color
        print(line('┌', '┬', '┐'))
        header_parts = []
        for i in range(len(columns)):
            col_name = _c_header(columns[i]) if C._enabled else columns[i]
            padding = widths[i] - len(columns[i])
            header_parts.append(f' {col_name}{" " * padding} ')
        print('│' + '│'.join(header_parts) + '│')
        print(line('├', '┼', '┤'))

        # Data rows
        for ri, (raw_row, color_row) in enumerate(zip(raw_rows, color_rows)):
            parts = []
            for ci in range(len(columns)):
                if ci < len(raw_row):
                    raw_val = raw_row[ci]
                    color_val = color_row[ci]
                    padding = widths[ci] - len(raw_val)
                    parts.append(f' {color_val}{" " * padding} ')
                else:
                    parts.append(' ' * (widths[ci] + 2))
            print('│' + '│'.join(parts) + '│')
        print(line('└', '┴', '┘'))

        n = len(rows)
        word = 'row' if n == 1 else 'rows'
        if timing > 0:
            print(f"{_c_number(str(n))} {word} {_c_dim(f'({timing:.3f} sec)')}")
        else:
            print(f"{n} {word}")
    else:
        msg = getattr(result, 'message', '') or 'OK'
        if timing > 0:
            print(f"{msg} {_c_dim(f'({timing:.3f} sec)')}")
        else:
            print(msg)


# ═══════════════════════════════════════════════════════════════
# Query History
# ═══════════════════════════════════════════════════════════════

class _QueryHistory:
    """Fixed-size ring buffer for query history."""

    def __init__(self, max_size: int = 50) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._max = max_size

    def add(self, sql: str, timing: float = 0.0, rows: int = 0) -> None:
        entry = {
            'sql': sql.strip()[:200],
            'timing': timing,
            'rows': rows,
            'ts': time.time(),
        }
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries.pop(0)

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return self._entries

    def clear(self) -> None:
        self._entries.clear()


# ═══════════════════════════════════════════════════════════════
# Profiler (parse / optimize / execute breakdown)
# ═══════════════════════════════════════════════════════════════

class _Profiler:
    """Measures each stage of query execution."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def profile(self, sql: str) -> Dict[str, float]:
        """Execute with per-stage timing."""
        from parser.lexer import Lexer
        from parser.parser import Parser
        from parser.resolver import Resolver
        from parser.validator import Validator

        stages: Dict[str, float] = {}

        # 1. Lex + Parse
        t0 = time.perf_counter()
        tokens = Lexer(sql).tokenize()
        ast = Parser(tokens).parse()
        stages['parse'] = time.perf_counter() - t0

        # 2. Resolve + Validate
        t0 = time.perf_counter()
        ast = Resolver().resolve(ast, self._engine._catalog)
        ast = Validator().validate(ast, self._engine._catalog)
        stages['resolve'] = time.perf_counter() - t0

        # 3. Optimize
        t0 = time.perf_counter()
        ast = self._engine._optimize_ast(ast)
        stages['optimize'] = time.perf_counter() - t0

        # 4. Execute
        t0 = time.perf_counter()
        from parser.ast import SelectStmt
        if self._engine._integrated and isinstance(ast, SelectStmt):
            result = self._engine._integrated.execute(ast, self._engine._catalog)
        else:
            result = self._engine._planner.execute(ast, self._engine._catalog)
        stages['execute'] = time.perf_counter() - t0

        stages['total'] = sum(stages.values())
        stages['rows'] = result.row_count
        stages['result'] = result
        return stages


# ═══════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════

def _run_benchmark(engine: Any, sql: str, iterations: int) -> Dict[str, float]:
    """Run SQL N times, compute avg/min/max/P50/P99."""
    timings: List[float] = []
    result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = engine.execute(sql)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

    timings.sort()
    n = len(timings)
    return {
        'iterations': n,
        'avg': sum(timings) / n,
        'min': timings[0],
        'max': timings[-1],
        'p50': timings[n // 2],
        'p95': timings[int(n * 0.95)] if n >= 20 else timings[-1],
        'p99': timings[int(n * 0.99)] if n >= 100 else timings[-1],
        'total': sum(timings),
        'rows': result.row_count if result else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Memory Stats
# ═══════════════════════════════════════════════════════════════

def _memory_stats(engine: Any) -> List[tuple]:
    """Estimate memory usage per table."""
    rows_data = []
    total_bytes = 0
    for tname in engine.get_table_names():
        store = engine._catalog.get_store(tname)
        row_count = store.row_count
        schema = engine._catalog.get_table(tname)
        # Estimate: each column chunk ≈ rows × avg_size
        est_bytes = 0
        for col in schema.columns:
            from storage.types import DTYPE_TO_ARRAY_CODE
            code = DTYPE_TO_ARRAY_CODE.get(col.dtype)
            if code:
                import struct
                item_size = struct.calcsize(code)
                est_bytes += row_count * item_size
            else:
                est_bytes += row_count * 20  # varchar estimate
            est_bytes += (row_count + 7) // 8  # null bitmap
        total_bytes += est_bytes
        rows_data.append((tname, str(row_count), _fmt_bytes(est_bytes)))
    rows_data.append(('─ TOTAL ─', '', _fmt_bytes(total_bytes)))
    return rows_data


def _fmt_bytes(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1024 * 1024: return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.1f} MB"


# ═══════════════════════════════════════════════════════════════
# Row Estimate
# ═══════════════════════════════════════════════════════════════

def _estimate_rows(engine: Any, sql: str) -> Optional[str]:
    """Quick row count estimate before execution."""
    try:
        from parser.lexer import Lexer
        from parser.parser import Parser
        from parser.ast import SelectStmt
        tokens = Lexer(sql).tokenize()
        ast = Parser(tokens).parse()
        if not isinstance(ast, SelectStmt):
            return None
        if ast.from_clause is None:
            return "~1 row"
        tname = ast.from_clause.table.name
        if not engine._catalog.table_exists(tname):
            return None
        total = engine._catalog.get_store(tname).row_count
        if ast.where:
            est = int(total * 0.33)  # heuristic
        else:
            est = total
        if ast.limit:
            from parser.ast import Literal
            if isinstance(ast.limit, Literal) and isinstance(ast.limit.value, int):
                est = min(est, ast.limit.value)
        return f"~{est:,} rows"
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# Main REPL
# ═══════════════════════════════════════════════════════════════

class REPL:
    def __init__(self, engine: object) -> None:
        self._engine = engine
        self._show_timer = True
        self._show_estimate = False  # .estimate on/off
        self._show_profile = False   # .profile on/off
        self._history = _QueryHistory()
        self._profiler = _Profiler(engine)

    def run(self) -> None:
        banner = (
            f"{_c_bold('Z1DB')} v1.2 — "
            f"{_c_dim('Pure Python OLAP Engine')}\n"
            f"{_c_dim('君 问 归 期 未 有 期 ， 曲 江 岸 上 月 华 盈')}\n"
            f"{_c_dim('Type .help for commands, .quit to exit')}"
        )
        print(banner)
        buffer = ''
        while True:
            prompt = f"{_c_bold('z1db>')} " if not buffer else f"{_c_dim('  ...>')} "
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print(f'\n{_c_dim("Bye")}')
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
        for stmt in self._split_statements(sql):
            stmt = stmt.strip()
            if not stmt or not self._has_real_content(stmt):
                continue
            if not stmt.endswith(';'):
                stmt += ';'
            self._execute_one(stmt)

    def _execute_one(self, sql: str) -> None:
        """Execute single statement with optional profiling + estimation."""
        # Row estimate
        if self._show_estimate:
            est = _estimate_rows(self._engine, sql)
            if est:
                print(_c_dim(f"  ⏳ Estimated: {est}"))

        try:
            if self._show_profile:
                stages = self._profiler.profile(sql)
                result = stages['result']
                result.timing = stages['total']
                _print_result_enhanced(result, self._show_timer)
                # Print profile breakdown
                total = stages['total']
                bar_width = 40
                parts = [
                    ('parse', stages['parse'], C.BLUE),
                    ('resolve', stages['resolve'], C.CYAN),
                    ('optimize', stages['optimize'], C.YELLOW),
                    ('execute', stages['execute'], C.GREEN),
                ]
                print(_c_dim("  ┌─ Profile ─────────────────────────────────┐"))
                for pname, pt, pcolor in parts:
                    pct = pt / total * 100 if total > 0 else 0
                    filled = int(pct / 100 * bar_width)
                    bar = '█' * filled + '░' * (bar_width - filled)
                    if C._enabled:
                        bar_colored = f"{pcolor}{bar}{C.RESET}"
                    else:
                        bar_colored = bar
                    ms_str = f'{pt * 1000:.2f}ms'
                    print(f"  │ {pname:>8s}: {bar_colored} {_c_number(ms_str)} ({pct:.1f}%)")
                total_str = f'{total * 1000:.2f}ms'
                print(f"  │ {'total':>8s}: {_c_bold(total_str)}")
                print(_c_dim("  └───────────────────────────────────────────┘"))
                self._history.add(sql, stages['total'], stages.get('rows', 0))
            else:
                result = self._engine.execute(sql)  # type: ignore
                if not self._show_timer:
                    result.timing = 0.0
                _print_result_enhanced(result, self._show_timer)
                self._history.add(sql, result.timing, result.row_count)
        except Z1Error as e:
            print(f"{_c_error('Error')}: {e.message}")

    # ═══ Meta commands ═══════════════════════════════════════
    def _handle_meta(self, cmd: str) -> None:
        parts = cmd.split()
        name = parts[0].lower()

        if name == '.quit':
            print(_c_dim('Bye'))
            raise SystemExit

        if name == '.help':
            self._print_help()
            return

        if name == '.tables':
            tables = self._engine.get_table_names()  # type: ignore
            if not tables:
                print(_c_dim("No tables."))
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
                print(f"{_c_error('Error')}: {e.message}")
                return
            rows = []
            for c in schema.columns:
                ts = c.dtype.name + (f'({c.max_length})' if c.max_length else '')
                rows.append((c.name, ts, 'YES' if c.nullable else 'NO'))
            self._draw(['Column', 'Type', 'Nullable'], rows)
            return

        if name == '.analyze':
            if len(parts) < 2:
                print("Usage: .analyze <table>")
                return
            try:
                stats = self._engine.analyze_table(parts[1])  # type: ignore
                print(_c_ok(f"Analyzed {parts[1]}: {stats.row_count} rows"))
            except Exception as e:
                print(f"{_c_error('Error')}: {e}")
            return

        if name == '.stats':
            if len(parts) < 2:
                print("Usage: .stats <table>")
                return
            try:
                stats = self._engine.get_table_stats(parts[1])  # type: ignore
                if stats is None:
                    print(_c_dim(f"No stats. Run .analyze {parts[1]} first."))
                    return
                rows = [(cn, str(cs.ndv), str(cs.null_count),
                         str(cs.min_val) if cs.min_val is not None else 'NULL',
                         str(cs.max_val) if cs.max_val is not None else 'NULL')
                        for cn, cs in stats.column_stats.items()]
                self._draw(['Column', 'NDV', 'Nulls', 'Min', 'Max'], rows)
            except Exception as e:
                print(f"{_c_error('Error')}: {e}")
            return

        # ── Feature toggles ──
        if name == '.timer':
            self._show_timer = self._toggle(parts, self._show_timer, 'Timer')
            return

        if name == '.color':
            on = self._parse_toggle(parts, C.is_enabled())
            if on:
                C.enable()
            else:
                C.disable()
            print(f"Color {_c_ok('on') if on else 'off'}.")
            return

        if name == '.profile':
            self._show_profile = self._toggle(parts, self._show_profile, 'Profile')
            return

        if name == '.estimate':
            self._show_estimate = self._toggle(parts, self._show_estimate, 'Estimate')
            return

        # ── History ──
        if name == '.history':
            if not self._history.entries:
                print(_c_dim("No history."))
                return
            n = int(parts[1]) if len(parts) > 1 else 20
            entries = self._history.entries[-n:]
            print(_c_header(f"  Last {len(entries)} queries:"))
            for i, e in enumerate(entries, 1):
                sql_preview = e['sql'][:60]
                t = e['timing']
                r = e['rows']
                print(f"  {_c_dim(f'{i:3d}.')} {sql_preview}"
                      f" {_c_dim(f'→ {r} rows, {t*1000:.1f}ms')}")
            return

        if name == '.history_clear':
            self._history.clear()
            print(_c_ok("History cleared."))
            return

        # ── Benchmark ──
        if name == '.benchmark':
            if len(parts) < 2:
                print("Usage: .benchmark <iterations> <sql>")
                print("  Example: .benchmark 100 SELECT COUNT(*) FROM t;")
                return
            try:
                n = int(parts[1])
            except ValueError:
                print(_c_error("First argument must be iteration count."))
                return
            sql = ' '.join(parts[2:])
            if not sql:
                print(_c_error("No SQL provided."))
                return
            if not sql.endswith(';'):
                sql += ';'
            print(_c_dim(f"Running {n} iterations..."))
            try:
                stats = _run_benchmark(self._engine, sql, n)
                iters = stats['iterations']
                rows = stats['rows']
                avg_ms = stats['avg'] * 1000
                min_ms = stats['min'] * 1000
                max_ms = stats['max'] * 1000
                p50_ms = stats['p50'] * 1000
                p95_ms = stats['p95'] * 1000
                p99_ms = stats['p99'] * 1000
                total_s = stats['total']
                print(_c_header("  ┌─ Benchmark Results ────────────────────┐"))
                print(f"  │ Iterations: {_c_number(str(iters))}")
                print(f"  │ Rows/exec:  {_c_number(str(rows))}")
                print(f"  │ Avg:        {_c_number(f'{avg_ms:.3f}ms')}")
                print(f"  │ Min:        {_c_number(f'{min_ms:.3f}ms')}")
                print(f"  │ Max:        {_c_number(f'{max_ms:.3f}ms')}")
                print(f"  │ P50:        {_c_number(f'{p50_ms:.3f}ms')}")
                print(f"  │ P95:        {_c_number(f'{p95_ms:.3f}ms')}")
                print(f"  │ P99:        {_c_number(f'{p99_ms:.3f}ms')}")
                print(f"  │ Total:      {_c_number(f'{total_s:.3f}s')}")
                if stats['avg'] > 0:
                    qps = 1.0 / stats['avg']
                    print(f"  │ Throughput: {_c_bold(f'{qps:.0f} queries/sec')}")
                print(_c_header("  └─────────────────────────────────────────┘"))
            except Z1Error as e:
                print(f"{_c_error('Error')}: {e.message}")
            return

        # ── Memory ──
        if name == '.memory':
            rows = _memory_stats(self._engine)
            self._draw(['Table', 'Rows', 'Est. Memory'], rows)
            return

        # ── CSV ──
        if name == '.import':
            if len(parts) < 3:
                print("Usage: .import <file> <table>")
                return
            self._import_csv(parts[1], parts[2])
            return

        if name == '.export':
            if len(parts) < 3:
                print("Usage: .export <file> <table>")
                return
            self._export_csv(parts[1], parts[2])
            return

        print(f"{_c_error('Unknown command')}: {name}. Type .help")

    def _print_help(self) -> None:
        sections = [
            (_c_header("  Data Commands:"), [
                (".tables", "List all tables"),
                (".schema <table>", "Show table schema"),
                (".import <file> <t>", "Import CSV into table"),
                (".export <file> <t>", "Export table to CSV"),
            ]),
            (_c_header("  Analysis:"), [
                (".analyze <table>", "Compute column statistics"),
                (".stats <table>", "Show computed statistics"),
                (".memory", "Show memory usage per table"),
            ]),
            (_c_header("  Performance:"), [
                (".timer on|off", "Toggle query timing display"),
                (".profile on|off", "Toggle per-stage profiling"),
                (".estimate on|off", "Toggle row count estimation"),
                (".benchmark <n> <sql>", "Run SQL n times, show P50/P99"),
                (".history [n]", "Show last n queries (default 20)"),
                (".history_clear", "Clear query history"),
            ]),
            (_c_header("  Display:"), [
                (".color on|off", "Toggle ANSI color output"),
            ]),
            (_c_header("  System:"), [
                (".help", "Show this message"),
                (".quit", "Exit Z1DB"),
            ]),
        ]
        for header, cmds in sections:
            print(header)
            for cmd, desc in cmds:
                print(f"    {cmd:<25s} {_c_dim(desc)}")

    # ═══ Toggle helpers ═══════════════════════════════════════
    @staticmethod
    def _parse_toggle(parts: list, current: bool) -> bool:
        if len(parts) >= 2:
            return parts[1].lower() in ('on', '1', 'true')
        return not current

    def _toggle(self, parts: list, current: bool, name: str) -> bool:
        val = self._parse_toggle(parts, current)
        label = _c_ok('on') if val else 'off'
        print(f"{name} {label}.")
        return val

    # ═══ Statement parsing ════════════════════════════════════
    @staticmethod
    def _has_real_content(stmt: str) -> bool:
        i = 0
        while i < len(stmt):
            ch = stmt[i]
            if ch.isspace() or ch == ';':
                i += 1; continue
            if ch == '-' and i + 1 < len(stmt) and stmt[i + 1] == '-':
                while i < len(stmt) and stmt[i] != '\n':
                    i += 1
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

    def _split_statements(self, sql: str) -> list:
        stmts: list = []; cur: list = []
        in_str = False; in_lc = False; in_bc = False; bcd = 0
        i = 0
        while i < len(sql):
            ch = sql[i]
            if not in_str and not in_lc and not in_bc:
                if ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
                    in_bc = True; bcd = 1; cur.append(ch); cur.append(sql[i + 1]); i += 2; continue
            if in_bc:
                if ch == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
                    bcd += 1; cur.append(ch); cur.append(sql[i + 1]); i += 2; continue
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
                if ch == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    cur.append(sql[i + 1]); i += 2; continue
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
                if ch == '-' and i + 1 < len(s) and s[i + 1] == '-':
                    in_lc = True; i += 2; continue
                if ch == '/' and i + 1 < len(s) and s[i + 1] == '*':
                    in_bc = True; bcd = 1; i += 2; continue
            if in_bc:
                if ch == '/' and i + 1 < len(s) and s[i + 1] == '*':
                    bcd += 1; i += 2; continue
                if ch == '*' and i + 1 < len(s) and s[i + 1] == '/':
                    bcd -= 1; i += 2
                    if bcd == 0: in_bc = False
                    continue
                i += 1; continue
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

    # ═══ CSV ══════════════════════════════════════════════════
    def _import_csv(self, filepath: str, table: str) -> None:
        try:
            from utils.csv_io import read_csv
            from storage.types import DataType
            headers, rows = read_csv(filepath)
            if not headers:
                print(_c_dim("Empty CSV."))
                return
            if not self._engine._catalog.table_exists(table):  # type: ignore
                from catalog.catalog import ColumnSchema, TableSchema
                cols = [ColumnSchema(name=h.lower().replace(' ', '_'),
                                     dtype=DataType.VARCHAR, nullable=True) for h in headers]
                self._engine._catalog.create_table(TableSchema(name=table, columns=cols))  # type: ignore
            store = self._engine._catalog.get_store(table)  # type: ignore
            schema = self._engine._catalog.get_table(table)  # type: ignore
            count = 0
            for row in rows:
                while len(row) < len(schema.columns): row.append(None)
                row = row[:len(schema.columns)]
                converted = []
                for val, col in zip(row, schema.columns):
                    if val is None or val == '':
                        converted.append(None)
                    elif col.dtype == DataType.INT:
                        try: converted.append(int(val))
                        except ValueError: converted.append(None)
                    elif col.dtype in (DataType.FLOAT, DataType.DOUBLE):
                        try: converted.append(float(val))
                        except ValueError: converted.append(None)
                    else:
                        converted.append(str(val))
                store.append_row(converted)
                count += 1
            print(_c_ok(f"Imported {count} rows into '{table}'."))
        except FileNotFoundError:
            print(_c_error(f"File not found: {filepath}"))
        except Exception as e:
            print(f"{_c_error('Error')}: {e}")

    def _export_csv(self, filepath: str, table: str) -> None:
        try:
            from utils.csv_io import write_csv
            schema = self._engine._catalog.get_table(table)  # type: ignore
            store = self._engine._catalog.get_store(table)  # type: ignore
            count = write_csv(filepath, schema.column_names, store.read_all_rows())
            print(_c_ok(f"Exported {count} rows to '{filepath}'."))
        except Z1Error as e:
            print(f"{_c_error('Error')}: {e.message}")
        except Exception as e:
            print(f"{_c_error('Error')}: {e}")

    # ═══ Table drawing ════════════════════════════════════════
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
        hdr_parts = []
        for i in range(nc):
            hdr_parts.append(f' {_c_header(headers[i])}{" " * (w[i] - len(headers[i]))} ')
        print('│' + '│'.join(hdr_parts) + '│')
        print(ln('├', '┼', '┤'))
        for r in rows:
            print('│' + '│'.join(f' {str(r[i]):<{w[i]}} ' for i in range(nc)) + '│')
        print(ln('└', '┴', '┘'))
