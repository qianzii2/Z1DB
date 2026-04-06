from __future__ import annotations
"""REPL 元命令实现。以 . 开头的命令。"""
from typing import Any
from server.colors import (
    C, Colors, c_bold, c_dim, c_error, c_header,
    c_number, c_ok)
from server.profiler import run_benchmark, memory_stats
from utils.errors import Z1Error


class MetaCommandHandler:
    """处理 .tables, .schema, .analyze 等元命令。"""

    def __init__(self, engine: Any, repl: Any) -> None:
        self._engine = engine
        self._repl = repl

    def handle(self, cmd: str) -> bool:
        """处理元命令。返回 True 表示需要退出。"""
        parts = cmd.split()
        name = parts[0].lower()
        dispatch = {
            '.quit': self._quit,
            '.help': self._help,
            '.tables': self._tables,
            '.schema': self._schema,
            '.analyze': self._analyze,
            '.stats': self._stats,
            '.timer': self._timer,
            '.color': self._color,
            '.profile': self._profile,
            '.estimate': self._estimate,
            '.history': self._history,
            '.history_clear': self._history_clear,
            '.benchmark': self._benchmark,
            '.memory': self._memory,
            '.import': self._import_csv,
            '.export': self._export_csv,
            '.vacuum': self._vacuum,
        }
        fn = dispatch.get(name)
        if fn is None:
            print(f"{c_error('未知命令')}: {name}. 输入 .help")
            return False
        return fn(parts)

    def _quit(self, parts) -> bool:
        print(c_dim('再见')); return True

    def _help(self, parts) -> bool:
        sections = [
            (c_header("  数据命令:"), [
                (".tables", "列出所有表"),
                (".schema <table>", "显示表结构"),
                (".import <file> <t>", "导入 CSV"),
                (".export <file> <t>", "导出 CSV")]),
            (c_header("  分析:"), [
                (".analyze <table>", "计算统计信息"),
                (".stats <table>", "显示统计信息"),
                (".memory", "显示内存使用")]),
            (c_header("  性能:"), [
                (".timer on|off", "开关计时"),
                (".profile on|off", "开关性能分析"),
                (".estimate on|off", "开关行数估算"),
                (".benchmark <n> <sql>", "基准测试"),
                (".history [n]", "查看历史"),
                (".history_clear", "清空历史")]),
            (c_header("  显示:"), [
                (".color on|off", "开关颜色")]),
            (c_header("  系统:"), [
                (".help", "此信息"),
                (".quit", "退出")])]
        for header, cmds in sections:
            print(header)
            for cmd, desc in cmds:
                print(f"    {cmd:<25s} {c_dim(desc)}")
        return False

    def _tables(self, parts) -> bool:
        tables = self._engine.get_table_names()
        if not tables:
            print(c_dim("无表。")); return False
        rows = [(t, str(self._engine.get_table_row_count(t)))
                for t in tables]
        self._draw(['表名', '行数'], rows)
        return False

    def _schema(self, parts) -> bool:
        if len(parts) < 2:
            print("用法: .schema <表名>"); return False
        try:
            schema = self._engine.get_table_schema(parts[1])
        except Z1Error as e:
            print(f"{c_error('错误')}: {e.message}"); return False
        rows = [(c.name,
                 c.dtype.name + (f'({c.max_length})' if c.max_length else ''),
                 '是' if c.nullable else '否')
                for c in schema.columns]
        self._draw(['列名', '类型', '可空'], rows)
        return False

    def _analyze(self, parts) -> bool:
        if len(parts) < 2:
            print("用法: .analyze <表名>"); return False
        try:
            stats = self._engine.analyze_table(parts[1])
            print(c_ok(f"分析完成 {parts[1]}: {stats.row_count} 行"))
        except Exception as e:
            print(f"{c_error('错误')}: {e}")
        return False

    def _stats(self, parts) -> bool:
        if len(parts) < 2:
            print("用法: .stats <表名>"); return False
        stats = self._engine.get_table_stats(parts[1])
        if stats is None:
            print(c_dim(f"无统计信息。先执行 .analyze {parts[1]}"))
            return False
        rows = [(cn, str(cs.ndv), str(cs.null_count),
                 str(cs.min_val) if cs.min_val is not None else 'NULL',
                 str(cs.max_val) if cs.max_val is not None else 'NULL')
                for cn, cs in stats.column_stats.items()]
        self._draw(['列', 'NDV', '空值数', '最小值', '最大值'], rows)
        return False

    def _timer(self, parts) -> bool:
        self._repl._show_timer = self._toggle(
            parts, self._repl._show_timer, '计时')
        return False

    def _color(self, parts) -> bool:
        on = self._parse_toggle(parts, C.is_enabled())
        Colors.enable() if on else Colors.disable()
        print(f"颜色 {c_ok('开') if on else '关'}。")
        return False

    def _profile(self, parts) -> bool:
        self._repl._show_profile = self._toggle(
            parts, self._repl._show_profile, '性能分析')
        return False

    def _estimate(self, parts) -> bool:
        self._repl._show_estimate = self._toggle(
            parts, self._repl._show_estimate, '估算')
        return False

    def _history(self, parts) -> bool:
        if not self._repl._history.entries:
            print(c_dim("无历史。")); return False
        n = int(parts[1]) if len(parts) > 1 else 20
        entries = self._repl._history.entries[-n:]
        print(c_header(f"  最近 {len(entries)} 条查询:"))
        for i, e in enumerate(entries, 1):
            print(f"  {c_dim(f'{i:3d}.')} {e['sql'][:60]} "
                  f"{c_dim(f'→ {e["rows"]} 行, {e["timing"]*1000:.1f}ms')}")
        return False

    def _history_clear(self, parts) -> bool:
        self._repl._history.clear()
        print(c_ok("历史已清空。"))
        return False

    def _benchmark(self, parts) -> bool:
        if len(parts) < 3:
            print("用法: .benchmark <次数> <SQL>"); return False
        try: n = int(parts[1])
        except ValueError:
            print(c_error("第一个参数必须是整数。")); return False
        sql = ' '.join(parts[2:])
        if not sql:
            print(c_error("未提供 SQL。")); return False
        if not sql.endswith(';'): sql += ';'
        print(c_dim(f"运行 {n} 次..."))
        try:
            stats = run_benchmark(self._engine, sql, n)
            print(c_header("  ┌─ 基准测试结果 ────────────────────┐"))
            for k in ('iterations', 'rows'):
                print(f"  │ {k:>12s}: {c_number(str(stats[k]))}")
            for k in ('avg', 'min', 'max', 'p50', 'p95', 'p99'):
                print(f"  │ {k:>12s}: {c_number(f'{stats[k]*1000:.3f}ms')}")
            print(f"  │ {'total':>12s}: {c_number(f'{stats["total"]:.3f}s')}")
            if stats['avg'] > 0:
                    print(f'  │ {"throughput":>12s}: '
                        f'{c_bold(f"{1/stats['avg']:.0f} qps")}')
            print(c_header("  └───────────────────────────────────┘"))
        except Z1Error as e:
            print(f"{c_error('错误')}: {e.message}")
        return False

    def _memory(self, parts) -> bool:
        self._draw(['表', '行数', '估算内存'],
                   memory_stats(self._engine))
        return False

    def _import_csv(self, parts) -> bool:
        if len(parts) < 3:
            print("用法: .import <文件> <表名>"); return False
        try:
            from utils.csv_io import read_csv
            from storage.types import DataType
            headers, rows = read_csv(parts[1])
            if not headers:
                print(c_dim("空 CSV。")); return False
            if not self._engine.table_exists(parts[2]):
                from catalog.catalog import ColumnSchema, TableSchema
                cols = [ColumnSchema(
                    name=h.lower().replace(' ', '_'),
                    dtype=DataType.VARCHAR, nullable=True)
                    for h in headers]
                self._engine.create_table_from_schema(
                    TableSchema(name=parts[2], columns=cols))
            store = self._engine.get_store(parts[2])
            schema = self._engine.get_table_schema(parts[2])
            count = 0
            for row in rows:
                while len(row) < len(schema.columns):
                    row.append(None)
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
                    else: converted.append(str(val))
                store.append_row(converted); count += 1
            print(c_ok(f"导入 {count} 行到 '{parts[2]}'。"))
        except FileNotFoundError:
            print(c_error(f"文件未找到: {parts[1]}"))
        except Exception as e:
            print(f"{c_error('错误')}: {e}")
        return False

    def _export_csv(self, parts) -> bool:
        if len(parts) < 3:
            print("用法: .export <文件> <表名>"); return False
        try:
            from utils.csv_io import write_csv
            schema = self._engine.get_table_schema(parts[2])
            store = self._engine.get_store(parts[2])
            count = write_csv(parts[1], schema.column_names,
                              store.read_all_rows())
            print(c_ok(f"导出 {count} 行到 '{parts[1]}'。"))
        except Z1Error as e:
            print(f"{c_error('错误')}: {e.message}")
        except Exception as e:
            print(f"{c_error('错误')}: {e}")
        return False

    # ═══ 辅助 ═══

    @staticmethod
    def _parse_toggle(parts, current):
        if len(parts) >= 2:
            return parts[1].lower() in ('on', '1', 'true')
        return not current

    def _toggle(self, parts, current, name):
        val = self._parse_toggle(parts, current)
        print(f"{name} {c_ok('开') if val else '关'}。")
        return val

    @staticmethod
    def _draw(headers, rows):
        from utils.table_renderer import render_table
        str_rows = [[str(v) for v in r] for r in rows]
        for line in render_table(headers, str_rows, header_fmt=c_header):
            print(line)

    def _vacuum(self, parts) -> bool:
        table = parts[1] if len(parts) > 1 else None
        sql = f"VACUUM {table};" if table else "VACUUM;"
        try:
            result = self._engine.execute(sql)
            print(c_ok(result.message))
        except Z1Error as e:
            print(f"{c_error('错误')}: {e.message}")
        return False