from __future__ import annotations
"""Z1DB REPL — SQL 交互式终端。"""
from typing import Any
from server.colors import (
    C, c_bold, c_dim, c_error, c_number, print_result_enhanced)
from server.meta_commands import MetaCommandHandler
from server.profiler import Profiler, estimate_rows
from server.query_history import QueryHistory
from server.sql_splitter import split_statements, is_complete, has_real_content
from utils.errors import Z1Error


class REPL:
    def __init__(self, engine: object) -> None:
        self._engine = engine
        self._show_timer = True
        self._show_estimate = False
        self._show_profile = False
        self._history = QueryHistory()
        self._profiler = Profiler(engine)
        self._meta = MetaCommandHandler(engine, self)

    def run(self) -> None:
        print(f"{c_bold('Z1DB')} v1.2 — {c_dim('纯 Python OLAP 引擎')}")
        print(c_dim('输入 .help 查看命令，.quit 退出'))
        buffer = ''
        while True:
            prompt = (f"{c_bold('z1db>')} " if not buffer
                      else f"{c_dim('  ...>')} ")
            try: line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print(f'\n{c_dim("再见")}'); break
            stripped = line.strip()
            if not stripped and not buffer:
                continue
            if not buffer and stripped.startswith('.'):
                try:
                    if self._meta.handle(stripped):
                        break  # .quit
                except SystemExit:
                    break
                continue
            buffer += line + '\n'
            if is_complete(buffer):
                sql = buffer.strip(); buffer = ''
                self._execute_statements(sql)

    def _execute_statements(self, sql: str) -> None:
        for stmt in split_statements(sql):
            stmt = stmt.strip()
            if not stmt or not has_real_content(stmt):
                continue
            if not stmt.endswith(';'):
                stmt += ';'
            self._execute_one(stmt)

    def _execute_one(self, sql: str) -> None:
        if self._show_estimate:
            est = estimate_rows(self._engine, sql)
            if est:
                print(c_dim(f"  ⏳ 估算: {est}"))
        try:
            if self._show_profile:
                stages = self._profiler.profile(sql)
                result = stages['result']
                result.timing = stages['total']
                print_result_enhanced(result, self._show_timer)
                self._print_profile(stages)
                self._history.add(
                    sql, stages['total'], stages.get('rows', 0))
            else:
                result = self._engine.execute(sql)
                if not self._show_timer:
                    result.timing = 0.0
                print_result_enhanced(result, self._show_timer)
                self._history.add(
                    sql, result.timing, result.row_count)
        except Z1Error as e:
            print(f"{c_error('错误')}: {e.message}")

    def _print_profile(self, stages: dict) -> None:
        total = stages['total']; bar_width = 40
        parts = [
            ('解析', stages['parse'], C.BLUE),
            ('解析', stages['resolve'], C.CYAN),
            ('优化', stages['optimize'], C.YELLOW),
            ('执行', stages['execute'], C.GREEN)]
        print(c_dim("  ┌─ 性能分析 ──────────────────────────────┐"))
        for pname, pt, pcolor in parts:
            pct = pt / total * 100 if total > 0 else 0
            filled = int(pct / 100 * bar_width)
            bar = '█' * filled + '░' * (bar_width - filled)
            bar_colored = (f"{pcolor}{bar}{C.RESET}"
                           if C._enabled else bar)
            print(f"  │ {pname:>6s}: {bar_colored} "
                  f"{c_number(f'{pt*1000:.2f}ms')} ({pct:.1f}%)")
        print(f"  │ {'合计':>6s}: {c_bold(f'{total*1000:.2f}ms')}")
        print(c_dim("  └──────────────────────────────────────────┘"))
