from __future__ import annotations
"""Z1DB 引擎 — WAL + 事务 + 缓存 + 后台合并 + LSM。
COPY 语句通过 Parser→CopyStmt→SimplePlanner 正式路径执行。"""
from typing import Any, Optional
import dataclasses
import sys

from catalog.catalog import Catalog, ColumnSchema, TableSchema
from catalog.statistics import TableStatistics
from executor.core.result import ExecutionResult
from executor.functions.registry import FunctionRegistry
from executor.memory_budget import MemoryBudget
from executor.optimizer import QueryOptimizer
from executor.result_cache import ResultCache
from executor.simple_planner import SimplePlanner
from parser.ast import (
    SelectStmt, ExplainStmt, SetOperationStmt,
    InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, AlterTableStmt,
    CopyStmt,
)
from parser.lexer import Lexer
from parser.parser import Parser
from parser.resolver import Resolver
from parser.validator import Validator
from storage.types import DataType
from utils.timer import Timer

try:
    from executor.query_coordinator import QueryCoordinator
    _HAS_COORDINATOR = True
except ImportError: _HAS_COORDINATOR = False

try:
    from storage.hybrid.merge_worker import MergeWorker
    _HAS_MERGE_WORKER = True
except ImportError: _HAS_MERGE_WORKER = False

_CTE_PREFIX = '__cte_'
_DML_TYPES = (InsertStmt, UpdateStmt, DeleteStmt)
_MUTATING_TYPES = (InsertStmt, UpdateStmt, DeleteStmt,
                   CreateTableStmt, DropTableStmt, AlterTableStmt,
                   CopyStmt)


class Engine:
    def __init__(self, data_dir: str = ':memory:') -> None:
        self._catalog = Catalog(data_dir)
        self._registry = FunctionRegistry()
        self._registry.register_defaults()
        self._budget = MemoryBudget()
        self._optimizer = QueryOptimizer()
        self._stats: dict[str, TableStatistics] = {}
        self._suppress_persist = False
        self._result_cache = ResultCache()
        self._table_versions: dict[str, int] = {}
        self._planner = SimplePlanner(self._registry, budget=self._budget)
        self._wal = None
        if data_dir != ':memory:':
            try:
                from storage.wal import WriteAheadLog
                self._wal = WriteAheadLog(data_dir); self._wal.open(); self._replay_wal()
            except Exception: self._wal = None
        self._txn = None
        try:
            from txn.manager import TransactionManager
            self._txn = TransactionManager()
        except ImportError: pass
        self._integrated = None
        try:
            from executor.integrated_planner import IntegratedPlanner
            self._integrated = IntegratedPlanner(self._registry, budget=self._budget)
        except ImportError: pass
        self._merge_worker = None
        if _HAS_MERGE_WORKER:
            try:
                self._merge_worker = MergeWorker(merge_fn=self._background_merge, interval=30.0, threshold=10000)
                self._merge_worker.start()
            except Exception: self._merge_worker = None

    def execute(self, sql: str) -> ExecutionResult:
        sql_stripped = sql.strip().rstrip(';').strip(); upper = sql_stripped.upper()
        if self._txn:
            if upper == 'BEGIN':
                txn_id = self._txn.begin(); return ExecutionResult(message=f'BEGIN (txn {txn_id})')
            if upper == 'COMMIT':
                self._txn.commit()
                if self._catalog.is_persistent: self._catalog.persist()
                return ExecutionResult(message='COMMIT')
            if upper == 'ROLLBACK':
                self._txn.rollback(self._catalog); return ExecutionResult(message='ROLLBACK')
        # R6：COPY 走正常 Parser 路径，不再在此拦截
        sql_hash = ResultCache.hash_sql(sql)
        cached = self._result_cache.get(sql_hash, self._table_versions)
        if cached is not None: return cached
        auto_txn_id = None
        with Timer() as t:
            tokens = Lexer(sql).tokenize(); ast = Parser(tokens).parse()
            cte_tables = self._materialize_ctes(ast)
            try:
                if cte_tables: ast = self._strip_ctes(ast)
                # CopyStmt 不需要 resolve/validate
                if not isinstance(ast, CopyStmt):
                    ast = Resolver().resolve(ast, self._catalog)
                    ast = Validator().validate(ast, self._catalog)
                    ast = self._optimize_ast(ast)
                    if _HAS_COORDINATOR and isinstance(ast, SelectStmt):
                        ast = self._optimize_subqueries(ast)
                is_dml = isinstance(ast, _DML_TYPES)
                if is_dml:
                    table = getattr(ast, 'table', None)
                    if table:
                        if self._txn:
                            auto_txn_id = self._txn.auto_begin()
                            self._txn.snapshot_table(table, self._catalog)
                        if self._wal:
                            self._wal.append(self._get_dml_type(ast), table, {'sql': sql.strip()})
                if self._integrated and isinstance(ast, SelectStmt):
                    result = self._integrated.execute(ast, self._catalog)
                else:
                    result = self._planner.execute(ast, self._catalog)
                if is_dml and auto_txn_id is not None and self._txn:
                    self._txn.commit(); auto_txn_id = None
            except Exception as e:
                if auto_txn_id is not None and self._txn:
                    self._txn.rollback(self._catalog); auto_txn_id = None
                raise
            finally:
                self._suppress_persist = True
                for tname in cte_tables:
                    try: self._catalog.drop_table(tname)
                    except Exception: pass
                self._suppress_persist = False
        result.timing = t.elapsed
        is_mutation = isinstance(ast, _MUTATING_TYPES)
        if self._catalog.is_persistent and not self._suppress_persist and is_mutation:
            self._catalog.persist()
            if self._wal:
                try: self._wal.checkpoint()
                except Exception: pass
        if is_mutation: self._invalidate_modified_tables(ast)
        elif isinstance(ast, SelectStmt):
            tables = self._extract_tables(ast)
            versions = {tbl: self._table_versions.get(tbl, 0) for tbl in tables}
            self._result_cache.put(sql_hash, result, versions)
        return result

    def _optimize_subqueries(self, ast):
        try:
            coordinator = QueryCoordinator(self._planner, self._catalog)
            return coordinator.optimize_subqueries(ast)
        except Exception: return ast

    def _background_merge(self):
        total = 0
        for tname in self._catalog.list_tables():
            if tname.startswith(_CTE_PREFIX): continue
            try:
                store = self._catalog.get_store(tname)
                if hasattr(store, '_deleted_global') and hasattr(store._deleted_global, '__len__') and len(store._deleted_global) > 0:
                    store._compact(); total += 1
                try:
                    from storage.lsm.lsm_store import LSMStore
                    if isinstance(store, LSMStore): store._compact(); total += 1
                except ImportError: pass
            except Exception: pass
        return total

    def _invalidate_modified_tables(self, ast):
        table = getattr(ast, 'table', None)
        if table:
            self._table_versions[table] = self._table_versions.get(table, 0) + 1
            self._result_cache.invalidate_table(table)

    def _extract_tables(self, ast):
        tables = set()
        if ast.from_clause and ast.from_clause.table:
            tables.add(ast.from_clause.table.name)
            for jc in ast.from_clause.joins:
                if jc.table: tables.add(jc.table.name)
        return tables

    @staticmethod
    def _get_dml_type(ast):
        if isinstance(ast, InsertStmt): return 'INSERT'
        if isinstance(ast, UpdateStmt): return 'UPDATE'
        if isinstance(ast, DeleteStmt): return 'DELETE'
        return 'UNKNOWN'

    def _replay_wal(self):
        if not self._wal: return
        try:
            entries = self._wal.recover(); replayed = failed = 0
            for entry in entries:
                if 'sql' in entry.data:
                    try:
                        s = entry.data['sql']; a = Parser(Lexer(s).tokenize()).parse()
                        a = Resolver().resolve(a, self._catalog)
                        a = Validator().validate(a, self._catalog)
                        self._planner.execute(a, self._catalog); replayed += 1
                    except Exception as e: failed += 1; print(f"[WAL] 回放失败 LSN={entry.lsn}: {e}", file=sys.stderr)
            if replayed > 0 or failed > 0: print(f"[WAL] 回放完成: {replayed} 成功, {failed} 失败", file=sys.stderr)
        except Exception as e: print(f"[WAL] 恢复异常: {e}", file=sys.stderr)

    def _materialize_ctes(self, ast):
        created = []
        if isinstance(ast, SelectStmt) and ast.ctes:
            for entry in ast.ctes:
                cn, cq = entry[0], entry[1]; ir = entry[2] if len(entry) > 2 else False; cc = entry[3] if len(entry) > 3 else None
                nested = self._materialize_ctes(cq); created.extend(nested)
                if ir: cr = self._execute_recursive_cte(cn, cq, cc)
                else:
                    try:
                        r = Resolver().resolve(cq, self._catalog); v = Validator().validate(r, self._catalog)
                        cr = self._planner.execute(self._optimize_ast(v), self._catalog)
                    except Exception: continue
                if cr.columns:
                    cols = list(cr.columns)
                    if cc and len(cc) == len(cols): cols = list(cc)
                    cs = [ColumnSchema(name=c, dtype=t, nullable=True) for c, t in zip(cols, cr.column_types)]
                    self._suppress_persist = True
                    try:
                        for nm in (f'{_CTE_PREFIX}{cn}', cn):
                            if not self._catalog.table_exists(nm):
                                self._catalog.create_table(TableSchema(name=nm, columns=cs))
                                st = self._catalog.get_store(nm)
                                for row in cr.rows: st.append_row(list(row))
                                created.append(nm)
                    finally: self._suppress_persist = False
        elif isinstance(ast, ExplainStmt): created.extend(self._materialize_ctes(ast.statement))
        elif isinstance(ast, SetOperationStmt):
            created.extend(self._materialize_ctes(ast.left)); created.extend(self._materialize_ctes(ast.right))
        return created

    def _execute_recursive_cte(self, name, query, columns=None):
        from executor.recursive_cte import RecursiveCTEExecutor
        if isinstance(query, SetOperationStmt) and query.op.upper() == 'UNION':
            return RecursiveCTEExecutor(self._planner, self._catalog).execute(name, query.left, query.right, query.all, columns)
        try:
            r = Resolver().resolve(query, self._catalog); v = Validator().validate(r, self._catalog)
            return self._planner.execute(self._optimize_ast(v), self._catalog)
        except Exception: return ExecutionResult()

    def _strip_ctes(self, ast):
        if isinstance(ast, SelectStmt) and ast.ctes: return dataclasses.replace(ast, ctes=[])
        if isinstance(ast, ExplainStmt): return dataclasses.replace(ast, statement=self._strip_ctes(ast.statement))
        return ast

    def _optimize_ast(self, ast):
        if isinstance(ast, SelectStmt): return self._optimizer.optimize(ast)
        if isinstance(ast, ExplainStmt) and isinstance(ast.statement, SelectStmt):
            return dataclasses.replace(ast, statement=self._optimizer.optimize(ast.statement))
        if isinstance(ast, SetOperationStmt):
            return dataclasses.replace(ast, left=self._optimize_ast(ast.left), right=self._optimize_ast(ast.right))
        return ast

    def analyze_table(self, name):
        schema = self._catalog.get_table(name); store = self._catalog.get_store(name)
        stats = TableStatistics.compute(name, store, schema); self._stats[name] = stats
        if self._integrated: self._integrated.update_stats(name, stats)
        return stats

    def get_table_stats(self, name): return self._stats.get(name)
    @property
    def data_dir(self): return self._catalog.data_dir
    @property
    def memory_budget(self) -> MemoryBudget: return self._budget
    def get_table_names(self): return [t for t in self._catalog.list_tables() if not t.startswith(_CTE_PREFIX)]
    def get_table_schema(self, name): return self._catalog.get_table(name)
    def get_table_row_count(self, name): return self._catalog.get_store(name).row_count

    def close(self):
        if self._merge_worker:
            try: self._merge_worker.stop()
            except Exception: pass
        for tname in self._catalog.list_tables():
            try:
                store = self._catalog.get_store(tname)
                if hasattr(store, 'close'): store.close()
            except Exception: pass
        if self._wal:
            try: self._wal.close()
            except Exception: pass
