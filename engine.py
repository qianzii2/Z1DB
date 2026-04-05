from __future__ import annotations
"""Z1DB引擎 — WAL + 事务 + 缓存。"""
from typing import Any, Optional
import dataclasses

from catalog.catalog import Catalog, ColumnSchema, TableSchema
from catalog.statistics import TableStatistics
from executor.core.result import ExecutionResult
from executor.functions.registry import FunctionRegistry
from executor.memory_budget import MemoryBudget
from executor.optimizer import QueryOptimizer
from executor.result_cache import ResultCache
from executor.simple_planner import SimplePlanner
from parser.ast import SelectStmt, ExplainStmt, SetOperationStmt
from parser.lexer import Lexer
from parser.parser import Parser
from parser.resolver import Resolver
from parser.validator import Validator
from storage.types import DataType
from utils.timer import Timer

_CTE_PREFIX = '__cte_'


class Engine:
    def __init__(self, data_dir: str = ':memory:') -> None:
        self._catalog = Catalog(data_dir)
        self._registry = FunctionRegistry()
        self._registry.register_defaults()
        self._planner = SimplePlanner(self._registry)
        self._optimizer = QueryOptimizer()
        self._stats: dict[str, TableStatistics] = {}
        self._suppress_persist = False
        self._budget = MemoryBudget()
        self._result_cache = ResultCache()
        self._table_versions: dict[str, int] = {}

        # WAL
        self._wal = None
        if data_dir != ':memory:':
            try:
                from storage.wal import WriteAheadLog
                self._wal = WriteAheadLog(data_dir)
                self._wal.open()
                self._replay_wal()
            except Exception:
                self._wal = None

        # 事务管理器
        self._txn = None
        try:
            from txn.manager import TransactionManager
            self._txn = TransactionManager()
        except ImportError:
            pass

        # 集成规划器
        try:
            from executor.integrated_planner import IntegratedPlanner
            self._integrated = IntegratedPlanner(self._registry)
        except ImportError:
            self._integrated = None

    def execute(self, sql: str) -> ExecutionResult:
        sql_stripped = sql.strip().rstrip(';').strip()
        upper = sql_stripped.upper()

        # 事务控制语句
        if self._txn:
            if upper == 'BEGIN':
                txn_id = self._txn.begin()
                return ExecutionResult(message=f'BEGIN (txn {txn_id})')
            if upper == 'COMMIT':
                self._txn.commit()
                if self._catalog.is_persistent:
                    self._catalog.persist()
                return ExecutionResult(message='COMMIT')
            if upper == 'ROLLBACK':
                self._txn.rollback(self._catalog)
                return ExecutionResult(message='ROLLBACK')

        # 查询缓存
        sql_hash = ResultCache.hash_sql(sql)
        cached = self._result_cache.get(sql_hash, self._table_versions)
        if cached is not None:
            return cached

        with Timer() as t:
            tokens = Lexer(sql).tokenize()
            ast = Parser(tokens).parse()
            cte_tables = self._materialize_ctes(ast)
            try:
                if cte_tables:
                    ast = self._strip_ctes(ast)
                ast = Resolver().resolve(ast, self._catalog)
                ast = Validator().validate(ast, self._catalog)
                ast = self._optimize_ast(ast)

                # DML前：事务快照 + WAL
                if self._is_dml(ast):
                    table = self._get_dml_table(ast)
                    if table:
                        if self._txn:
                            auto_id = self._txn.auto_begin()
                            self._txn.snapshot_table(table, self._catalog)
                        if self._wal:
                            self._wal.append(self._get_dml_type(ast), table, {'sql': sql.strip()})

                if self._integrated and isinstance(ast, SelectStmt):
                    result = self._integrated.execute(ast, self._catalog)
                else:
                    result = self._planner.execute(ast, self._catalog)

                # DML后：自动提交
                if self._is_dml(ast) and self._txn and self._txn.auto_commit:
                    self._txn.commit()

            except Exception as e:
                # 出错回滚
                if self._txn and self._txn.in_transaction and self._txn.auto_commit:
                    self._txn.rollback(self._catalog)
                raise
            finally:
                self._suppress_persist = True
                for tname in cte_tables:
                    try: self._catalog.drop_table(tname)
                    except Exception: pass
                self._suppress_persist = False

        result.timing = t.elapsed
        is_mutation = result.affected_rows > 0 or result.message == 'OK'

        if self._catalog.is_persistent and not self._suppress_persist and is_mutation:
            self._catalog.persist()
            if self._wal:
                try: self._wal.checkpoint()
                except Exception: pass

        if is_mutation:
            self._invalidate_modified_tables(ast)
        elif isinstance(ast, SelectStmt):
            tables = self._extract_tables(ast)
            versions = {tbl: self._table_versions.get(tbl, 0) for tbl in tables}
            self._result_cache.put(sql_hash, result, versions)

        return result

    # ═══ 辅助 ═══
    def _invalidate_modified_tables(self, ast):
        from parser.ast import InsertStmt, UpdateStmt, DeleteStmt, CreateTableStmt, DropTableStmt, AlterTableStmt
        table = None
        for cls in (InsertStmt, UpdateStmt, DeleteStmt, CreateTableStmt, DropTableStmt, AlterTableStmt):
            if isinstance(ast, cls): table = ast.table; break
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

    def _is_dml(self, ast):
        from parser.ast import InsertStmt, UpdateStmt, DeleteStmt
        return isinstance(ast, (InsertStmt, UpdateStmt, DeleteStmt))

    def _get_dml_type(self, ast):
        from parser.ast import InsertStmt, UpdateStmt, DeleteStmt
        if isinstance(ast, InsertStmt): return 'INSERT'
        if isinstance(ast, UpdateStmt): return 'UPDATE'
        if isinstance(ast, DeleteStmt): return 'DELETE'
        return 'UNKNOWN'

    def _get_dml_table(self, ast):
        return getattr(ast, 'table', None)

    def _replay_wal(self):
        if not self._wal: return
        try:
            for entry in self._wal.recover():
                if 'sql' in entry.data:
                    try:
                        s = entry.data['sql']
                        a = Parser(Lexer(s).tokenize()).parse()
                        a = Resolver().resolve(a, self._catalog)
                        a = Validator().validate(a, self._catalog)
                        self._planner.execute(a, self._catalog)
                    except Exception: pass
        except Exception: pass

    def _materialize_ctes(self, ast):
        created = []
        if isinstance(ast, SelectStmt) and ast.ctes:
            for entry in ast.ctes:
                cn, cq = entry[0], entry[1]
                ir = entry[2] if len(entry) > 2 else False
                cc = entry[3] if len(entry) > 3 else None
                nested = self._materialize_ctes(cq); created.extend(nested)
                if ir:
                    cr = self._execute_recursive_cte(cn, cq, cc)
                else:
                    try:
                        r = Resolver().resolve(cq, self._catalog)
                        v = Validator().validate(r, self._catalog)
                        o = self._optimize_ast(v)
                        cr = self._planner.execute(o, self._catalog)
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
            created.extend(self._materialize_ctes(ast.left))
            created.extend(self._materialize_ctes(ast.right))
        return created

    def _execute_recursive_cte(self, name, query, columns=None):
        from executor.recursive_cte import RecursiveCTEExecutor
        from executor.core.result import ExecutionResult
        if isinstance(query, SetOperationStmt) and query.op.upper() == 'UNION':
            return RecursiveCTEExecutor(self._planner, self._catalog).execute(name, query.left, query.right, query.all, columns)
        try:
            r = Resolver().resolve(query, self._catalog)
            v = Validator().validate(r, self._catalog)
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
        stats = TableStatistics.compute(name, store, schema)
        self._stats[name] = stats
        if self._integrated: self._integrated.update_stats(name, stats)
        return stats

    def get_table_stats(self, name): return self._stats.get(name)

    @property
    def data_dir(self): return self._catalog.data_dir
    def get_table_names(self): return [t for t in self._catalog.list_tables() if not t.startswith(_CTE_PREFIX)]
    def get_table_schema(self, name): return self._catalog.get_table(name)
    def get_table_row_count(self, name): return self._catalog.get_store(name).row_count

    def close(self):
        if self._wal:
            try: self._wal.close()
            except Exception: pass
