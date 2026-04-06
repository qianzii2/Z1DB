from __future__ import annotations
"""COPY 执行器 — CSV 批量加载/导出。
[M03] 新增 COPY TO 导出功能。"""
import time
from typing import Any, Dict, List, Optional
from catalog.catalog import Catalog, ColumnSchema, TableSchema
from executor.core.result import ExecutionResult
from storage.types import DataType
from utils.errors import ExecutionError

try:
    from executor.string_algo.compiled_date import ISO_DATE_PARSER
    _HAS_COMPILED_DATE = True
except ImportError:
    _HAS_COMPILED_DATE = False

try:
    from utils.csv_io import read_csv, write_csv, parse_csv
    _HAS_CSV = True
except ImportError:
    _HAS_CSV = False


class CopyExecutor:
    """执行 COPY FROM/TO 批量数据操作。"""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def copy_from(self, table_name: str, file_path: str,
                  has_header: bool = True, delimiter: str = ',',
                  batch_size: int = 10000) -> ExecutionResult:
        """从 CSV 批量加载到表。"""
        if not _HAS_CSV:
            raise ExecutionError("CSV 模块不可用")
        if not self._catalog.table_exists(table_name):
            raise ExecutionError(f"表 '{table_name}' 不存在")

        schema = self._catalog.get_table(table_name)
        store = self._catalog.get_store(table_name)
        t0 = time.perf_counter()
        headers, rows = read_csv(file_path, delimiter=delimiter,
                                  has_header=has_header)
        converters = self._build_converters(schema)
        count = 0
        errors = 0

        for raw_row in rows:
            while len(raw_row) < len(schema.columns):
                raw_row.append(None)
            raw_row = raw_row[:len(schema.columns)]
            converted = []
            valid = True
            for ci, (val, col) in enumerate(zip(raw_row, schema.columns)):
                try:
                    converted.append(converters[ci](val))
                except Exception:
                    converted.append(None)
                    if not col.nullable:
                        valid = False
                        break
            if valid:
                store.append_row(converted)
                count += 1
            else:
                errors += 1

        elapsed = time.perf_counter() - t0
        rate = count / elapsed if elapsed > 0 else 0
        msg = f"COPY {count} 行 ({elapsed:.3f}s, {rate:.0f} 行/秒)"
        if errors:
            msg += f", {errors} 个错误"
        return ExecutionResult(affected_rows=count, message=msg)

    def copy_to(self, table_name: str, file_path: str,
                has_header: bool = True,
                delimiter: str = ',') -> ExecutionResult:
        """[M03] 导出表到 CSV 文件。"""
        if not _HAS_CSV:
            raise ExecutionError("CSV 模块不可用")
        if not self._catalog.table_exists(table_name):
            raise ExecutionError(f"表 '{table_name}' 不存在")

        schema = self._catalog.get_table(table_name)
        store = self._catalog.get_store(table_name)
        t0 = time.perf_counter()

        all_rows = store.read_all_rows()
        # 格式化值用于 CSV 输出
        formatted_rows = []
        for row in all_rows:
            fmt_row = []
            for ci, val in enumerate(row):
                col = schema.columns[ci] if ci < len(schema.columns) else None
                fmt_row.append(self._format_for_csv(val, col))
            formatted_rows.append(fmt_row)

        headers = schema.column_names
        count = write_csv(file_path, headers, formatted_rows,
                          delimiter=delimiter)

        elapsed = time.perf_counter() - t0
        rate = count / elapsed if elapsed > 0 else 0
        msg = f"COPY {count} 行到 '{file_path}' ({elapsed:.3f}s, {rate:.0f} 行/秒)"
        return ExecutionResult(affected_rows=count, message=msg)

    @staticmethod
    def _format_for_csv(val: Any, col: Optional[ColumnSchema]) -> Any:
        """将内部值格式化为 CSV 可写形式。"""
        if val is None:
            return None
        if col is None:
            return val
        dt = col.dtype
        if dt == DataType.DATE:
            try:
                import datetime
                d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(val))
                return d.isoformat()
            except Exception:
                return val
        if dt == DataType.TIMESTAMP:
            try:
                import datetime
                dt_obj = datetime.datetime(1970, 1, 1) + datetime.timedelta(
                    microseconds=int(val))
                return dt_obj.isoformat()
            except Exception:
                return val
        if dt == DataType.BOOLEAN:
            return 'true' if val else 'false'
        return val

    def _build_converters(self, schema: TableSchema) -> List[Any]:
        converters = []
        for col in schema.columns:
            dt = col.dtype
            if dt == DataType.INT:
                converters.append(self._conv_int)
            elif dt == DataType.BIGINT:
                converters.append(self._conv_bigint)
            elif dt in (DataType.FLOAT, DataType.DOUBLE):
                converters.append(self._conv_float)
            elif dt == DataType.BOOLEAN:
                converters.append(self._conv_bool)
            elif dt == DataType.DATE:
                converters.append(self._conv_date)
            elif dt == DataType.TIMESTAMP:
                converters.append(self._conv_timestamp)
            elif dt in (DataType.VARCHAR, DataType.TEXT):
                max_len = col.max_length
                if max_len:
                    converters.append(
                        lambda v, ml=max_len: self._conv_varchar(v, ml))
                else:
                    converters.append(self._conv_str)
            else:
                converters.append(self._conv_str)
        return converters

    @staticmethod
    def _conv_int(val): return int(val) if val is not None and val != '' else None
    @staticmethod
    def _conv_bigint(val): return int(val) if val is not None and val != '' else None
    @staticmethod
    def _conv_float(val): return float(val) if val is not None and val != '' else None

    @staticmethod
    def _conv_bool(val):
        if val is None or val == '':
            return None
        if isinstance(val, bool):
            return val
        lower = str(val).strip().lower()
        if lower in ('true', '1', 'yes', 't', 'y', 'on'):
            return True
        if lower in ('false', '0', 'no', 'f', 'n', 'off'):
            return False
        return None

    @staticmethod
    def _conv_date(val):
        if val is None or val == '':
            return None
        if _HAS_COMPILED_DATE:
            result = ISO_DATE_PARSER.parse_date(str(val))
            if result is not None:
                return result
        import datetime
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d.%m.%Y'):
            try:
                d = datetime.datetime.strptime(str(val), fmt).date()
                return (d - datetime.date(1970, 1, 1)).days
            except ValueError:
                continue
        return None

    @staticmethod
    def _conv_timestamp(val):
        if val is None or val == '':
            return None
        if _HAS_COMPILED_DATE:
            result = ISO_DATE_PARSER.parse_timestamp(str(val))
            if result is not None:
                return result
        import datetime
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
            try:
                dt = datetime.datetime.strptime(str(val), fmt)
                return int((dt - datetime.datetime(1970, 1, 1))
                           .total_seconds() * 1_000_000)
            except ValueError:
                continue
        return None

    @staticmethod
    def _conv_varchar(val, max_len):
        if val is None or val == '':
            return None
        s = str(val)
        return s[:max_len] if len(s) > max_len else s

    @staticmethod
    def _conv_str(val):
        return str(val) if val is not None and val != '' else None
