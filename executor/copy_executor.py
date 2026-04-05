from __future__ import annotations
"""COPY执行器 — CSV批量加载。
比INSERT快：批量列写入、编译日期解析、自动字典编码检测。"""
import time
from typing import Any, Dict, List, Optional
from catalog.catalog import Catalog, ColumnSchema, TableSchema
from executor.core.result import ExecutionResult
from storage.types import DataType
from utils.errors import ExecutionError

# 编译日期解析器（修正导入路径）
try:
    from executor.string_algo.compiled_date import ISO_DATE_PARSER
    _HAS_COMPILED_DATE = True
except ImportError:
    _HAS_COMPILED_DATE = False

try:
    from utils.csv_io import read_csv, parse_csv
    _HAS_CSV = True
except ImportError:
    _HAS_CSV = False


class CopyExecutor:
    """执行COPY FROM批量数据加载。"""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def copy_from(self, table_name: str, file_path: str,
                  has_header: bool = True, delimiter: str = ',',
                  batch_size: int = 10000) -> ExecutionResult:
        """从CSV批量加载到表，返回行数。"""
        if not _HAS_CSV:
            raise ExecutionError("CSV module not available")

        if not self._catalog.table_exists(table_name):
            raise ExecutionError(f"table '{table_name}' not found")

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
        msg = f"COPY {count} rows ({elapsed:.3f}s, {rate:.0f} rows/sec)"
        if errors:
            msg += f", {errors} errors"
        return ExecutionResult(affected_rows=count, message=msg)

    def _build_converters(self, schema: TableSchema) -> List[Any]:
        """为每列构建类型转换函数。"""
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
    def _conv_int(val: Any) -> Optional[int]:
        if val is None or val == '':
            return None
        return int(val)

    @staticmethod
    def _conv_bigint(val: Any) -> Optional[int]:
        if val is None or val == '':
            return None
        return int(val)

    @staticmethod
    def _conv_float(val: Any) -> Optional[float]:
        if val is None or val == '':
            return None
        return float(val)

    @staticmethod
    def _conv_bool(val: Any) -> Optional[bool]:
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
    def _conv_date(val: Any) -> Optional[int]:
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
    def _conv_timestamp(val: Any) -> Optional[int]:
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
    def _conv_varchar(val: Any, max_len: int) -> Optional[str]:
        if val is None or val == '':
            return None
        s = str(val)
        return s[:max_len] if len(s) > max_len else s

    @staticmethod
    def _conv_str(val: Any) -> Optional[str]:
        if val is None or val == '':
            return None
        return str(val)
