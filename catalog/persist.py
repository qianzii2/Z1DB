from __future__ import annotations
"""持久化 — Schema JSON + 数据（LSM 或二进制/JSON 回退）。
LSM 模式下此模块只负责 schema 读写，数据由 LSMStore 管理。"""
import json
import struct
from pathlib import Path
from typing import Any, Dict, List
from catalog.catalog import ColumnSchema, TableSchema
from storage.types import DataType


def save(schemas: Dict[str, TableSchema],
         path: Path) -> None:
    """保存 schema 到 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    for name, schema in schemas.items():
        data[name] = {
            'name': schema.name,
            'columns': [{
                'name': c.name,
                'dtype': c.dtype.name,
                'nullable': c.nullable,
                'primary_key': c.primary_key,
                'max_length': c.max_length,
            } for c in schema.columns],
        }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load(path: Path) -> Dict[str, TableSchema]:
    """从 JSON 加载 schema。"""
    with open(path, 'r') as f:
        data = json.load(f)
    schemas: Dict[str, TableSchema] = {}
    for name, td in data.items():
        columns = [ColumnSchema(
            name=cd['name'],
            dtype=DataType[cd['dtype']],
            nullable=cd.get('nullable', True),
            primary_key=cd.get('primary_key', False),
            max_length=cd.get('max_length'))
            for cd in td['columns']]
        schemas[name] = TableSchema(
            name=td['name'], columns=columns)
    return schemas


def save_data(stores: dict, data_dir: Path) -> None:
    """保存表数据（非 LSM 模式的回退路径）。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, store in stores.items():
        # LSMStore 自行管理持久化，跳过
        try:
            from storage.lsm.lsm_store import LSMStore
            if isinstance(store, LSMStore):
                continue
        except ImportError:
            pass
        try:
            _save_binary(name, store, data_dir)
        except Exception:
            _save_json(name, store, data_dir)


def load_data(stores: dict, data_dir: Path) -> None:
    """加载表数据（非 LSM 模式的回退路径）。"""
    for name, store in stores.items():
        try:
            from storage.lsm.lsm_store import LSMStore
            if isinstance(store, LSMStore):
                continue
        except ImportError:
            pass
        bin_path = data_dir / f'{name}.z1db'
        json_path = data_dir / f'{name}.data.json'
        if bin_path.exists():
            try:
                _load_binary(name, store, bin_path)
                continue
            except Exception:
                pass
        if json_path.exists():
            _load_json(store, json_path)


def remove_table_data(table_name: str,
                      data_dir: Path) -> None:
    for suffix in ('.z1db', '.data.json'):
        p = data_dir / f'{table_name}{suffix}'
        if p.exists():
            p.unlink()


# ═══ 二进制格式（旧模式回退）═══

def _save_binary(name: str, store: Any,
                 data_dir: Path) -> None:
    from storage.table_file import (
        TableFileWriter, serialize_column)
    from metal.config import CHUNK_SIZE
    schema = store.schema
    path = data_dir / f'{name}.z1db'
    writer = TableFileWriter(path)
    all_rows = store.read_all_rows()
    n = len(all_rows)
    chunk_data = []
    for ci, col in enumerate(schema.columns):
        values = [row[ci] for row in all_rows]
        null_flags = [v is None for v in values]
        null_bmp, data = serialize_column(
            values, null_flags, col.dtype.name)
        chunk_data.append((null_bmp, data))
    writer.write_chunk_group(chunk_data)
    schema_json = {
        'name': schema.name,
        'columns': [{
            'name': c.name, 'dtype': c.dtype.name,
            'nullable': c.nullable,
            'primary_key': c.primary_key,
            'max_length': c.max_length,
        } for c in schema.columns],
    }
    writer.finalize(schema_json, n,
                    len(schema.columns), CHUNK_SIZE)


def _load_binary(name: str, store: Any,
                 path: Path) -> None:
    from storage.table_file import (
        TableFileReader, deserialize_column)
    reader = TableFileReader(path)
    header = reader.read_header()
    if not header:
        return
    schema_data, chunk_groups = reader.read()
    if not schema_data or not chunk_groups:
        return
    n = header['row_count']
    col_count = header['col_count']
    columns = schema_data.get('columns', [])
    for null_bmps, data_blobs in chunk_groups:
        col_values = []
        for ci in range(min(col_count, len(null_bmps))):
            dtype_name = (columns[ci]['dtype']
                          if ci < len(columns)
                          else 'VARCHAR')
            values, nulls = deserialize_column(
                null_bmps[ci], data_blobs[ci],
                n, dtype_name)
            col_values.append(values)
        for ri in range(n):
            row = [col_values[ci][ri]
                   if ci < len(col_values) else None
                   for ci in range(col_count)]
            store.append_row(row)


# ═══ JSON 回退 ═══

def _save_json(name: str, store: Any,
               data_dir: Path) -> None:
    rows = store.read_all_rows()
    safe_rows = []
    for row in rows:
        safe_row = []
        for val in row:
            if val is None: safe_row.append(None)
            elif isinstance(val, bool): safe_row.append(val)
            elif isinstance(val, (int, float, str)):
                safe_row.append(val)
            else: safe_row.append(str(val))
        safe_rows.append(safe_row)
    with open(data_dir / f'{name}.data.json', 'w') as f:
        json.dump(safe_rows, f)


def _load_json(store: Any, path: Path) -> None:
    with open(path, 'r') as f:
        rows = json.load(f)
    for row in rows:
        store.append_row(row)
