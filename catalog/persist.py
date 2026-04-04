from __future__ import annotations
"""JSON persistence for catalog schemas + table data."""
import json
from pathlib import Path
from typing import Dict
from catalog.catalog import ColumnSchema, TableSchema
from storage.types import DataType


def save(schemas: Dict[str, TableSchema], path: Path) -> None:
    """Write all table schemas to catalog.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    for name, schema in schemas.items():
        data[name] = {
            'name': schema.name,
            'columns': [
                {
                    'name': c.name,
                    'dtype': c.dtype.name,
                    'nullable': c.nullable,
                    'primary_key': c.primary_key,
                    'max_length': c.max_length,
                }
                for c in schema.columns
            ],
        }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load(path: Path) -> Dict[str, TableSchema]:
    """Read all table schemas from catalog.json."""
    with open(path, 'r') as f:
        data = json.load(f)
    schemas: Dict[str, TableSchema] = {}
    for name, td in data.items():
        columns = []
        for cd in td['columns']:
            columns.append(ColumnSchema(
                name=cd['name'],
                dtype=DataType[cd['dtype']],
                nullable=cd.get('nullable', True),
                primary_key=cd.get('primary_key', False),
                max_length=cd.get('max_length'),
            ))
        schemas[name] = TableSchema(name=td['name'], columns=columns)
    return schemas


def save_data(stores: dict, data_dir: Path) -> None:
    """Write all table row data to {table}.data.json files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, store in stores.items():
        rows = store.read_all_rows()
        table_path = data_dir / f'{name}.data.json'
        safe_rows = []
        for row in rows:
            safe_row = []
            for val in row:
                if val is None:
                    safe_row.append(None)
                elif isinstance(val, bool):
                    safe_row.append(val)
                elif isinstance(val, (int, float, str)):
                    safe_row.append(val)
                else:
                    safe_row.append(str(val))
            safe_rows.append(safe_row)
        with open(table_path, 'w') as f:
            json.dump(safe_rows, f)


def load_data(stores: dict, data_dir: Path) -> None:
    """Read table row data from {table}.data.json files."""
    for name, store in stores.items():
        table_path = data_dir / f'{name}.data.json'
        if table_path.exists():
            with open(table_path, 'r') as f:
                rows = json.load(f)
            for row in rows:
                store.append_row(row)


def remove_table_data(table_name: str, data_dir: Path) -> None:
    """Remove data file for a dropped table."""
    table_path = data_dir / f'{table_name}.data.json'
    if table_path.exists():
        table_path.unlink()
