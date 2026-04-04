from __future__ import annotations
"""JSON persistence for the catalog."""

import json
from pathlib import Path
from typing import Dict

from catalog.catalog import ColumnSchema, TableSchema
from storage.types import DataType


def save(schemas: Dict[str, TableSchema], path: Path) -> None:
    """Serialize schemas to a JSON file."""
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
    """Deserialize schemas from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    schemas: Dict[str, TableSchema] = {}
    for name, table_data in data.items():
        columns = []
        for cd in table_data['columns']:
            columns.append(ColumnSchema(
                name=cd['name'],
                dtype=DataType[cd['dtype']],
                nullable=cd.get('nullable', True),
                primary_key=cd.get('primary_key', False),
                max_length=cd.get('max_length'),
            ))
        schemas[name] = TableSchema(name=table_data['name'], columns=columns)
    return schemas
