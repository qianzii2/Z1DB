from __future__ import annotations
"""Binary columnar file format for Z1DB.

File layout:
  [Header 64B]
    Magic b'Z1DB' (4B) | Version u16 (2B) | Flags u16 (2B)
    RowCount u64 (8B) | ColumnCount u16 (2B) | ChunkSize u32 (4B)
    SchemaOffset u64 (8B) | Created u64 (8B) | Padding (26B)
  [ChunkGroup 0..N]
    For each column:
      [NullBitmapLen u32 (4B)][NullBitmap bytes]
      [DataLen u32 (4B)][Data bytes]
  [Footer]
    [Schema JSON bytes][FooterLen u32 (4B)]
"""
import json
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAGIC = b'Z1DB'
VERSION = 1
HEADER_SIZE = 64


class TableFileWriter:
    """Writes a table to binary columnar format."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._f = open(path, 'wb')
        self._row_count = 0
        self._col_count = 0
        self._chunk_size = 0
        self._schema_data: bytes = b''
        # Write placeholder header — fill in at finalize
        self._f.write(b'\x00' * HEADER_SIZE)

    def write_chunk_group(self, chunk_data: List[Tuple[bytes, bytes]]) -> None:
        """Write one chunk group. chunk_data = [(null_bitmap_bytes, data_bytes), ...] per column."""
        for null_bmp, data in chunk_data:
            self._f.write(struct.pack('<I', len(null_bmp)))
            self._f.write(null_bmp)
            self._f.write(struct.pack('<I', len(data)))
            self._f.write(data)

    def finalize(self, schema_json: dict, row_count: int, col_count: int,
                 chunk_size: int) -> None:
        """Write footer and update header."""
        schema_bytes = json.dumps(schema_json).encode('utf-8')
        self._f.write(schema_bytes)
        self._f.write(struct.pack('<I', len(schema_bytes)))
        # Seek back and write header
        self._f.seek(0)
        header = bytearray(HEADER_SIZE)
        header[0:4] = MAGIC
        struct.pack_into('<H', header, 4, VERSION)
        struct.pack_into('<H', header, 6, 0)  # flags
        struct.pack_into('<Q', header, 8, row_count)
        struct.pack_into('<H', header, 16, col_count)
        struct.pack_into('<I', header, 18, chunk_size)
        struct.pack_into('<Q', header, 22, 0)  # schema offset (unused)
        struct.pack_into('<Q', header, 30, int(time.time()))
        self._f.write(bytes(header))
        self._f.close()


class TableFileReader:
    """Reads a binary columnar file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self) -> Tuple[dict, List[List[list]]]:
        """Returns (schema_json, chunk_groups).
        chunk_groups = [[col0_rows, col1_rows, ...], ...]"""
        with open(self._path, 'rb') as f:
            data = f.read()

        if len(data) < HEADER_SIZE:
            return {}, []

        # Read header
        magic = data[0:4]
        if magic != MAGIC:
            raise ValueError(f"Invalid file magic: {magic}")
        row_count = struct.unpack_from('<Q', data, 8)[0]
        col_count = struct.unpack_from('<H', data, 16)[0]
        chunk_size = struct.unpack_from('<I', data, 18)[0]

        # Read footer (schema)
        footer_len = struct.unpack_from('<I', data, len(data) - 4)[0]
        schema_bytes = data[len(data) - 4 - footer_len: len(data) - 4]
        schema = json.loads(schema_bytes.decode('utf-8'))

        # Read chunk groups
        pos = HEADER_SIZE
        end = len(data) - 4 - footer_len
        all_null_bmps: List[List[bytes]] = []
        all_data_blobs: List[List[bytes]] = []

        # Read all chunks
        chunk_null_bmps: List[bytes] = []
        chunk_data_blobs: List[bytes] = []
        col_idx = 0

        while pos < end:
            # Null bitmap
            if pos + 4 > end: break
            nb_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
            if pos + nb_len > end: break
            null_bmp = data[pos:pos + nb_len]; pos += nb_len
            # Data
            if pos + 4 > end: break
            d_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
            if pos + d_len > end: break
            data_blob = data[pos:pos + d_len]; pos += d_len

            chunk_null_bmps.append(null_bmp)
            chunk_data_blobs.append(data_blob)
            col_idx += 1

            if col_idx >= col_count:
                all_null_bmps.append(chunk_null_bmps)
                all_data_blobs.append(chunk_data_blobs)
                chunk_null_bmps = []
                chunk_data_blobs = []
                col_idx = 0

        return schema, list(zip(all_null_bmps, all_data_blobs))

    def read_header(self) -> dict:
        with open(self._path, 'rb') as f:
            header = f.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE or header[0:4] != MAGIC:
            return {}
        return {
            'version': struct.unpack_from('<H', header, 4)[0],
            'row_count': struct.unpack_from('<Q', header, 8)[0],
            'col_count': struct.unpack_from('<H', header, 16)[0],
            'chunk_size': struct.unpack_from('<I', header, 18)[0],
        }


# ═══ Serialization helpers ═══

def serialize_column(values: list, null_flags: list, dtype_name: str) -> Tuple[bytes, bytes]:
    """Serialize a column's data + null bitmap to bytes."""
    n = len(values)
    # Null bitmap
    bmp = bytearray((n + 7) // 8)
    for i in range(n):
        if null_flags[i]:
            bmp[i >> 3] |= (1 << (i & 7))
    # Data
    if dtype_name in ('INT', 'DATE'):
        data = struct.pack(f'<{n}i', *[v if v is not None else 0 for v in values])
    elif dtype_name in ('BIGINT', 'TIMESTAMP'):
        data = struct.pack(f'<{n}q', *[v if v is not None else 0 for v in values])
    elif dtype_name in ('FLOAT', 'DOUBLE'):
        data = struct.pack(f'<{n}d', *[float(v) if v is not None else 0.0 for v in values])
    elif dtype_name == 'BOOLEAN':
        bool_bmp = bytearray((n + 7) // 8)
        for i in range(n):
            if values[i] and not null_flags[i]:
                bool_bmp[i >> 3] |= (1 << (i & 7))
        data = bytes(bool_bmp)
    else:
        # VARCHAR/TEXT — length-prefixed strings
        parts = bytearray()
        for i in range(n):
            s = str(values[i]).encode('utf-8') if values[i] is not None and not null_flags[i] else b''
            parts.extend(struct.pack('<H', len(s)))
            parts.extend(s)
        data = bytes(parts)
    return bytes(bmp), data


def deserialize_column(null_bmp: bytes, data: bytes, n: int,
                        dtype_name: str) -> Tuple[list, list]:
    """Deserialize column bytes → (values, null_flags)."""
    null_flags = [False] * n
    for i in range(n):
        if i // 8 < len(null_bmp):
            if null_bmp[i >> 3] & (1 << (i & 7)):
                null_flags[i] = True
    values: list = []
    if dtype_name in ('INT', 'DATE'):
        raw = struct.unpack_from(f'<{n}i', data, 0) if len(data) >= n * 4 else [0] * n
        values = [None if null_flags[i] else raw[i] for i in range(n)]
    elif dtype_name in ('BIGINT', 'TIMESTAMP'):
        raw = struct.unpack_from(f'<{n}q', data, 0) if len(data) >= n * 8 else [0] * n
        values = [None if null_flags[i] else raw[i] for i in range(n)]
    elif dtype_name in ('FLOAT', 'DOUBLE'):
        raw = struct.unpack_from(f'<{n}d', data, 0) if len(data) >= n * 8 else [0.0] * n
        values = [None if null_flags[i] else raw[i] for i in range(n)]
    elif dtype_name == 'BOOLEAN':
        for i in range(n):
            if null_flags[i]:
                values.append(None)
            else:
                if i // 8 < len(data):
                    values.append(bool(data[i >> 3] & (1 << (i & 7))))
                else:
                    values.append(False)
    else:
        pos = 0
        for i in range(n):
            if null_flags[i]:
                if pos + 2 <= len(data):
                    slen = struct.unpack_from('<H', data, pos)[0]; pos += 2 + slen
                values.append(None)
            else:
                if pos + 2 > len(data):
                    values.append(''); continue
                slen = struct.unpack_from('<H', data, pos)[0]; pos += 2
                s = data[pos:pos + slen].decode('utf-8', errors='replace'); pos += slen
                values.append(s)
    return values, null_flags
