from __future__ import annotations

"""Interleaved Sort Key — encode multi-column sort key as comparable bytes.
One memcmp replaces multiple column comparisons."""
import struct
from typing import Any, List, Optional, Tuple
from storage.types import DataType


def encode_sort_key(values: list, directions: list, null_positions: list,
                    dtypes: list) -> bytes:
    """Encode a multi-column sort key into a single comparable byte string.

    Rules:
      INT ASC:     big-endian + flip sign bit
      INT DESC:    big-endian + flip sign bit + flip all bytes
      VARCHAR ASC: UTF-8 + \\x00 terminator
      VARCHAR DESC: flip all bytes
      NULL FIRST:  \\x00 prefix (sorts before any value)
      NULL LAST:   \\xFF prefix (sorts after any value)
    """
    parts: list = []
    for val, direction, null_pos, dtype in zip(values, directions, null_positions, dtypes):
        is_desc = direction == 'DESC'
        null_last = null_pos == 'NULLS_LAST' or (null_pos is None and not is_desc)

        if val is None:
            parts.append(b'\xFF' if null_last else b'\x00')
            # Pad with zeros for consistent length
            if dtype in (DataType.INT, DataType.BIGINT):
                parts.append(b'\x00' * 8)
            elif dtype in (DataType.FLOAT, DataType.DOUBLE):
                parts.append(b'\x00' * 8)
            else:
                parts.append(b'\x00')
            continue

        # Non-null prefix
        parts.append(b'\x01' if null_last else b'\x80')

        if dtype in (DataType.INT, DataType.BIGINT):
            # Big-endian + flip sign bit
            encoded = struct.pack('>q', val)
            encoded = bytes([encoded[0] ^ 0x80] + list(encoded[1:]))
            if is_desc:
                encoded = bytes(b ^ 0xFF for b in encoded)
            parts.append(encoded)

        elif dtype in (DataType.FLOAT, DataType.DOUBLE):
            raw = struct.pack('>d', val)
            # IEEE 754 → comparable encoding
            as_int = int.from_bytes(raw, 'big')
            if as_int & (1 << 63):  # negative
                as_int = ~as_int & ((1 << 64) - 1)
            else:
                as_int |= (1 << 63)
            encoded = as_int.to_bytes(8, 'big')
            if is_desc:
                encoded = bytes(b ^ 0xFF for b in encoded)
            parts.append(encoded)

        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            encoded = str(val).encode('utf-8') + b'\x00'
            if is_desc:
                encoded = bytes(b ^ 0xFF for b in encoded)
            parts.append(encoded)

        elif dtype == DataType.BOOLEAN:
            encoded = b'\x01' if val else b'\x00'
            if is_desc:
                encoded = bytes(b ^ 0xFF for b in encoded)
            parts.append(encoded)

        else:
            encoded = str(val).encode('utf-8') + b'\x00'
            if is_desc:
                encoded = bytes(b ^ 0xFF for b in encoded)
            parts.append(encoded)

    return b''.join(parts)


def build_sort_keys(rows: list, col_indices: list,
                    directions: list, null_positions: list,
                    dtypes: list) -> List[Tuple[bytes, int]]:
    """Build interleaved sort keys for all rows.
    Returns list of (encoded_key, original_row_index)."""
    result = []
    for row_idx, row in enumerate(rows):
        values = [row[ci] for ci in col_indices]
        key = encode_sort_key(values, directions, null_positions, dtypes)
        result.append((key, row_idx))
    return result


def sort_by_interleaved_keys(rows: list, col_indices: list,
                             directions: list, null_positions: list,
                             dtypes: list) -> list:
    """Sort rows using interleaved keys. One comparison per row pair."""
    keyed = build_sort_keys(rows, col_indices, directions, null_positions, dtypes)
    keyed.sort(key=lambda x: x[0])
    return [rows[k[1]] for k in keyed]
