from __future__ import annotations
"""FSST — Fast Static Symbol Table compression for strings.
Paper: Boncz et al., 2020.
Greedily selects 256 most frequent substrings (1-8 bytes) as symbols.
Encodes each symbol with 1 byte. Decompression = table lookup (very fast)."""
from typing import Dict, List, Tuple


class SymbolTable:
    """FSST symbol table — maps frequent substrings to 1-byte codes."""

    __slots__ = ('_symbols', '_codes', '_max_len')

    def __init__(self) -> None:
        self._symbols: List[bytes] = []  # code → symbol bytes
        self._codes: Dict[bytes, int] = {}  # symbol → code
        self._max_len = 0

    @staticmethod
    def train(strings: List[str], max_symbols: int = 255) -> SymbolTable:
        """Train symbol table on sample strings.
        Greedily picks highest-gain substrings."""
        st = SymbolTable()
        # Count all substrings of length 1-8
        freq: Dict[bytes, int] = {}
        for s in strings:
            encoded = s.encode('utf-8')
            for length in range(1, min(9, len(encoded) + 1)):
                for start in range(len(encoded) - length + 1):
                    sub = encoded[start:start + length]
                    freq[sub] = freq.get(sub, 0) + 1

        # Score = frequency * (length - 1): saving = (length-1) bytes per occurrence
        scored: List[Tuple[float, bytes]] = []
        for sub, count in freq.items():
            gain = count * (len(sub) - 1)
            if gain > 0:
                scored.append((gain, sub))
        scored.sort(key=lambda x: -x[0])

        # Greedily pick non-overlapping symbols
        used_code = 0
        for _, sub in scored:
            if used_code >= max_symbols:
                break
            # Check if this symbol conflicts with already selected ones
            # (simplified: just add all top symbols)
            st._symbols.append(sub)
            st._codes[sub] = used_code
            used_code += 1
            if len(sub) > st._max_len:
                st._max_len = len(sub)

        # Ensure single bytes are always present (escape path)
        for b in range(256):
            single = bytes([b])
            if single not in st._codes and used_code < 256:
                st._symbols.append(single)
                st._codes[single] = used_code
                used_code += 1

        return st

    @property
    def num_symbols(self) -> int:
        return len(self._symbols)


def fsst_encode(text: str, table: SymbolTable) -> bytes:
    """Encode string using FSST symbol table."""
    encoded = text.encode('utf-8')
    result = bytearray()
    i = 0
    while i < len(encoded):
        best_len = 0
        best_code = -1
        # Try longest match first (greedy)
        for length in range(min(table._max_len, len(encoded) - i), 0, -1):
            sub = encoded[i:i + length]
            if sub in table._codes:
                best_len = length
                best_code = table._codes[sub]
                break
        if best_code >= 0:
            result.append(best_code)
            i += best_len
        else:
            # Escape: 0xFF followed by raw byte
            result.append(0xFF)
            result.append(encoded[i])
            i += 1
    return bytes(result)


def fsst_decode(data: bytes, table: SymbolTable) -> str:
    """Decode FSST-compressed data."""
    result = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFF and i + 1 < len(data):
            result.append(data[i + 1])
            i += 2
        else:
            code = data[i]
            if code < len(table._symbols):
                result.extend(table._symbols[code])
            i += 1
    return result.decode('utf-8', errors='replace')


def fsst_compression_ratio(strings: List[str], table: SymbolTable) -> float:
    """Estimate compression ratio."""
    original = sum(len(s.encode('utf-8')) for s in strings)
    compressed = sum(len(fsst_encode(s, table)) for s in strings)
    return compressed / original if original > 0 else 1.0
