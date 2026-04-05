from __future__ import annotations
"""Dictionary Encoding — best for string columns with NDV < 65536.
Comparisons on codes (integers) instead of strings: 10-100x faster."""
from typing import Any, Dict, List, Optional, Tuple
import array


class DictEncoded:
    """Dictionary-encoded column. Stores unique values + integer codes."""

    __slots__ = ('dictionary', 'codes', '_reverse')

    def __init__(self) -> None:
        self.dictionary: List[Any] = []
        self.codes: array.array = array.array('H')  # uint16
        self._reverse: Dict[Any, int] = {}

    @staticmethod
    def encode(values: list) -> DictEncoded:
        """Encode a list of values into dictionary form."""
        de = DictEncoded()
        for v in values:
            if v not in de._reverse:
                code = len(de.dictionary)
                de.dictionary.append(v)
                de._reverse[v] = code
            de.codes.append(de._reverse[v])
        return de

    def decode_all(self) -> list:
        """Decode back to original values."""
        return [self.dictionary[c] for c in self.codes]

    def decode(self, index: int) -> Any:
        return self.dictionary[self.codes[index]]

    @property
    def ndv(self) -> int:
        return len(self.dictionary)

    def __len__(self) -> int:
        return len(self.codes)

    def lookup_code(self, value: Any) -> Optional[int]:
        """Get code for a value. None if not in dictionary."""
        return self._reverse.get(value)

    def filter_eq(self, value: Any) -> list:
        """Find all indices where value matches. Operates on codes only."""
        code = self._reverse.get(value)
        if code is None:
            return []
        return [i for i, c in enumerate(self.codes) if c == code]

    def filter_eq_bitmap(self, value: Any, size: int) -> bytearray:
        """Return a bitmap of matching positions."""
        bm = bytearray((size + 7) // 8)
        code = self._reverse.get(value)
        if code is None:
            return bm
        for i, c in enumerate(self.codes):
            if c == code:
                bm[i >> 3] |= (1 << (i & 7))
        return bm

    def remap_codes(self, code_map: dict) -> list:
        """Apply a mapping on codes (for GROUP BY on dict-encoded columns)."""
        return [code_map.get(c, c) for c in self.codes]

    def compression_ratio(self) -> float:
        """Estimate vs storing raw strings."""
        if not self.dictionary:
            return 1.0
        avg_len = sum(len(str(v)) for v in self.dictionary) / len(self.dictionary)
        # codes: 2 bytes each; dict: avg_len * ndv
        compressed = len(self.codes) * 2 + avg_len * len(self.dictionary)
        original = avg_len * len(self.codes)
        return compressed / original if original > 0 else 1.0
