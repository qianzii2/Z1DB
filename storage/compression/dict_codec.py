from __future__ import annotations
"""字典编码 — 最适合低基数字符串列（NDV < 65536）。
将重复字符串替换为 uint16 编码，比较和聚合直接在编码上操作。
字符串比较从 O(len) 变为 O(1) 的整数比较。"""
from typing import Any, Dict, List, Optional, Tuple
import array


class DictEncoded:
    """字典编码列。dictionary = 唯一值列表，codes = 整数编码序列。"""

    __slots__ = ('dictionary', 'codes', '_reverse')

    def __init__(self) -> None:
        self.dictionary: List[Any] = []     # code → 原始值
        self.codes: array.array = array.array('H')  # uint16 编码
        self._reverse: Dict[Any, int] = {}  # 原始值 → code

    @staticmethod
    def encode(values: list) -> 'DictEncoded':
        """编码值列表。"""
        de = DictEncoded()
        for v in values:
            if v not in de._reverse:
                code = len(de.dictionary)
                de.dictionary.append(v)
                de._reverse[v] = code
            de.codes.append(de._reverse[v])
        return de

    def decode_all(self) -> list:
        """全量解码。"""
        return [self.dictionary[c] for c in self.codes]

    def decode(self, index: int) -> Any:
        return self.dictionary[self.codes[index]]

    @property
    def ndv(self) -> int:
        """不同值数量。"""
        return len(self.dictionary)

    def __len__(self) -> int:
        return len(self.codes)

    def lookup_code(self, value: Any) -> Optional[int]:
        """查找值对应的编码。None = 不在字典中。"""
        return self._reverse.get(value)

    def filter_eq(self, value: Any) -> list:
        """等值过滤，返回匹配行索引。直接在编码上操作，O(n) 但常数极小。"""
        code = self._reverse.get(value)
        if code is None:
            return []
        return [i for i, c in enumerate(self.codes)
                if c == code]

    def filter_eq_bitmap(self, value: Any,
                         size: int) -> bytearray:
        """等值过滤，返回位图。"""
        bm = bytearray((size + 7) // 8)
        code = self._reverse.get(value)
        if code is None:
            return bm
        for i, c in enumerate(self.codes):
            if c == code:
                bm[i >> 3] |= (1 << (i & 7))
        return bm

    def remap_codes(self, code_map: dict) -> list:
        """编码重映射（GROUP BY 字典编码列时使用）。"""
        return [code_map.get(c, c) for c in self.codes]

    def compression_ratio(self) -> float:
        """估算压缩率。"""
        if not self.dictionary:
            return 1.0
        avg_len = sum(len(str(v)) for v in self.dictionary) / len(self.dictionary)
        compressed = len(self.codes) * 2 + avg_len * len(self.dictionary)
        original = avg_len * len(self.codes)
        return compressed / original if original > 0 else 1.0
