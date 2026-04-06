from __future__ import annotations
"""FSST — 快速静态符号表字符串压缩。
论文: Boncz et al., 2020
贪心选择 256 个最高频子串 (1-8字节) 作为符号。
每个符号用 1 字节编码。解压 = 查表（极快）。"""
from typing import Dict, List, Tuple


class SymbolTable:
    """FSST 符号表 — 频繁子串 → 1 字节编码映射。"""

    __slots__ = ('_symbols', '_codes', '_max_len')

    def __init__(self) -> None:
        self._symbols: List[bytes] = []    # code → 符号字节
        self._codes: Dict[bytes, int] = {} # 符号 → code
        self._max_len = 0

    @staticmethod
    def train(strings: List[str],
              max_symbols: int = 255) -> 'SymbolTable':
        """从样本字符串训练符号表。贪心选择收益最高的子串。"""
        st = SymbolTable()
        # 统计 1-8 字节子串频率
        freq: Dict[bytes, int] = {}
        for s in strings:
            encoded = s.encode('utf-8')
            for length in range(1, min(9, len(encoded) + 1)):
                for start in range(len(encoded) - length + 1):
                    sub = encoded[start:start + length]
                    freq[sub] = freq.get(sub, 0) + 1

        # 收益 = 频率 × (长度-1)：每次命中节省 (长度-1) 字节
        scored: List[Tuple[float, bytes]] = []
        for sub, count in freq.items():
            gain = count * (len(sub) - 1)
            if gain > 0:
                scored.append((gain, sub))
        scored.sort(key=lambda x: -x[0])

        # 贪心选取
        used_code = 0
        for _, sub in scored:
            if used_code >= max_symbols:
                break
            st._symbols.append(sub)
            st._codes[sub] = used_code
            used_code += 1
            if len(sub) > st._max_len:
                st._max_len = len(sub)

        # 确保所有单字节都有编码（转义路径）
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
    """用 FSST 符号表编码字符串。贪心最长匹配。"""
    encoded = text.encode('utf-8')
    result = bytearray()
    i = 0
    while i < len(encoded):
        best_len = 0
        best_code = -1
        for length in range(
                min(table._max_len, len(encoded) - i), 0, -1):
            sub = encoded[i:i + length]
            if sub in table._codes:
                best_len = length
                best_code = table._codes[sub]
                break
        if best_code >= 0:
            result.append(best_code)
            i += best_len
        else:
            # 转义：0xFF + 原始字节
            result.append(0xFF)
            result.append(encoded[i])
            i += 1
    return bytes(result)


def fsst_decode(data: bytes, table: SymbolTable) -> str:
    """解码 FSST 压缩数据。查表操作，极快。"""
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


def fsst_compression_ratio(strings: List[str],
                           table: SymbolTable) -> float:
    original = sum(len(s.encode('utf-8')) for s in strings)
    compressed = sum(len(fsst_encode(s, table)) for s in strings)
    return compressed / original if original > 0 else 1.0
