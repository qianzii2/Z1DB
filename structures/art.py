from __future__ import annotations
"""自适应基数树 (ART) — 按数据密度自动选择节点大小 (4/16/48/256)。
论文: Leis et al., 2013 "The Adaptive Radix Tree"
用于 VARCHAR 列索引和前缀查询。O(k) 查找（k = 键长度）。"""
from typing import Any, Iterator, List, Optional, Tuple


class _Node4:
    """最多 4 个子节点 — 线性查找。"""
    __slots__ = ('keys', 'children', 'count', 'value', 'has_value')
    def __init__(self) -> None:
        self.keys = [0] * 4
        self.children: list = [None] * 4
        self.count = 0
        self.value: Any = None
        self.has_value = False


class _Node16:
    """5-16 个子节点 — 线性查找（可用 SIMD 加速）。"""
    __slots__ = ('keys', 'children', 'count', 'value', 'has_value')
    def __init__(self) -> None:
        self.keys = [0] * 16
        self.children: list = [None] * 16
        self.count = 0
        self.value: Any = None
        self.has_value = False


class _Node48:
    """17-48 个子节点 — 按字节值间接索引。"""
    __slots__ = ('index', 'children', 'count', 'value', 'has_value')
    def __init__(self) -> None:
        self.index = [-1] * 256      # byte → 子节点位置
        self.children: list = [None] * 48
        self.count = 0
        self.value: Any = None
        self.has_value = False


class _Node256:
    """49-256 个子节点 — 直接按字节值索引。"""
    __slots__ = ('children', 'count', 'value', 'has_value')
    def __init__(self) -> None:
        self.children: list = [None] * 256
        self.count = 0
        self.value: Any = None
        self.has_value = False


class AdaptiveRadixTree:
    """ART 自适应基数树。节点类型随子节点数量自动升级：
    4 → 16 → 48 → 256。"""

    __slots__ = ('_root', '_size')

    def __init__(self) -> None:
        self._root: Any = None
        self._size = 0

    def insert(self, key: bytes, value: Any) -> None:
        """插入键值对。key 为字节序列。"""
        self._root = self._insert(self._root, key, 0, value)
        self._size += 1

    def search(self, key: bytes) -> Optional[Any]:
        """查找。O(k) 其中 k = len(key)。"""
        return self._search(self._root, key, 0)

    def prefix_scan(self, prefix: bytes) -> List[Tuple[bytes, Any]]:
        """前缀扫描：返回所有以 prefix 开头的 (key, value) 对。"""
        results: list = []
        node = self._find_prefix_node(self._root, prefix, 0)
        if node is not None:
            self._collect(node, list(prefix), results)
        return results

    @property
    def size(self) -> int:
        return self._size

    # ═══ 内部方法 ═══

    def _insert(self, node, key, depth, value):
        if node is None:
            node = _Node4()
        if depth == len(key):
            node.value = value
            node.has_value = True
            return node
        byte = key[depth]
        child = self._find_child(node, byte)
        if child is not None:
            new_child = self._insert(child, key, depth + 1, value)
            self._set_child(node, byte, new_child)
        else:
            new_child = self._insert(None, key, depth + 1, value)
            node = self._add_child(node, byte, new_child)
        return node

    def _search(self, node, key, depth):
        if node is None:
            return None
        if depth == len(key):
            return node.value if node.has_value else None
        child = self._find_child(node, key[depth])
        if child is None:
            return None
        return self._search(child, key, depth + 1)

    def _find_prefix_node(self, node, prefix, depth):
        if node is None:
            return None
        if depth == len(prefix):
            return node
        child = self._find_child(node, prefix[depth])
        if child is None:
            return None
        return self._find_prefix_node(child, prefix, depth + 1)

    def _collect(self, node, path, results):
        if node is None:
            return
        if node.has_value:
            results.append((bytes(path), node.value))
        for byte in range(256):
            child = self._find_child(node, byte)
            if child is not None:
                path.append(byte)
                self._collect(child, path, results)
                path.pop()

    def _find_child(self, node, byte):
        if isinstance(node, _Node4):
            for i in range(node.count):
                if node.keys[i] == byte:
                    return node.children[i]
        elif isinstance(node, _Node16):
            for i in range(node.count):
                if node.keys[i] == byte:
                    return node.children[i]
        elif isinstance(node, _Node48):
            idx = node.index[byte]
            if idx >= 0:
                return node.children[idx]
        elif isinstance(node, _Node256):
            return node.children[byte]
        return None

    def _set_child(self, node, byte, child):
        if isinstance(node, _Node4):
            for i in range(node.count):
                if node.keys[i] == byte:
                    node.children[i] = child
                    return
        elif isinstance(node, _Node16):
            for i in range(node.count):
                if node.keys[i] == byte:
                    node.children[i] = child
                    return
        elif isinstance(node, _Node48):
            idx = node.index[byte]
            if idx >= 0:
                node.children[idx] = child
        elif isinstance(node, _Node256):
            node.children[byte] = child

    def _add_child(self, node, byte, child):
        """添加子节点，必要时升级节点类型。"""
        if isinstance(node, _Node4):
            if node.count < 4:
                node.keys[node.count] = byte
                node.children[node.count] = child
                node.count += 1
                return node
            return self._grow_to_16(node, byte, child)
        if isinstance(node, _Node16):
            if node.count < 16:
                node.keys[node.count] = byte
                node.children[node.count] = child
                node.count += 1
                return node
            return self._grow_to_48(node, byte, child)
        if isinstance(node, _Node48):
            if node.count < 48:
                node.index[byte] = node.count
                node.children[node.count] = child
                node.count += 1
                return node
            return self._grow_to_256(node, byte, child)
        if isinstance(node, _Node256):
            node.children[byte] = child
            node.count += 1
            return node
        return node

    # ═══ 节点升级 ═══

    def _grow_to_16(self, n4, byte, child):
        n16 = _Node16()
        n16.value = n4.value
        n16.has_value = n4.has_value
        for i in range(4):
            n16.keys[i] = n4.keys[i]
            n16.children[i] = n4.children[i]
        n16.keys[4] = byte
        n16.children[4] = child
        n16.count = 5
        return n16

    def _grow_to_48(self, n16, byte, child):
        n48 = _Node48()
        n48.value = n16.value
        n48.has_value = n16.has_value
        for i in range(16):
            n48.index[n16.keys[i]] = i
            n48.children[i] = n16.children[i]
        n48.index[byte] = 16
        n48.children[16] = child
        n48.count = 17
        return n48

    def _grow_to_256(self, n48, byte, child):
        n256 = _Node256()
        n256.value = n48.value
        n256.has_value = n48.has_value
        for b in range(256):
            idx = n48.index[b]
            if idx >= 0:
                n256.children[b] = n48.children[idx]
                n256.count += 1
        n256.children[byte] = child
        n256.count += 1
        return n256
