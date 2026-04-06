from __future__ import annotations
"""批处理内存池 — Arena 管理列数据 + Slab 池化 Bitmap。
用法:
    pool = BatchPool()
    packed = pool.alloc_packed(n)      # 分配 uint64 数组
    bitmap = pool.alloc_bitmap(n)      # 取 Bitmap
    pool.release_bitmap(bitmap)        # 归还
    pool.reset()                       # 一次性释放全部
"""
import array as _array
from typing import Any, Dict, List, Optional

try:
    from metal.arena import Arena
    _HAS_ARENA = True
except ImportError:
    _HAS_ARENA = False

try:
    from metal.slab import SlabAllocator
    _HAS_SLAB = True
except ImportError:
    _HAS_SLAB = False

# Bitmap 对象池
_BITMAP_POOL: List['_PooledBitmap'] = []
_BITMAP_POOL_MAX = 256


class BatchPool:
    """批处理级内存管理器。一个算子的 open→close 周期内共享一个 Pool。"""

    __slots__ = ('_arena', '_arena_arrays',
                 '_allocated_bitmaps')

    def __init__(self) -> None:
        self._arena: Optional[Arena] = None
        if _HAS_ARENA:
            try:
                self._arena = Arena()
            except Exception:
                self._arena = None
        self._arena_arrays: List[_array.array] = []
        self._allocated_bitmaps: List[Any] = []

    def alloc_packed(self, n: int) -> _array.array:
        """分配 n 个 uint64 的 packed 数组。"""
        # 注意：Python array.array 无法直接指向 Arena 内存
        # Arena 在此作为 GC 延迟释放的保障
        packed = _array.array('Q', [0] * n)
        self._arena_arrays.append(packed)
        return packed

    def alloc_typed(self, code: str,
                    values: list) -> Any:
        """分配 TypedVector。"""
        from metal.typed_vector import TypedVector
        return TypedVector(code, values)

    def alloc_bitmap(self, size: int) -> 'Bitmap':
        """从池中取 Bitmap。"""
        from metal.bitmap import Bitmap
        global _BITMAP_POOL
        for i, bm in enumerate(_BITMAP_POOL):
            if bm.size >= size:
                _BITMAP_POOL.pop(i)
                bm._data = bytearray((size + 7) // 8)
                bm._logical_size = size
                self._allocated_bitmaps.append(bm)
                return bm
        bm = Bitmap(size)
        self._allocated_bitmaps.append(bm)
        return bm

    def release_bitmap(self, bm: Any) -> None:
        """归还 Bitmap 到池中。"""
        global _BITMAP_POOL
        if len(_BITMAP_POOL) < _BITMAP_POOL_MAX:
            _BITMAP_POOL.append(bm)

    def reset(self) -> None:
        """释放所有 Arena 内存。算子 close 时调用。"""
        if self._arena is not None:
            self._arena.reset()
        self._arena_arrays.clear()
        global _BITMAP_POOL
        for bm in self._allocated_bitmaps:
            if len(_BITMAP_POOL) < _BITMAP_POOL_MAX:
                _BITMAP_POOL.append(bm)
        self._allocated_bitmaps.clear()

    @property
    def arena_bytes_used(self) -> int:
        if self._arena:
            return self._arena.bytes_used()
        return 0

    @property
    def bitmap_pool_size(self) -> int:
        return len(_BITMAP_POOL)


# 全局默认池（轻量级操作用）
_DEFAULT_POOL: Optional[BatchPool] = None


def get_default_pool() -> BatchPool:
    """获取默认全局池。长时间查询应创建独立 Pool。"""
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = BatchPool()
    return _DEFAULT_POOL
