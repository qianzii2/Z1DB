"""Z1DB 可选特性注册与查询。

在程序启动时检测特性是否可用，其他模块通过此集中接口引用。
"""

from typing import Dict, Optional, Callable

# ═══ 特性变量初始化 ═══

_HAS_BLOOM = False
_HAS_ROARING = False
_HAS_ART = False
_HAS_SKIP = False
_HAS_ROBIN_HOOD = False
_HAS_CUCKOO = False
_HAS_RIBBON = False
_HAS_XOR = False
_HAS_FENWICK = False
_HAS_SEGTREE = False
_HAS_SPARSE = False
_HAS_WAVELET = False
_HAS_SORTED = False
_HAS_LOSER = False
_HAS_ARENA = False
_HAS_SLAB = False
_HAS_INLINE = False
_HAS_NANBOX = False
_HAS_NANBOX_BATCH = False
_HAS_JIT = False
_HAS_LAZY = False
_HAS_MORSEL = False
_HAS_BM = False
_HAS_COMPILED_DATE = False
_HAS_VEC_OPS = False
_HAS_SIMILARITY = False
_HAS_PLANNER = False
_HAS_DPCCP = False
_HAS_RUNTIME_OPT = False
_HAS_TOPN = False
_HAS_NL = False
_HAS_RADIX = False
_HAS_GRACE = False
_HAS_SMJ = False
_HAS_INDEX_SCAN = False
_HAS_PARALLEL = False
_HAS_ZONEMAP_SCAN = False
_HAS_COMPRESSED = False
_HAS_EXTERNAL = False
_HAS_COORDINATOR = False
_HAS_INDEX = False
_HAS_BUFFER_POOL = False
_HAS_DELTA_STORE = False
_HAS_MERGE_WORKER = False
_HAS_SWIPE = False  # Add or remove feature flags as needed


class FeatureRegistry:
    """管理所有特性的注册与查询。"""

    def __init__(self) -> None:
        self._features: Dict[str, bool] = {}

    def register(self, name: str, available: bool) -> None:
        self._features[name] = available

    def is_available(self, name: str) -> bool:
        return self._features.get(name, False)


FEATURES = FeatureRegistry()

# 逐项检测
try:
    from structures.bloom_filter import BloomFilter
    _HAS_BLOOM = True
except ImportError:
    pass
FEATURES.register('BLOOM', _HAS_BLOOM)

try:
    from structures.roaring_bitmap import RoaringBitmap
    _HAS_ROARING = True
except ImportError:
    pass
FEATURES.register('ROARING', _HAS_ROARING)

# ... 依此类推，注册所有其他特性 ...

# 示例查询函数
def has_feature(name: str) -> bool:
    """检查特性是否可用。"""
    return FEATURES.is_available(name)

def require_feature(name: str) -> None:
    """要求某个特性可用，否则抛异常。"""
    if not has_feature(name):
        raise ImportError(f"特性 {name} 不可用")

# 方便其他模块从此导入
HAS_BLOOM = _HAS_BLOOM
HAS_ROARING = _HAS_ROARING
# ... 其他 HAS_ 前缀变量

# 使用示例:
# from metal.features import HAS_BLOOM
# if HAS_BLOOM:
#     bf = BloomFilter(...)
