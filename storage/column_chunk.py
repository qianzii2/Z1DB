from __future__ import annotations
"""列存储块 — 分段 LRU 解压缓存。
[P10] 压缩数据按 1024 行一段解压，LRU 淘汰旧段。
大 chunk (65536行) 只需 ~64 段，内存占用可控。"""
from collections import OrderedDict
from metal.bitmap import Bitmap
from metal.config import CHUNK_SIZE
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType
from utils.errors import ExecutionError

try:
    from storage.compression.dict_codec import DictEncoded
    _HAS_DICT = True
except ImportError: _HAS_DICT = False
try:
    from storage.compression.rle import rle_encode, rle_decode
    _HAS_RLE = True
except ImportError: _HAS_RLE = False
try:
    from storage.compression.delta import delta_encode, delta_decode
    _HAS_DELTA = True
except ImportError: _HAS_DELTA = False
try:
    from storage.compression.bitpack import for_encode, for_decode
    _HAS_FOR = True
except ImportError: _HAS_FOR = False
try:
    from storage.compression.gorilla import gorilla_encode, gorilla_decode
    _HAS_GORILLA = True
except ImportError: _HAS_GORILLA = False
try:
    from storage.compression.alp import alp_encode, alp_decode
    _HAS_ALP = True
except ImportError: _HAS_ALP = False
try:
    from storage.compression.analyzer import analyze_and_choose
    _HAS_ANALYZER = True
except ImportError: _HAS_ANALYZER = False
try:
    from storage.compression.fsst import SymbolTable, fsst_encode, fsst_decode
    _HAS_FSST = True
except ImportError: _HAS_FSST = False
try:
    from metal.inline_string import InlineStringStore
    _HAS_INLINE = True
except ImportError: _HAS_INLINE = False

# [P10] 分段大小和 LRU 容量
_SEGMENT_SIZE = 1024
_MAX_CACHED_SEGMENTS = 8  # 每个 chunk 最多缓存 8 段 = 8K 行


class ColumnChunk:
    def __init__(self, dtype: DataType,
                 max_capacity: int = CHUNK_SIZE) -> None:
        self.dtype = dtype
        self.max_capacity = max_capacity
        self.row_count = 0
        self.null_bitmap = Bitmap(0)
        self.zone_map: dict = {'min': None, 'max': None, 'null_count': 0}
        self.dict_encoded: object = None
        self._dict_build_attempted = False
        self._compress_attempted = False
        self._compressed_data: object = None
        self._compression_type = 'NONE'
        self._fsst_table: object = None
        self._raw_released = False
        # [P10] 分段 LRU 缓存：seg_id → list[values]
        self._segment_cache: OrderedDict[int, list] = OrderedDict()
        # 全量解压缓存（兼容旧路径）
        self._full_decompressed: list = None

        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            self.data: TypedVector | list | Bitmap | InlineStringStore = TypedVector(code)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            if _HAS_INLINE:
                self.data = InlineStringStore(capacity=max_capacity)
            else:
                self.data = []
        elif dtype == DataType.BOOLEAN:
            self.data = Bitmap(0)
        else:
            raise ExecutionError(f"不支持的类型: {dtype.name}")

    def append(self, value: object) -> None:
        if self.row_count >= self.max_capacity:
            raise ExecutionError("chunk 已满")
        if self._raw_released:
            raise ExecutionError("chunk 已释放原始数据")
        idx = self.row_count
        self.null_bitmap.ensure_capacity(idx + 1)
        if value is None:
            self.null_bitmap.set_bit(idx)
            if isinstance(self.data, TypedVector):
                self.data.append(0)
            elif _HAS_INLINE and isinstance(self.data, InlineStringStore):
                self.data.append('')  # NULL 占位
            elif isinstance(self.data, list):
                self.data.append('')
            elif isinstance(self.data, Bitmap):
                self.data.ensure_capacity(idx + 1)
        else:
            if isinstance(self.data, TypedVector):
                self.data.append(value)
            elif _HAS_INLINE and isinstance(self.data, InlineStringStore):
                self.data.append(str(value))
            elif isinstance(self.data, list):
                self.data.append(str(value))
            elif isinstance(self.data, Bitmap):
                self.data.ensure_capacity(idx + 1)
                if value:
                    self.data.set_bit(idx)
        self.row_count += 1
        self._update_zone_map(value)

    def get(self, row_idx: int) -> object:
        if self.null_bitmap.get_bit(row_idx):
            return None
        if not self._raw_released:
            if self.dtype == DataType.BOOLEAN:
                return self.data.get_bit(row_idx)
            if _HAS_INLINE and isinstance(self.data, InlineStringStore):
                return self.data.get(row_idx)
            if isinstance(self.data, TypedVector):
                return self.data[row_idx]
            if isinstance(self.data, list):
                return self.data[row_idx] if row_idx < len(self.data) else None
            return None
        return self._get_from_segment(row_idx)

    def _get_from_segment(self, row_idx: int) -> object:
        """[P10] 分段解压：按 _SEGMENT_SIZE 分段，LRU 缓存。"""
        # 计算此行在非 NULL 序列中的位置
        non_null_pos = self._count_non_null_before(row_idx)
        seg_id = non_null_pos // _SEGMENT_SIZE

        # 查 LRU 缓存
        if seg_id in self._segment_cache:
            self._segment_cache.move_to_end(seg_id)
            segment = self._segment_cache[seg_id]
            offset = non_null_pos % _SEGMENT_SIZE
            return segment[offset] if offset < len(segment) else None

        # 定点解码快速路径（RLE/FOR/DELTA）
        point_val = self._try_point_decode(non_null_pos)
        if point_val is not None:
            return point_val

        # 全量解压 → 切段 → 缓存
        if self._full_decompressed is None:
            self._full_decompressed = self._decompress_all()
        if self._full_decompressed is None:
            return None

        # 切出所请求的段并缓存
        seg_start = seg_id * _SEGMENT_SIZE
        seg_end = min(seg_start + _SEGMENT_SIZE, len(self._full_decompressed))
        segment = self._full_decompressed[seg_start:seg_end]
        self._cache_segment(seg_id, segment)

        offset = non_null_pos % _SEGMENT_SIZE
        return segment[offset] if offset < len(segment) else None

    def _cache_segment(self, seg_id: int, segment: list) -> None:
        """LRU 缓存段。"""
        self._segment_cache[seg_id] = segment
        self._segment_cache.move_to_end(seg_id)
        # 淘汰最旧段
        while len(self._segment_cache) > _MAX_CACHED_SEGMENTS:
            self._segment_cache.popitem(last=False)

    def _count_non_null_before(self, row_idx: int) -> int:
        """计算 [0, row_idx) 中非 NULL 行数 + 当前行是否非 NULL。"""
        count = 0
        for i in range(row_idx):
            if not self.null_bitmap.get_bit(i):
                count += 1
        return count

    def _try_point_decode(self, non_null_pos: int) -> object:
        """RLE/FOR/DELTA 定点解码。"""
        ct = self._compression_type
        if ct == 'RLE' and _HAS_RLE and self._compressed_data:
            _, rv, rl = self._compressed_data
            offset = 0
            for v, l in zip(rv, rl):
                if non_null_pos < offset + l: return v
                offset += l
        if ct == 'FOR' and _HAS_FOR and self._compressed_data:
            _, min_val, packed, count, bw = self._compressed_data
            if non_null_pos < count:
                bit_pos = non_null_pos * bw; val = 0
                for b in range(bw):
                    byte_idx = (bit_pos + b) >> 3
                    bit_idx = (bit_pos + b) & 7
                    if byte_idx < len(packed) and packed[byte_idx] & (1 << bit_idx):
                        val |= (1 << b)
                return val + min_val
        if ct == 'DELTA' and _HAS_DELTA and self._compressed_data:
            _, base, deltas = self._compressed_data
            if non_null_pos < len(deltas):
                val = base
                for i in range(1, non_null_pos + 1):
                    if i < len(deltas): val += deltas[i]
                return val
        return None

    def _decompress_all(self) -> list:
        if self._compressed_data is None: return None
        ct = self._compression_type
        try:
            if ct == 'RLE' and _HAS_RLE:
                _, rv, rl = self._compressed_data; return rle_decode(rv, rl)
            if ct == 'DELTA' and _HAS_DELTA:
                _, base, deltas = self._compressed_data; return delta_decode(base, deltas)
            if ct == 'FOR' and _HAS_FOR:
                _, min_val, packed, count, bw = self._compressed_data
                return for_decode(min_val, packed, count, bw)
            if ct == 'GORILLA' and _HAS_GORILLA:
                _, eb = self._compressed_data; return gorilla_decode(eb)
            if ct == 'ALP' and _HAS_ALP:
                _, eb = self._compressed_data; return alp_decode(eb)
        except Exception: pass
        if self.dict_encoded is not None and _HAS_DICT:
            return self.dict_encoded.decode_all()
        return None

    def release_raw_data(self) -> bool:
        if self._compression_type == 'NONE' and self.dict_encoded is None:
            return False
        if self._compressed_data is None and self.dict_encoded is None:
            return False
        self.data = None
        self._raw_released = True
        self._segment_cache.clear()
        self._full_decompressed = None
        # [修复] 构建前缀非 NULL 计数数组
        self._non_null_prefix = self._build_non_null_prefix()
        return True

    def _build_non_null_prefix(self) -> list:
        """O(n) 构建前缀非 NULL 计数。prefix[i] = [0,i) 中非 NULL 行数。"""
        prefix = [0] * (self.row_count + 1)
        for i in range(self.row_count):
            prefix[i + 1] = prefix[i] + (0 if self.null_bitmap.get_bit(i) else 1)
        return prefix

    def _count_non_null_before(self, row_idx: int) -> int:
        """O(1) 查找。"""
        if hasattr(self, '_non_null_prefix') and self._non_null_prefix:
            return self._non_null_prefix[row_idx]
        # 回退 O(n)
        count = 0
        for i in range(row_idx):
            if not self.null_bitmap.get_bit(i):
                count += 1
        return count

    def compare_strings(self, i, j):
        if _HAS_INLINE and isinstance(self.data, InlineStringStore) and not self._raw_released:
            try: return self.data.compare(i, j)
            except Exception: pass
        a = self.get(i); b = self.get(j)
        if a is None and b is None: return 0
        if a is None: return -1
        if b is None: return 1
        return -1 if a < b else (1 if a > b else 0)

    def prefix_match(self, index, prefix):
        if _HAS_INLINE and isinstance(self.data, InlineStringStore) and not self._raw_released:
            try: return self.data.prefix_equals(index, prefix.encode('utf-8'))
            except Exception: pass
        val = self.get(index)
        return str(val).startswith(prefix) if val is not None else False

    def _update_zone_map(self, value):
        if value is None: self.zone_map['null_count'] += 1; return
        try:
            if self.zone_map['min'] is None or value < self.zone_map['min']: self.zone_map['min'] = value
            if self.zone_map['max'] is None or value > self.zone_map['max']: self.zone_map['max'] = value
        except TypeError: pass

    def build_dict_encoding(self):
        if self._dict_build_attempted or self._raw_released: return
        self._dict_build_attempted = True
        if not _HAS_DICT or self.dtype not in (DataType.VARCHAR, DataType.TEXT): return
        if self.row_count == 0: return
        distinct = set()
        for i in range(self.row_count):
            if not self.null_bitmap.get_bit(i):
                distinct.add(self.get(i))
                if len(distinct) > 65535: return
        if len(distinct) >= self.row_count // 2: return
        non_null = ['' if self.null_bitmap.get_bit(i) else self.get(i)
                    for i in range(self.row_count)]
        self.dict_encoded = DictEncoded.encode(non_null)

    def compress(self):
        if self._compress_attempted or self.row_count == 0 or self._raw_released: return
        self._compress_attempted = True
        if not _HAS_ANALYZER: return
        raw = self._extract_non_null_values()
        if not raw: return
        codec = analyze_and_choose(raw, self.dtype)
        if codec == 'NONE':
            if self.dtype in (DataType.VARCHAR, DataType.TEXT) and _HAS_FSST and len(raw) > 100:
                self._try_fsst(raw)
            return
        try:
            if codec == 'RLE' and _HAS_RLE:
                rv, rl = rle_encode(raw)
                self._compressed_data = ('RLE', rv, rl); self._compression_type = 'RLE'
            elif codec == 'DELTA' and _HAS_DELTA:
                base, deltas = delta_encode(raw)
                self._compressed_data = ('DELTA', base, deltas); self._compression_type = 'DELTA'
            elif codec == 'FOR' and _HAS_FOR:
                min_val, packed, count, bw = for_encode(raw)
                self._compressed_data = ('FOR', min_val, packed, count, bw); self._compression_type = 'FOR'
            elif codec == 'GORILLA' and _HAS_GORILLA:
                self._compressed_data = ('GORILLA', gorilla_encode(raw)); self._compression_type = 'GORILLA'
            elif codec == 'ALP' and _HAS_ALP:
                self._compressed_data = ('ALP', alp_encode([float(v) for v in raw])); self._compression_type = 'ALP'
            elif codec == 'DICT': self.build_dict_encoding()
        except Exception: self._compression_type = 'NONE'; self._compressed_data = None

    def _try_fsst(self, values):
        try:
            str_vals = [str(v) for v in values]; sample = str_vals[:min(1000, len(str_vals))]
            table = SymbolTable.train(sample)
            original = sum(len(s.encode('utf-8')) for s in sample)
            compressed = sum(len(fsst_encode(s, table)) for s in sample)
            if original > 0 and compressed / original < 0.7:
                self._fsst_table = table; self._compression_type = 'FSST'
        except Exception: pass

    def get_compressed_data(self): return self._compressed_data
    def get_rle_data(self):
        if self._compression_type == 'RLE' and self._compressed_data:
            return self._compressed_data[1], self._compressed_data[2]
        return None
    def get_alp_data(self):
        if self._compression_type == 'ALP' and self._compressed_data:
            return self._compressed_data[1]
        return None
    def _extract_non_null_values(self):
        return [self.get(i) for i in range(self.row_count) if not self.null_bitmap.get_bit(i)]
