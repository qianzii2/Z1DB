from __future__ import annotations
"""列存储块 — 分段 LRU 解压缓存。
每个 ColumnChunk 存储一列的一个 chunk（最多 CHUNK_SIZE 行）。
支持：字典编码、7 种压缩算法、ZoneMap 裁剪。
读取压缩数据时按 SEGMENT_SIZE 分段解压，LRU 缓存热段。"""
from collections import OrderedDict
from metal.bitmap import Bitmap
from metal.config import CHUNK_SIZE, SEGMENT_SIZE, MAX_CACHED_SEGMENTS
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType
from utils.errors import ExecutionError

try:
    from storage.compression.dict_codec import DictEncoded
    _HAS_DICT = True
except ImportError:
    _HAS_DICT = False
try:
    from storage.compression.rle import rle_encode, rle_decode
    _HAS_RLE = True
except ImportError:
    _HAS_RLE = False
try:
    from storage.compression.delta import delta_encode, delta_decode
    _HAS_DELTA = True
except ImportError:
    _HAS_DELTA = False
try:
    from storage.compression.bitpack import for_encode, for_decode
    _HAS_FOR = True
except ImportError:
    _HAS_FOR = False
try:
    from storage.compression.gorilla import gorilla_encode, gorilla_decode
    _HAS_GORILLA = True
except ImportError:
    _HAS_GORILLA = False
try:
    from storage.compression.alp import alp_encode, alp_decode
    _HAS_ALP = True
except ImportError:
    _HAS_ALP = False
try:
    from storage.compression.analyzer import analyze_and_choose
    _HAS_ANALYZER = True
except ImportError:
    _HAS_ANALYZER = False
try:
    from storage.compression.fsst import SymbolTable, fsst_encode, fsst_decode
    _HAS_FSST = True
except ImportError:
    _HAS_FSST = False
try:
    from metal.inline_string import InlineStringStore
    _HAS_INLINE = True
except ImportError:
    _HAS_INLINE = False


class ColumnChunk:
    """一列的一个 chunk。最多 max_capacity 行。
    写入时维护 ZoneMap (min/max/null_count)。
    chunk 满后可触发字典编码和压缩。"""

    def __init__(self, dtype: DataType,
                 max_capacity: int = CHUNK_SIZE) -> None:
        self.dtype = dtype
        self.max_capacity = max_capacity
        self.row_count = 0
        self.null_bitmap = Bitmap(0)
        self.zone_map: dict = {
            'min': None, 'max': None, 'null_count': 0}
        self.dict_encoded: object = None
        self._dict_build_attempted = False
        self._compress_attempted = False
        self._compressed_data: object = None
        self._compression_type = 'NONE'
        self._fsst_table: object = None
        self._raw_released = False
        # 分段 LRU 缓存：seg_id → list[values]
        self._segment_cache: OrderedDict[int, list] = OrderedDict()
        # 全量解压缓存（回退路径）
        self._full_decompressed: list = None
        # 前缀非 NULL 计数数组（release_raw_data 后构建）
        self._non_null_prefix: list = None

        # 按类型初始化数据存储
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            self.data: TypedVector | list | Bitmap | InlineStringStore = (
                TypedVector(code))
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            if _HAS_INLINE:
                self.data = InlineStringStore(capacity=max_capacity)
            else:
                self.data = []
        elif dtype == DataType.BOOLEAN:
            self.data = Bitmap(0)
        else:
            raise ExecutionError(f"不支持的类型: {dtype.name}")

    # ═══ 写入 ═══

    def append(self, value: object) -> None:
        """追加一个值。自动维护 null_bitmap 和 zone_map。"""
        if self.row_count >= self.max_capacity:
            raise ExecutionError("chunk 已满")
        if self._raw_released:
            raise ExecutionError("chunk 已释放原始数据")
        idx = self.row_count
        self.null_bitmap.ensure_capacity(idx + 1)
        if value is None:
            self.null_bitmap.set_bit(idx)
            # NULL 占位
            if isinstance(self.data, TypedVector):
                self.data.append(0)
            elif _HAS_INLINE and isinstance(self.data, InlineStringStore):
                self.data.append('')
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

    # ═══ 读取 ═══

    def get(self, row_idx: int) -> object:
        """读取指定行的值。NULL 返回 None。"""
        if self.null_bitmap.get_bit(row_idx):
            return None
        # 原始数据未释放 → 直接读取
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
        # 原始数据已释放 → 从压缩数据分段解压
        return self._get_from_segment(row_idx)

    def _get_from_segment(self, row_idx: int) -> object:
        """从压缩数据分段解压读取。LRU 缓存热段。"""
        non_null_pos = self._count_non_null_before(row_idx)
        seg_id = non_null_pos // SEGMENT_SIZE

        # 查 LRU 缓存
        if seg_id in self._segment_cache:
            self._segment_cache.move_to_end(seg_id)
            segment = self._segment_cache[seg_id]
            offset = non_null_pos % SEGMENT_SIZE
            return segment[offset] if offset < len(segment) else None

        # 定点解码快速路径（RLE/FOR/DELTA 支持不解压全部直接取值）
        point_val = self._try_point_decode(non_null_pos)
        if point_val is not None:
            return point_val

        # 全量解压 → 切段 → 缓存
        if self._full_decompressed is None:
            self._full_decompressed = self._decompress_all()
        if self._full_decompressed is None:
            return None

        seg_start = seg_id * SEGMENT_SIZE
        seg_end = min(seg_start + SEGMENT_SIZE,
                      len(self._full_decompressed))
        segment = self._full_decompressed[seg_start:seg_end]
        self._cache_segment(seg_id, segment)

        offset = non_null_pos % SEGMENT_SIZE
        return segment[offset] if offset < len(segment) else None

    def _cache_segment(self, seg_id: int,
                       segment: list) -> None:
        """LRU 缓存段，淘汰最旧段。"""
        self._segment_cache[seg_id] = segment
        self._segment_cache.move_to_end(seg_id)
        while len(self._segment_cache) > MAX_CACHED_SEGMENTS:
            self._segment_cache.popitem(last=False)

    def _count_non_null_before(self, row_idx: int) -> int:
        """计算 [0, row_idx) 中非 NULL 行数。
        有前缀数组时 O(1)，否则 O(n) 回退。"""
        if self._non_null_prefix:
            return self._non_null_prefix[row_idx]
        count = 0
        for i in range(row_idx):
            if not self.null_bitmap.get_bit(i):
                count += 1
        return count

    def _try_point_decode(self, non_null_pos: int) -> object:
        """RLE/FOR/DELTA 定点解码，无需解压全部。"""
        ct = self._compression_type
        if ct == 'RLE' and _HAS_RLE and self._compressed_data:
            _, rv, rl = self._compressed_data
            offset = 0
            for v, l in zip(rv, rl):
                if non_null_pos < offset + l:
                    return v
                offset += l
        if ct == 'FOR' and _HAS_FOR and self._compressed_data:
            _, min_val, packed, count, bw = self._compressed_data
            if non_null_pos < count:
                bit_pos = non_null_pos * bw
                val = 0
                for b in range(bw):
                    byte_idx = (bit_pos + b) >> 3
                    bit_idx = (bit_pos + b) & 7
                    if (byte_idx < len(packed)
                            and packed[byte_idx] & (1 << bit_idx)):
                        val |= (1 << b)
                return val + min_val
        if ct == 'DELTA' and _HAS_DELTA and self._compressed_data:
            _, base, deltas = self._compressed_data
            if non_null_pos < len(deltas):
                val = base
                for i in range(1, non_null_pos + 1):
                    if i < len(deltas):
                        val += deltas[i]
                return val
        return None

    def _decompress_all(self) -> list:
        """全量解压。回退路径。"""
        if self._compressed_data is None:
            return None
        ct = self._compression_type
        try:
            if ct == 'RLE' and _HAS_RLE:
                _, rv, rl = self._compressed_data
                return rle_decode(rv, rl)
            if ct == 'DELTA' and _HAS_DELTA:
                _, base, deltas = self._compressed_data
                return delta_decode(base, deltas)
            if ct == 'FOR' and _HAS_FOR:
                _, min_val, packed, count, bw = self._compressed_data
                return for_decode(min_val, packed, count, bw)
            if ct == 'GORILLA' and _HAS_GORILLA:
                _, eb = self._compressed_data
                return gorilla_decode(eb)
            if ct == 'ALP' and _HAS_ALP:
                _, eb = self._compressed_data
                return alp_decode(eb)
        except Exception:
            pass
        if self.dict_encoded is not None and _HAS_DICT:
            return self.dict_encoded.decode_all()
        return None

    # ═══ 压缩管理 ═══

    def release_raw_data(self) -> bool:
        """释放原始数据，仅保留压缩/字典编码数据。
        释放后读取走分段解压路径。"""
        if (self._compression_type == 'NONE'
                and self.dict_encoded is None):
            return False
        if (self._compressed_data is None
                and self.dict_encoded is None):
            return False
        self.data = None
        self._raw_released = True
        self._segment_cache.clear()
        self._full_decompressed = None
        self._non_null_prefix = self._build_non_null_prefix()
        return True

    def _build_non_null_prefix(self) -> list:
        """O(n) 构建前缀非 NULL 计数数组。prefix[i] = [0,i) 中非 NULL 行数。"""
        prefix = [0] * (self.row_count + 1)
        for i in range(self.row_count):
            prefix[i + 1] = prefix[i] + (
                0 if self.null_bitmap.get_bit(i) else 1)
        return prefix

    def build_dict_encoding(self):
        """构建字典编码。NDV 过高或已尝试过则跳过。"""
        if self._dict_build_attempted or self._raw_released:
            return
        self._dict_build_attempted = True
        if not _HAS_DICT:
            return
        if self.dtype not in (DataType.VARCHAR, DataType.TEXT):
            return
        if self.row_count == 0:
            return
        distinct = set()
        for i in range(self.row_count):
            if not self.null_bitmap.get_bit(i):
                distinct.add(self.get(i))
                if len(distinct) > 65535:
                    return  # NDV 太高不值得
        if len(distinct) >= self.row_count // 2:
            return  # 压缩率不够
        non_null = ['' if self.null_bitmap.get_bit(i) else self.get(i)
                    for i in range(self.row_count)]
        self.dict_encoded = DictEncoded.encode(non_null)

    def compress(self):
        """分析数据并选择最优压缩算法。"""
        if (self._compress_attempted or self.row_count == 0
                or self._raw_released):
            return
        self._compress_attempted = True
        if not _HAS_ANALYZER:
            return
        raw = self._extract_non_null_values()
        if not raw:
            return
        codec = analyze_and_choose(raw, self.dtype)
        if codec == 'NONE':
            if (self.dtype in (DataType.VARCHAR, DataType.TEXT)
                    and _HAS_FSST and len(raw) > 100):
                self._try_fsst(raw)
            return
        try:
            if codec == 'RLE' and _HAS_RLE:
                rv, rl = rle_encode(raw)
                self._compressed_data = ('RLE', rv, rl)
                self._compression_type = 'RLE'
            elif codec == 'DELTA' and _HAS_DELTA:
                base, deltas = delta_encode(raw)
                self._compressed_data = ('DELTA', base, deltas)
                self._compression_type = 'DELTA'
            elif codec == 'FOR' and _HAS_FOR:
                min_val, packed, count, bw = for_encode(raw)
                self._compressed_data = (
                    'FOR', min_val, packed, count, bw)
                self._compression_type = 'FOR'
            elif codec == 'GORILLA' and _HAS_GORILLA:
                self._compressed_data = (
                    'GORILLA', gorilla_encode(raw))
                self._compression_type = 'GORILLA'
            elif codec == 'ALP' and _HAS_ALP:
                self._compressed_data = (
                    'ALP', alp_encode([float(v) for v in raw]))
                self._compression_type = 'ALP'
            elif codec == 'DICT':
                self.build_dict_encoding()
        except Exception:
            self._compression_type = 'NONE'
            self._compressed_data = None

    def _try_fsst(self, values):
        """尝试 FSST 字符串压缩。压缩率 < 70% 才启用。"""
        try:
            str_vals = [str(v) for v in values]
            sample = str_vals[:min(1000, len(str_vals))]
            table = SymbolTable.train(sample)
            original = sum(len(s.encode('utf-8')) for s in sample)
            compressed = sum(
                len(fsst_encode(s, table)) for s in sample)
            if original > 0 and compressed / original < 0.7:
                self._fsst_table = table
                self._compression_type = 'FSST'
        except Exception:
            pass

    # ═══ 字符串比较（InlineStringStore 加速）═══

    def compare_strings(self, i, j):
        """比较两行的字符串值。InlineStringStore 用 4 字节前缀快速比较。"""
        if (_HAS_INLINE
                and isinstance(self.data, InlineStringStore)
                and not self._raw_released):
            try:
                return self.data.compare(i, j)
            except Exception:
                pass
        a = self.get(i)
        b = self.get(j)
        if a is None and b is None:
            return 0
        if a is None:
            return -1
        if b is None:
            return 1
        return -1 if a < b else (1 if a > b else 0)

    def prefix_match(self, index, prefix):
        """前缀匹配。InlineStringStore 时只读 slot 内前缀。"""
        if (_HAS_INLINE
                and isinstance(self.data, InlineStringStore)
                and not self._raw_released):
            try:
                return self.data.prefix_equals(
                    index, prefix.encode('utf-8'))
            except Exception:
                pass
        val = self.get(index)
        return str(val).startswith(prefix) if val is not None else False

    # ═══ ZoneMap ═══

    def _update_zone_map(self, value):
        if value is None:
            self.zone_map['null_count'] += 1
            return
        try:
            if (self.zone_map['min'] is None
                    or value < self.zone_map['min']):
                self.zone_map['min'] = value
            if (self.zone_map['max'] is None
                    or value > self.zone_map['max']):
                self.zone_map['max'] = value
        except TypeError:
            pass

    # ═══ 压缩数据访问 ═══

    def get_compressed_data(self):
        return self._compressed_data

    def get_rle_data(self):
        if (self._compression_type == 'RLE'
                and self._compressed_data):
            return (self._compressed_data[1],
                    self._compressed_data[2])
        return None

    def get_alp_data(self):
        if (self._compression_type == 'ALP'
                and self._compressed_data):
            return self._compressed_data[1]
        return None

    def _extract_non_null_values(self):
        return [self.get(i) for i in range(self.row_count)
                if not self.null_bitmap.get_bit(i)]
