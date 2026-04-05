from __future__ import annotations
"""列存储块 — 支持自动压缩选择（RLE/Delta/FOR/Dict）。"""
from metal.bitmap import Bitmap
from metal.config import CHUNK_SIZE
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
    from storage.compression.analyzer import analyze_and_choose
    _HAS_ANALYZER = True
except ImportError:
    _HAS_ANALYZER = False


class ColumnChunk:
    """存储单列在一个chunk group中的数据。支持多种压缩编码。"""

    def __init__(self, dtype: DataType, max_capacity: int = CHUNK_SIZE) -> None:
        self.dtype = dtype
        self.max_capacity = max_capacity
        self.row_count = 0
        self.null_bitmap = Bitmap(0)
        self.zone_map: dict = {'min': None, 'max': None, 'null_count': 0}
        self.compression = 'NONE'
        self.dict_encoded: object = None
        self._dict_build_attempted = False
        self._compress_attempted = False
        # 压缩后数据
        self._compressed_data: object = None
        self._compression_type = 'NONE'

        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            self.data: TypedVector | list | Bitmap = TypedVector(code)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            self.data = []
        elif dtype == DataType.BOOLEAN:
            self.data = Bitmap(0)
        else:
            raise ExecutionError(f"unsupported column type: {dtype.name}")

    def append(self, value: object) -> None:
        if self.row_count >= self.max_capacity:
            raise ExecutionError("chunk full")
        idx = self.row_count
        self.null_bitmap.ensure_capacity(idx + 1)
        if value is None:
            self.null_bitmap.set_bit(idx)
            if isinstance(self.data, TypedVector): self.data.append(0)
            elif isinstance(self.data, list): self.data.append('')
            elif isinstance(self.data, Bitmap): self.data.ensure_capacity(idx + 1)
        else:
            if isinstance(self.data, TypedVector): self.data.append(value)
            elif isinstance(self.data, list): self.data.append(str(value))
            elif isinstance(self.data, Bitmap):
                self.data.ensure_capacity(idx + 1)
                if value: self.data.set_bit(idx)
        self.row_count += 1
        self._update_zone_map(value)

    def _update_zone_map(self, value: object) -> None:
        if value is None:
            self.zone_map['null_count'] += 1; return
        try:
            if self.zone_map['min'] is None or value < self.zone_map['min']:
                self.zone_map['min'] = value
            if self.zone_map['max'] is None or value > self.zone_map['max']:
                self.zone_map['max'] = value
        except TypeError:
            pass

    def get(self, row_idx: int) -> object:
        if self.null_bitmap.get_bit(row_idx): return None
        if self.dtype == DataType.BOOLEAN:
            assert isinstance(self.data, Bitmap)
            return self.data.get_bit(row_idx)
        if isinstance(self.data, TypedVector): return self.data[row_idx]
        if isinstance(self.data, list): return self.data[row_idx]
        return None

    def build_dict_encoding(self) -> None:
        """惰性构建字典编码（只尝试一次）。"""
        if self._dict_build_attempted: return
        self._dict_build_attempted = True
        if not _HAS_DICT: return
        if self.dtype not in (DataType.VARCHAR, DataType.TEXT): return
        if not isinstance(self.data, list) or self.row_count == 0: return
        distinct: set = set()
        for i in range(self.row_count):
            if not self.null_bitmap.get_bit(i):
                distinct.add(self.data[i])
                if len(distinct) > 65535: return
        if len(distinct) >= self.row_count // 2: return
        non_null = ['' if self.null_bitmap.get_bit(i) else self.data[i] for i in range(self.row_count)]
        self.dict_encoded = DictEncoded.encode(non_null)

    def compress(self) -> None:
        """分析数据并选择最优压缩编码。只尝试一次。"""
        if self._compress_attempted or self.row_count == 0: return
        self._compress_attempted = True

        if not _HAS_ANALYZER: return
        raw = self._extract_non_null_values()
        if not raw: return

        codec = analyze_and_choose(raw, self.dtype)
        if codec == 'NONE': return

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
                self._compressed_data = ('FOR', min_val, packed, count, bw)
                self._compression_type = 'FOR'
            elif codec == 'GORILLA' and _HAS_GORILLA:
                encoded = gorilla_encode(raw)
                self._compressed_data = ('GORILLA', encoded)
                self._compression_type = 'GORILLA'
            elif codec == 'DICT':
                self.build_dict_encoding()
        except Exception:
            self._compression_type = 'NONE'
            self._compressed_data = None

    def get_compressed_data(self) -> object:
        """获取压缩后的数据（供compressed_execution使用）。"""
        return self._compressed_data

    def get_rle_data(self) -> object:
        """获取RLE编码数据（供compressed_execution直接聚合）。"""
        if self._compression_type == 'RLE' and self._compressed_data:
            _, rv, rl = self._compressed_data
            return rv, rl
        return None

    def _extract_non_null_values(self) -> list:
        result = []
        for i in range(self.row_count):
            if not self.null_bitmap.get_bit(i):
                result.append(self.get(i))
        return result
