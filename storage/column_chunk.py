from __future__ import annotations
"""Column chunk — the basic unit of columnar storage.
Auto-creates dictionary encoding for low-cardinality VARCHAR columns."""
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


class ColumnChunk:
    """Stores data for a single column within a chunk group."""

    def __init__(self, dtype: DataType, max_capacity: int = CHUNK_SIZE) -> None:
        self.dtype = dtype
        self.max_capacity = max_capacity
        self.row_count = 0
        self.null_bitmap = Bitmap(0)
        self.zone_map: dict = {'min': None, 'max': None, 'null_count': 0}
        self.compression = 'NONE'
        self.dict_encoded: object = None  # DictEncoded for VARCHAR, built lazily

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
            if isinstance(self.data, TypedVector):
                self.data.append(0)
            elif isinstance(self.data, list):
                self.data.append('')
            elif isinstance(self.data, Bitmap):
                self.data.ensure_capacity(idx + 1)
        else:
            if isinstance(self.data, TypedVector):
                self.data.append(value)
            elif isinstance(self.data, list):
                self.data.append(str(value))
            elif isinstance(self.data, Bitmap):
                self.data.ensure_capacity(idx + 1)
                if value:
                    self.data.set_bit(idx)

        self.row_count += 1
        self._update_zone_map(value)

    def _update_zone_map(self, value: object) -> None:
        if value is None:
            self.zone_map['null_count'] += 1
            return
        cur_min = self.zone_map['min']
        cur_max = self.zone_map['max']
        try:
            if cur_min is None or value < cur_min:
                self.zone_map['min'] = value
            if cur_max is None or value > cur_max:
                self.zone_map['max'] = value
        except TypeError:
            pass

    def get(self, row_idx: int) -> object:
        if self.null_bitmap.get_bit(row_idx):
            return None
        if self.dtype == DataType.BOOLEAN:
            assert isinstance(self.data, Bitmap)
            return self.data.get_bit(row_idx)
        if isinstance(self.data, TypedVector):
            return self.data[row_idx]
        if isinstance(self.data, list):
            return self.data[row_idx]
        return None

    def build_dict_encoding(self) -> None:
        """Build dictionary encoding for VARCHAR/TEXT columns if beneficial.
        Called after chunk is full or before query execution."""
        if not _HAS_DICT:
            return
        if self.dtype not in (DataType.VARCHAR, DataType.TEXT):
            return
        if not isinstance(self.data, list) or self.row_count == 0:
            return
        # Only build if NDV < 65536 and NDV < row_count / 2
        distinct = set()
        for i in range(self.row_count):
            if not self.null_bitmap.get_bit(i):
                distinct.add(self.data[i])
                if len(distinct) > 65535:
                    return  # Too many distinct values
        if len(distinct) >= self.row_count // 2:
            return  # Not worth it
        # Build dictionary
        non_null_values = []
        for i in range(self.row_count):
            if self.null_bitmap.get_bit(i):
                non_null_values.append('')
            else:
                non_null_values.append(self.data[i])
        self.dict_encoded = DictEncoded.encode(non_null_values)
