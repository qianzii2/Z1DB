from __future__ import annotations
from tests.conftest import *
from storage.column_chunk import ColumnChunk
from storage.types import DataType

def test_basic_append_get():
    cc = ColumnChunk(DataType.INT)
    cc.append(10); cc.append(20); cc.append(None); cc.append(40)
    assert cc.get(0) == 10
    assert cc.get(1) == 20
    assert cc.get(2) is None
    assert cc.get(3) == 40

def test_varchar():
    cc = ColumnChunk(DataType.VARCHAR)
    cc.append('hello'); cc.append(None); cc.append('world')
    assert cc.get(0) == 'hello'
    assert cc.get(1) is None
    assert cc.get(2) == 'world'

def test_boolean():
    cc = ColumnChunk(DataType.BOOLEAN)
    cc.append(True); cc.append(False); cc.append(None)
    assert cc.get(0) == True
    assert cc.get(1) == False
    assert cc.get(2) is None

def test_zone_map():
    cc = ColumnChunk(DataType.INT)
    cc.append(30); cc.append(10); cc.append(None); cc.append(50)
    assert cc.zone_map['min'] == 10
    assert cc.zone_map['max'] == 50
    assert cc.zone_map['null_count'] == 1

def test_dict_encoding():
    cc = ColumnChunk(DataType.VARCHAR)
    for s in ['apple', 'banana', 'apple', 'cherry', 'banana']:
        cc.append(s)
    cc.build_dict_encoding()
    assert cc.dict_encoded is not None
    assert cc.dict_encoded.ndv == 3

def test_compress_decompress():
    """压缩后释放原始数据，get() 仍能读取。"""
    cc = ColumnChunk(DataType.INT)
    original = list(range(100))  # 排序 → DELTA
    for v in original:
        cc.append(v)
    cc.compress()
    if cc._compression_type != 'NONE':
        cc.release_raw_data()
        for i, v in enumerate(original):
            assert cc.get(i) == v, f"行 {i}: {cc.get(i)} != {v}"

def test_compress_with_nulls():
    cc = ColumnChunk(DataType.INT)
    data = [1, None, 2, None, 3, None, 4, None, 5]
    for v in data:
        cc.append(v)
    cc.compress()
    for i, v in enumerate(data):
        assert cc.get(i) == v

def test_rle_compress():
    cc = ColumnChunk(DataType.INT)
    data = [1]*50 + [2]*30 + [3]*20
    for v in data:
        cc.append(v)
    cc.compress()
    # 验证读取
    for i, v in enumerate(data):
        assert cc.get(i) == v

def test_release_and_read():
    cc = ColumnChunk(DataType.INT)
    data = [10, 20, 30, 40, 50]
    for v in data:
        cc.append(v)
    cc.compress()
    if cc._compression_type != 'NONE':
        assert cc.release_raw_data() == True
        assert cc.get(0) == 10
        assert cc.get(4) == 50

def test_string_compare():
    cc = ColumnChunk(DataType.VARCHAR)
    cc.append('banana'); cc.append('apple'); cc.append('cherry')
    assert cc.compare_strings(0, 1) > 0  # banana > apple
    assert cc.compare_strings(1, 2) < 0  # apple < cherry

def test_prefix_match():
    cc = ColumnChunk(DataType.VARCHAR)
    cc.append('hello world'); cc.append('help me'); cc.append('goodbye')
    assert cc.prefix_match(0, 'hel') == True
    assert cc.prefix_match(2, 'hel') == False
