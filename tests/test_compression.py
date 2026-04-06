from __future__ import annotations
from tests.conftest import *

def test_rle_roundtrip():
    from storage.compression.rle import rle_encode, rle_decode
    data = [1,1,1,2,2,3,3,3,3]
    rv, rl = rle_encode(data)
    assert rle_decode(rv, rl) == data

def test_rle_single():
    from storage.compression.rle import rle_encode, rle_decode
    data = [42]
    rv, rl = rle_encode(data)
    assert rle_decode(rv, rl) == data

def test_rle_empty():
    from storage.compression.rle import rle_encode, rle_decode
    rv, rl = rle_encode([])
    assert rle_decode(rv, rl) == []

def test_delta_roundtrip():
    from storage.compression.delta import delta_encode, delta_decode
    data = [10, 20, 30, 40, 50]
    base, deltas = delta_encode(data)
    assert delta_decode(base, deltas) == data

def test_delta_negative():
    from storage.compression.delta import delta_encode, delta_decode
    data = [100, 90, 80, 70]
    base, deltas = delta_encode(data)
    assert delta_decode(base, deltas) == data

def test_for_roundtrip():
    from storage.compression.bitpack import for_encode, for_decode
    data = [100, 105, 110, 103, 108]
    min_val, packed, count, bw = for_encode(data)
    assert for_decode(min_val, packed, count, bw) == data

def test_for_single_value():
    from storage.compression.bitpack import for_encode, for_decode
    data = [42, 42, 42]
    min_val, packed, count, bw = for_encode(data)
    assert for_decode(min_val, packed, count, bw) == data

def test_dict_encoding():
    from storage.compression.dict_codec import DictEncoded
    data = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
    de = DictEncoded.encode(data)
    assert de.ndv == 3
    assert de.decode_all() == data
    assert de.lookup_code('banana') is not None
    assert de.lookup_code('grape') is None

def test_dict_filter():
    from storage.compression.dict_codec import DictEncoded
    data = ['a', 'b', 'a', 'c', 'a']
    de = DictEncoded.encode(data)
    indices = de.filter_eq('a')
    assert indices == [0, 2, 4]

def test_gorilla_roundtrip():
    from storage.compression.gorilla import gorilla_encode, gorilla_decode
    data = [1.0, 1.1, 1.2, 1.3, 1.0, 1.1]
    encoded = gorilla_encode(data)
    decoded = gorilla_decode(encoded)
    assert len(decoded) == len(data)
    for a, b in zip(data, decoded):
        assert abs(a - b) < 1e-10

def test_alp_roundtrip():
    from storage.compression.alp import alp_encode, alp_decode
    data = [3.14, 2.71, 1.41, 1.73, 2.23]
    encoded = alp_encode(data)
    decoded = alp_decode(encoded)
    assert len(decoded) == len(data)
    for a, b in zip(data, decoded):
        assert abs(a - b) < 1e-6

def test_analyzer_choice():
    from storage.compression.analyzer import analyze_and_choose
    from storage.types import DataType
    # 排序整数 → DELTA
    sorted_ints = list(range(100))
    assert analyze_and_choose(sorted_ints, DataType.INT) == 'DELTA'
    # 重复值多 → RLE
    repeated = [1]*50 + [2]*30 + [3]*20
    codec = analyze_and_choose(repeated, DataType.INT)
    assert codec in ('RLE', 'DELTA')  # 取决于阈值
    # 字符串低基数 → DICT
    strings = ['a', 'b', 'c'] * 100
    assert analyze_and_choose(strings, DataType.VARCHAR) == 'DICT'
