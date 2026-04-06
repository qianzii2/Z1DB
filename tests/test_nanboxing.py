from __future__ import annotations
from tests.conftest import *

def test_nan_pack_unpack_int():
    from metal.bitmagic import nan_pack_int, nan_unpack
    val = 42
    packed = nan_pack_int(val)
    tag, unpacked = nan_unpack(packed)
    assert unpacked == val

def test_nan_pack_unpack_float():
    from metal.bitmagic import nan_pack_float, nan_unpack
    val = 3.14
    packed = nan_pack_float(val)
    tag, unpacked = nan_unpack(packed)
    assert abs(unpacked - val) < 1e-10

def test_nan_pack_null():
    from metal.bitmagic import nan_pack_null, nan_unpack, NULL_TAG
    packed = nan_pack_null()
    assert packed == NULL_TAG
    tag, val = nan_unpack(packed)
    assert val is None

def test_nan_pack_bool():
    from metal.bitmagic import nan_pack_bool, nan_unpack
    packed_true = nan_pack_bool(True)
    packed_false = nan_pack_bool(False)
    _, val_true = nan_unpack(packed_true)
    _, val_false = nan_unpack(packed_false)
    assert val_true == True
    assert val_false == False

def test_nan_batch_add():
    from metal.bitmagic import nan_pack_int, nanbox_batch_add
    import array as _array
    a = _array.array('Q', [nan_pack_int(1), nan_pack_int(2), nan_pack_int(3)])
    b = _array.array('Q', [nan_pack_int(10), nan_pack_int(20), nan_pack_int(30)])
    result, nulls = nanbox_batch_add(a, b, 3, is_float=False)
    from metal.bitmagic import nan_unpack
    _, v0 = nan_unpack(result[0])
    _, v1 = nan_unpack(result[1])
    _, v2 = nan_unpack(result[2])
    assert v0 == 11 and v1 == 22 and v2 == 33

def test_nan_batch_sub():
    from metal.bitmagic import nan_pack_int, nanbox_batch_sub, nan_unpack
    import array as _array
    a = _array.array('Q', [nan_pack_int(10), nan_pack_int(20)])
    b = _array.array('Q', [nan_pack_int(3), nan_pack_int(5)])
    result, nulls = nanbox_batch_sub(a, b, 2, is_float=False)
    _, v0 = nan_unpack(result[0])
    _, v1 = nan_unpack(result[1])
    assert v0 == 7 and v1 == 15

def test_nan_batch_mul():
    from metal.bitmagic import nan_pack_int, nanbox_batch_mul, nan_unpack
    import array as _array
    a = _array.array('Q', [nan_pack_int(2), nan_pack_int(3)])
    b = _array.array('Q', [nan_pack_int(4), nan_pack_int(5)])
    result, nulls = nanbox_batch_mul(a, b, 2, is_float=False)
    _, v0 = nan_unpack(result[0])
    _, v1 = nan_unpack(result[1])
    assert v0 == 8 and v1 == 15

def test_nan_int_range():
    """INT 32 位范围测试。"""
    from metal.bitmagic import nan_pack_int, nan_unpack
    # 最大 32 位有符号整数
    max_int = 2147483647
    min_int = -2147483648
    packed_max = nan_pack_int(max_int)
    packed_min = nan_pack_int(min_int)
    _, v_max = nan_unpack(packed_max)
    _, v_min = nan_unpack(packed_min)
    assert v_max == max_int
    assert v_min == min_int

def test_nan_float_precision():
    """FLOAT 精度测试。"""
    from metal.bitmagic import nan_pack_float, nan_unpack
    values = [0.0, 1.5, -3.14, 1e10, 1e-10]
    for v in values:
        packed = nan_pack_float(v)
        _, unpacked = nan_unpack(packed)
        assert abs(unpacked - v) < 1e-6 or abs(unpacked - v) / abs(v) < 1e-6
