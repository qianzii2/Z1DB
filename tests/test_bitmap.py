from __future__ import annotations
from tests.conftest import *

def test_bitmap_set_get():
    from metal.bitmap import Bitmap
    bm = Bitmap(100)
    bm.set_bit(10)
    bm.set_bit(50)
    assert bm.get_bit(10) == True
    assert bm.get_bit(50) == True
    assert bm.get_bit(25) == False

def test_bitmap_clear():
    from metal.bitmap import Bitmap
    bm = Bitmap(100)
    bm.set_bit(10)
    assert bm.get_bit(10) == True
    bm.clear_bit(10)
    assert bm.get_bit(10) == False

def test_bitmap_popcount():
    from metal.bitmap import Bitmap
    bm = Bitmap(100)
    for i in [10, 20, 30, 40, 50]:
        bm.set_bit(i)
    assert bm.popcount() == 5

def test_bitmap_to_indices():
    from metal.bitmap import Bitmap
    bm = Bitmap(100)
    for i in [5, 15, 25, 35]:
        bm.set_bit(i)
    indices = bm.to_indices()
    assert indices == [5, 15, 25, 35]

def test_bitmap_and():
    from metal.bitmap import Bitmap
    bm1 = Bitmap(100)
    bm2 = Bitmap(100)
    for i in [10, 20, 30]:
        bm1.set_bit(i)
    for i in [20, 30, 40]:
        bm2.set_bit(i)
    result = bm1.and_op(bm2)
    assert result.popcount() == 2
    assert result.get_bit(20) == True
    assert result.get_bit(30) == True

def test_bitmap_or():
    from metal.bitmap import Bitmap
    bm1 = Bitmap(100)
    bm2 = Bitmap(100)
    for i in [10, 20]:
        bm1.set_bit(i)
    for i in [30, 40]:
        bm2.set_bit(i)
    result = bm1.or_op(bm2)
    assert result.popcount() == 4

def test_bitmap_not():
    from metal.bitmap import Bitmap
    bm = Bitmap(10)
    bm.set_bit(0)
    bm.set_bit(5)
    result = bm.not_op()
    assert result.get_bit(0) == False
    assert result.get_bit(1) == True
    assert result.get_bit(5) == False

def test_bitmap_pooled():
    from metal.bitmap import Bitmap
    bm1 = Bitmap.pooled(100)
    bm1.set_bit(10)
    Bitmap.recycle(bm1)
    # 回收后原对象应被标记为不可用
    assert bm1.get_bit(10) == False

def test_bitmap_copy():
    from metal.bitmap import Bitmap
    bm1 = Bitmap(100)
    bm1.set_bit(10)
    bm1.set_bit(20)
    bm2 = bm1.copy()
    assert bm2.get_bit(10) == True
    assert bm2.get_bit(20) == True
    bm2.clear_bit(10)
    assert bm1.get_bit(10) == True  # 原对象不受影响

def test_bitmap_select():
    from metal.bitmap import Bitmap
    bm = Bitmap(100)
    for i in [5, 15, 25, 35, 45]:
        bm.set_bit(i)
    # select(k) 返回第 k 个设置位的位置
    assert bm.select(0) == 5
    assert bm.select(2) == 25
    assert bm.select(4) == 45

def test_bitmap_gather():
    from metal.bitmap import Bitmap
    bm = Bitmap(10)
    bm.set_bit(1)
    bm.set_bit(3)
    bm.set_bit(5)
    data = list(range(10))
    values = bm.gather_values(data)
    assert values == [1, 3, 5]

def test_bitmap_append_from():
    from metal.bitmap import Bitmap
    bm1 = Bitmap(10)
    bm1.set_bit(2)
    bm1.set_bit(5)
    bm2 = Bitmap(10)
    bm2.set_bit(1)
    bm2.set_bit(3)
    bm1.append_from(bm2, 10)
    assert bm1.size == 20
    assert bm1.get_bit(2) == True
    assert bm1.get_bit(11) == True  # bm2 的位 1 → bm1 的位 11
