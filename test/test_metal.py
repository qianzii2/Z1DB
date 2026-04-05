from __future__ import annotations
"""Layer 0 — 裸金属层测试。"""
import struct
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRawMemoryBlock:
    def test_append_get(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('q', 4)
        b.append(42); b.append(-1); b.append(0)
        assert b.size == 3
        assert b.get(0) == 42
        assert b.get(1) == -1
        assert b.get(2) == 0

    def test_batch_append(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('q', 4)
        b.batch_append([10, 20, 30, 40])
        assert b.size == 4
        assert b.get_batch(0, 4) == [10, 20, 30, 40]

    def test_auto_grow(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('q', 2)
        for i in range(100):
            b.append(i)
        assert b.size == 100
        for i in range(100):
            assert b.get(i) == i

    def test_get_slice(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('i', 8)
        b.batch_append([1, 2, 3, 4])
        mv = b.get_slice(0, 4)
        assert isinstance(mv, memoryview)

    def test_float64(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('d', 4)
        b.append(3.14); b.append(-2.718)
        assert abs(b.get(0) - 3.14) < 1e-10
        assert abs(b.get(1) - (-2.718)) < 1e-10

    def test_set(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('q', 4)
        b.append(0); b.append(0)
        b.set(0, 99); b.set(1, -99)
        assert b.get(0) == 99
        assert b.get(1) == -99

    def test_empty(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('q', 4)
        assert b.size == 0
        assert b.get_batch(0, 0) == []

    def test_index_error(self):
        from metal.memory import RawMemoryBlock
        b = RawMemoryBlock('q', 4)
        b.append(1)
        try:
            b.get(5)
            assert False, "should raise"
        except IndexError:
            pass


class TestArena:
    def test_alloc_reset(self):
        from metal.arena import Arena
        a = Arena()
        buf1, off1 = a.alloc(100)
        buf2, off2 = a.alloc(200)
        assert a.bytes_used() > 0
        a.reset()
        assert a.bytes_used() == 0

    def test_large_alloc(self):
        from metal.arena import Arena
        a = Arena()
        buf, off = a.alloc(2 * 1024 * 1024)  # 2MB > BLOCK_SIZE
        assert buf is not None

    def test_alignment(self):
        from metal.arena import Arena
        a = Arena()
        _, off1 = a.alloc(3)
        _, off2 = a.alloc(5)
        assert off2 % 8 == 0  # 8-byte aligned

    def test_write(self):
        from metal.arena import Arena
        a = Arena()
        buf, off = a.alloc_and_write(b'hello world')
        import ctypes
        raw = bytes(buf[off:off + 11])
        assert raw == b'hello world'


class TestSlab:
    def test_alloc_free(self):
        from metal.slab import SlabAllocator
        s = SlabAllocator(object_size=16, slab_capacity=10)
        ids = [s.alloc() for _ in range(10)]
        assert s.size == 10
        s.free(ids[0])
        assert s.size == 9
        new_id = s.alloc()
        assert s.size == 10

    def test_write_read(self):
        from metal.slab import SlabAllocator
        s = SlabAllocator(object_size=16)
        slot = s.alloc()
        s.write(slot, b'hello world 1234')
        assert s.read(slot) == b'hello world 1234'

    def test_grow(self):
        from metal.slab import SlabAllocator
        s = SlabAllocator(object_size=8, slab_capacity=4)
        for _ in range(20):
            s.alloc()
        assert s.size == 20


class TestInlineString:
    def test_inline(self):
        from metal.inline_string import InlineStringStore
        store = InlineStringStore(capacity=10)
        i0 = store.append("hello")
        assert store.get(i0) == "hello"

    def test_overflow(self):
        from metal.inline_string import InlineStringStore
        store = InlineStringStore(capacity=10)
        long_str = "this is a very long string that overflows the 12-byte inline limit"
        i0 = store.append(long_str)
        assert store.get(i0) == long_str

    def test_compare(self):
        from metal.inline_string import InlineStringStore
        store = InlineStringStore(capacity=10)
        i0 = store.append("apple")
        i1 = store.append("banana")
        assert store.compare(i0, i1) < 0
        assert store.compare(i1, i0) > 0
        i2 = store.append("apple")
        assert store.compare(i0, i2) == 0

    def test_prefix_equals(self):
        from metal.inline_string import InlineStringStore
        store = InlineStringStore(capacity=10)
        i0 = store.append("hello world")
        assert store.prefix_equals(i0, b'hel')
        assert not store.prefix_equals(i0, b'xyz')

    def test_grow(self):
        from metal.inline_string import InlineStringStore
        store = InlineStringStore(capacity=2)
        for i in range(20):
            store.append(f"string_{i}")
        assert len(store) == 20

    def test_empty_string(self):
        from metal.inline_string import InlineStringStore
        store = InlineStringStore()
        i0 = store.append("")
        assert store.get(i0) == ""


class TestSWAR:
    def test_to_upper(self):
        from metal.swar import to_upper_ascii_8, pack_bytes, unpack_bytes
        word = pack_bytes(b'hello wo')
        upper = to_upper_ascii_8(word)
        assert unpack_bytes(upper, 8) == b'HELLO WO'

    def test_to_lower(self):
        from metal.swar import to_lower_ascii_8, pack_bytes, unpack_bytes
        word = pack_bytes(b'HELLO WO')
        lower = to_lower_ascii_8(word)
        assert unpack_bytes(lower, 8) == b'hello wo'

    def test_batch_to_upper(self):
        from metal.swar import batch_to_upper
        result = batch_to_upper(bytearray(b'Hello World 123'))
        assert result == bytearray(b'HELLO WORLD 123')

    def test_batch_to_lower(self):
        from metal.swar import batch_to_lower
        result = batch_to_lower(bytearray(b'Hello World 123'))
        assert result == bytearray(b'hello world 123')

    def test_has_zero_byte(self):
        from metal.swar import has_zero_byte, pack_bytes
        assert has_zero_byte(pack_bytes(b'hel\x00o wo'))
        assert not has_zero_byte(pack_bytes(b'hello wo'))

    def test_has_byte_equal_to(self):
        from metal.swar import has_byte_equal_to, pack_bytes
        word = pack_bytes(b'hello wo')
        assert has_byte_equal_to(word, ord('l'))
        assert not has_byte_equal_to(word, ord('z'))

    def test_find_byte(self):
        from metal.swar import find_byte, pack_bytes
        word = pack_bytes(b'hello wo')
        assert find_byte(word, ord('e')) == 1
        assert find_byte(word, ord('z')) == -1

    def test_non_ascii_fallback(self):
        from metal.swar import batch_to_upper
        # Non-ASCII should pass through unchanged
        data = bytearray(b'abc\x80\x90xyz')
        result = batch_to_upper(data)
        assert result[0:3] == bytearray(b'ABC')
        assert result[3] == 0x80
        assert result[5:8] == bytearray(b'XYZ')


class TestBitMagic:
    def test_nan_boxing_int(self):
        from metal.bitmagic import nan_pack_int, nan_unpack, nan_is_int
        bits = nan_pack_int(42)
        assert nan_is_int(bits)
        typ, val = nan_unpack(bits)
        assert typ == 'INT' and val == 42

    def test_nan_boxing_negative(self):
        from metal.bitmagic import nan_pack_int, nan_unpack
        bits = nan_pack_int(-1)
        typ, val = nan_unpack(bits)
        assert typ == 'INT' and val == -1

    def test_nan_boxing_null(self):
        from metal.bitmagic import nan_pack_null, nan_unpack, nan_is_null
        bits = nan_pack_null()
        assert nan_is_null(bits)
        assert nan_unpack(bits) == ('NULL', None)

    def test_nan_boxing_bool(self):
        from metal.bitmagic import nan_pack_bool, nan_unpack
        assert nan_unpack(nan_pack_bool(True)) == ('BOOL', True)
        assert nan_unpack(nan_pack_bool(False)) == ('BOOL', False)

    def test_nan_boxing_float(self):
        from metal.bitmagic import nan_pack_float, nan_unpack
        bits = nan_pack_float(3.14)
        typ, val = nan_unpack(bits)
        assert typ == 'FLOAT' and abs(val - 3.14) < 1e-10

    def test_pdep(self):
        from metal.bitmagic import pdep
        assert pdep(0b1011, 0b11010100) == 0b10010100

    def test_pext(self):
        from metal.bitmagic import pext
        assert pext(0b10010100, 0b11010100) == 0b1011

    def test_select64(self):
        from metal.bitmagic import select64
        word = 0b10110100
        assert select64(word, 0) == 2
        assert select64(word, 1) == 4
        assert select64(word, 2) == 5

    def test_rank64(self):
        from metal.bitmagic import rank64
        word = 0b10110100
        assert rank64(word, 5) == 2


class TestBranchless:
    def test_min_max(self):
        from metal.branchless import min_int, max_int
        assert min_int(3, 5) == 3
        assert min_int(5, 3) == 3
        assert max_int(3, 5) == 5
        assert max_int(-10, 10) == 10

    def test_abs(self):
        from metal.branchless import abs_int
        assert abs_int(-42) == 42
        assert abs_int(42) == 42
        assert abs_int(0) == 0

    def test_clamp(self):
        from metal.branchless import clamp
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_sign(self):
        from metal.branchless import sign
        assert sign(-3) == -1
        assert sign(0) == 0
        assert sign(7) == 1


class TestAdvancedHash:
    def test_zobrist(self):
        from metal.advanced_hash import ZobristHasher
        zh = ZobristHasher(3)
        h1 = zh.hash_row([1, "Alice", 30])
        h2 = zh.update_hash(h1, 2, 30, 31)
        h3 = zh.hash_row([1, "Alice", 31])
        assert h2 == h3

    def test_cuckoo(self):
        from metal.advanced_hash import CuckooHashMap
        cm = CuckooHashMap(32)
        for i in range(50):
            cm.put(i, f"val_{i}")
        for i in range(50):
            assert cm.get(i) == f"val_{i}"
        assert cm.get(999) is None
        cm.remove(25)
        assert cm.get(25) is None

    def test_siphash(self):
        from metal.advanced_hash import siphash_2_4
        h = siphash_2_4(b'hello', 0, 0)
        assert isinstance(h, int) and h != 0
        # Deterministic
        assert siphash_2_4(b'hello', 0, 0) == h

    def test_write_combining(self):
        from metal.advanced_hash import WriteCombiningBuffer
        wcb = WriteCombiningBuffer(4)
        for i in range(40):
            wcb.write(i % 4, i)
        wcb.flush_all()
        assert len(wcb.get_partition(0)) == 10
        assert len(wcb.get_partition(1)) == 10


class TestBitmap:
    def test_basic(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(10)
        bm.set_bit(3); bm.set_bit(7)
        assert bm.get_bit(3)
        assert not bm.get_bit(4)
        assert bm.get_bit(7)

    def test_popcount(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(16)
        bm.set_bit(0); bm.set_bit(5); bm.set_bit(10)
        assert bm.popcount() == 3

    def test_and_or_not(self):
        from metal.bitmap import Bitmap
        a = Bitmap(8); a.set_bit(1); a.set_bit(3); a.set_bit(5)
        b = Bitmap(8); b.set_bit(3); b.set_bit(5); b.set_bit(7)
        c = a.and_op(b)
        assert c.get_bit(3) and c.get_bit(5)
        assert not c.get_bit(1) and not c.get_bit(7)
        d = a.or_op(b)
        assert d.get_bit(1) and d.get_bit(3) and d.get_bit(5) and d.get_bit(7)

    def test_to_indices(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(10)
        bm.set_bit(2); bm.set_bit(5); bm.set_bit(8)
        assert bm.to_indices() == [2, 5, 8]

    def test_dynamic_growth(self):
        from metal.bitmap import Bitmap
        bm = Bitmap(0)
        bm.set_bit(100)
        assert bm.get_bit(100)
        assert bm.size >= 101


class TestBitwise:
    def test_clz64(self):
        from metal.bitwise import clz64
        assert clz64(0) == 64
        assert clz64(1) == 63

    def test_ctz64(self):
        from metal.bitwise import ctz64
        assert ctz64(0) == 64
        assert ctz64(1) == 0
        assert ctz64(8) == 3

    def test_popcount64(self):
        from metal.bitwise import popcount64
        assert popcount64(0) == 0
        assert popcount64(0xFF) == 8
        assert popcount64(0b1010101) == 4

    def test_next_power_of_2(self):
        from metal.bitwise import next_power_of_2
        assert next_power_of_2(1) == 1
        assert next_power_of_2(5) == 8
        assert next_power_of_2(16) == 16


def run_metal_tests():
    classes = [TestRawMemoryBlock, TestArena, TestSlab, TestInlineString,
               TestSWAR, TestBitMagic, TestBranchless, TestAdvancedHash,
               TestBitmap, TestBitwise]
    total = 0; passed = 0; failed = 0
    for cls in classes:
        obj = cls()
        for method_name in dir(obj):
            if not method_name.startswith('test_'):
                continue
            total += 1
            try:
                getattr(obj, method_name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {cls.__name__}.{method_name}: {e}")
    print(f"\nMetal: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Metal Layer Tests ===")
    run_metal_tests()
