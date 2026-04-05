from __future__ import annotations
"""Tests for metal/ layer."""


def test_raw_memory():
    from metal.memory import RawMemoryBlock
    b = RawMemoryBlock('q', 8)
    b.append(42); b.append(-1)
    assert b.size == 2
    assert b.get(0) == 42 and b.get(1) == -1
    b.batch_append([10, 20, 30])
    assert b.get_batch(2, 3) == [10, 20, 30]
    for i in range(100): b.append(i)
    assert b.size == 105
    print("    RawMemoryBlock ✓")

def test_arena():
    from metal.arena import Arena
    a = Arena()
    a.alloc(100); a.alloc(200)
    assert a.bytes_used() > 0
    a.reset()
    assert a.bytes_used() == 0
    print("    Arena ✓")

def test_slab():
    from metal.slab import SlabAllocator
    s = SlabAllocator(16, 10)
    ids = [s.alloc() for _ in range(10)]
    s.write(ids[0], b'hello world!!!!!'); assert s.read(ids[0])[:5] == b'hello'
    s.free(ids[0]); assert s.size == 9
    for _ in range(20): s.alloc()
    assert s.size == 29
    print("    Slab ✓")

def test_swar():
    from metal.swar import batch_to_upper, batch_to_lower, has_zero_byte, pack_bytes, unpack_bytes, to_upper_ascii_8
    assert batch_to_upper(bytearray(b'hello')) == bytearray(b'HELLO')
    assert batch_to_lower(bytearray(b'WORLD')) == bytearray(b'world')
    word = pack_bytes(b'abcdefgh')
    upper = to_upper_ascii_8(word)
    assert unpack_bytes(upper, 8) == b'ABCDEFGH'
    assert batch_to_upper(bytearray(b'Test 123!')) == bytearray(b'TEST 123!')
    print("    SWAR ✓")

def test_bitmap():
    from metal.bitmap import Bitmap
    bm = Bitmap(100)
    bm.set_bit(0); bm.set_bit(50); bm.set_bit(99)
    assert bm.get_bit(0) and bm.get_bit(50) and not bm.get_bit(1)
    assert bm.popcount() == 3
    assert bm.to_indices() == [0, 50, 99]
    bm2 = Bitmap(100); bm2.set_bit(50); bm2.set_bit(51)
    assert bm.and_op(bm2).to_indices() == [50]
    print("    Bitmap ✓")

def test_branchless():
    from metal.branchless import min_int, max_int, abs_int, sign
    assert min_int(3, 5) == 3 and min_int(5, 3) == 3
    assert max_int(3, 5) == 5
    assert abs_int(-42) == 42 and abs_int(42) == 42
    assert sign(-3) == -1 and sign(0) == 0 and sign(7) == 1
    print("    Branchless ✓")

def test_hash():
    from metal.hash import murmur3_64, fibonacci_hash, hash_value
    h1 = murmur3_64(b'hello')
    h2 = murmur3_64(b'hello')
    assert h1 == h2 and h1 != 0
    assert hash_value(None, 'q') == 0
    assert hash_value(42, 'q') != 0
    print("    Hash ✓")

def test_bitmagic():
    from metal.bitmagic import nan_pack_int, nan_pack_null, nan_unpack, pdep, pext, select64
    assert nan_unpack(nan_pack_int(42)) == ('INT', 42)
    assert nan_unpack(nan_pack_null()) == ('NULL', None)
    assert pdep(0b1011, 0b11010100) == 0b10010100
    assert pext(0b10010100, 0b11010100) == 0b1011
    assert select64(0b10110100, 0) == 2
    print("    BitMagic ✓")

def test_inline_string():
    from metal.inline_string import InlineStringStore
    store = InlineStringStore(10)
    i0 = store.append("hello"); i1 = store.append("a very long string overflow")
    assert store.get(i0) == "hello"
    assert store.get(i1) == "a very long string overflow"
    assert store.compare(i0, i1) != 0
    print("    InlineString ✓")

def test_advanced_hash():
    from metal.advanced_hash import ZobristHasher, CuckooHashMap, siphash_2_4
    zh = ZobristHasher(3)
    h1 = zh.hash_row([1, "a", 3])
    h2 = zh.update_hash(h1, 2, 3, 4)
    assert h2 == zh.hash_row([1, "a", 4])
    cm = CuckooHashMap(32)
    for i in range(50): cm.put(i, f"v{i}")
    for i in range(50): assert cm.get(i) == f"v{i}"
    assert siphash_2_4(b'test') != 0
    print("    AdvancedHash ✓")


ALL_TESTS = [
    ("RawMemoryBlock", test_raw_memory),
    ("Arena", test_arena),
    ("Slab", test_slab),
    ("SWAR", test_swar),
    ("Bitmap", test_bitmap),
    ("Branchless", test_branchless),
    ("Hash", test_hash),
    ("BitMagic", test_bitmagic),
    ("InlineString", test_inline_string),
    ("AdvancedHash", test_advanced_hash),
]
