from __future__ import annotations
"""Tests for structures/ layer."""


def test_bloom_filter():
    from structures.bloom_filter import BloomFilter
    bf = BloomFilter(1000, 0.01)
    for i in range(100): bf.add(i)
    for i in range(100): assert bf.contains(i)
    fp = sum(1 for i in range(100, 200) if bf.contains(i))
    assert fp < 15
    print(f"    BloomFilter ✓ (FP={fp}/100)")

def test_robin_hood():
    from structures.robin_hood_ht import RobinHoodHashTable
    ht = RobinHoodHashTable(16)
    for i in range(100): ht.put(i, f"v{i}")
    assert ht.size == 100
    for i in range(100): assert ht.get(i) == (True, f"v{i}")
    assert not ht.get(999)[0]
    ht.remove(50); assert not ht.get(50)[0]
    print("    RobinHood ✓")

def test_roaring():
    from structures.roaring_bitmap import RoaringBitmap
    rb = RoaringBitmap()
    for i in range(0, 1000, 3): rb.add(i)
    assert rb.contains(0) and rb.contains(999) and not rb.contains(1)
    rb2 = RoaringBitmap()
    for i in range(0, 1000, 5): rb2.add(i)
    assert rb.and_op(rb2).contains(15) and not rb.and_op(rb2).contains(3)
    print("    Roaring ✓")

def test_segment_tree():
    from structures.segment_tree import SegmentTree, MinSegmentTree
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    st = SegmentTree(data); assert st.query(0, 7) == sum(data)
    mst = MinSegmentTree(data); assert mst.query(0, 7) == 1
    print("    SegmentTree ✓")

def test_sparse_table():
    from structures.sparse_table import SparseTableMin, SparseTableMax
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    assert SparseTableMin(data).query(0, 7) == 1
    assert SparseTableMax(data).query(0, 7) == 9
    print("    SparseTable ✓")

def test_fenwick():
    from structures.fenwick_tree import FenwickTree
    ft = FenwickTree.from_list([1, 2, 3, 4, 5])
    assert ft.prefix_sum(4) == 15
    assert ft.range_sum(1, 3) == 9
    print("    Fenwick ✓")

def test_loser_tree():
    from structures.tournament_tree import LoserTree
    result = LoserTree([iter([1, 4, 7]), iter([2, 5, 8]), iter([3, 6, 9])]).merge_all()
    assert result == list(range(1, 10))
    print("    LoserTree ✓")

def test_xor_filter():
    from structures.xor_filter import XorFilter
    keys = list(range(100))
    xf = XorFilter(keys)
    for k in keys: assert xf.contains(k)
    fp = sum(1 for i in range(100, 200) if xf.contains(i))
    assert fp < 15
    print(f"    XorFilter ✓ (FP={fp})")

def test_cuckoo_filter():
    from structures.cuckoo_filter import CuckooFilter
    cf = CuckooFilter(256)
    for i in range(50): cf.add(i)
    for i in range(50): assert cf.contains(i)
    cf.delete(25); assert not cf.contains(25)
    print("    CuckooFilter ✓")

def test_wavelet_tree():
    from structures.wavelet_tree import WaveletTree
    wt = WaveletTree([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], sigma=10)
    assert wt.quantile(0, 5, 2) == 3
    assert wt.rank(0, 5, 1) == 2
    print("    WaveletTree ✓")

def test_art():
    from structures.art import AdaptiveRadixTree
    art = AdaptiveRadixTree()
    for i, w in enumerate(['apple', 'app', 'banana']):
        art.insert(w.encode(), i)
    assert art.search(b'apple') == 0
    assert len(art.prefix_scan(b'app')) == 2
    print("    ART ✓")

def test_skip_list():
    from structures.skip_list import SkipList
    sl = SkipList()
    for i in range(50): sl.insert(i, f"v{i}")
    assert sl.search(25) == "v25" and sl.search(999) is None
    assert len(sl.range_query(10, 20)) == 11
    print("    SkipList ✓")

def test_perfect_hash():
    from structures.perfect_hash import PerfectHashMap
    phm = PerfectHashMap(list(range(20)), [f"v{i}" for i in range(20)])
    for i in range(20): assert phm.get(i) == f"v{i}"
    assert phm.get(999) is None
    print("    PerfectHash ✓")

def test_sorted_container():
    from structures.sorted_container import SortedList
    sl = SortedList()
    for v in [5, 3, 8, 1, 9, 2]: sl.add(v)
    assert sl.kth(0) == 1 and sl.kth(5) == 9
    sl.remove(5); assert sl.size == 5
    print("    SortedContainer ✓")


ALL_TESTS = [
    ("BloomFilter", test_bloom_filter),
    ("RobinHood", test_robin_hood),
    ("Roaring", test_roaring),
    ("SegmentTree", test_segment_tree),
    ("SparseTable", test_sparse_table),
    ("Fenwick", test_fenwick),
    ("LoserTree", test_loser_tree),
    ("XorFilter", test_xor_filter),
    ("CuckooFilter", test_cuckoo_filter),
    ("WaveletTree", test_wavelet_tree),
    ("ART", test_art),
    ("SkipList", test_skip_list),
    ("PerfectHash", test_perfect_hash),
    ("SortedContainer", test_sorted_container),
]
