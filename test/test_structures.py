from __future__ import annotations
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRobinHood:
    def test_basic(self):
        from structures.robin_hood_ht import RobinHoodHashTable
        ht = RobinHoodHashTable(16)
        for i in range(100):
            ht.put(i, f"v{i}")
        assert ht.size == 100
        for i in range(100):
            found, val = ht.get(i)
            assert found and val == f"v{i}"

    def test_miss(self):
        from structures.robin_hood_ht import RobinHoodHashTable
        ht = RobinHoodHashTable()
        ht.put(1, 'a')
        assert not ht.get(999)[0]

    def test_remove(self):
        from structures.robin_hood_ht import RobinHoodHashTable
        ht = RobinHoodHashTable()
        ht.put(42, 'x')
        assert ht.remove(42)
        assert not ht.get(42)[0]
        assert not ht.remove(42)

    def test_update(self):
        from structures.robin_hood_ht import RobinHoodHashTable
        ht = RobinHoodHashTable()
        ht.put(1, 'old')
        ht.put(1, 'new')
        assert ht.get(1) == (True, 'new')
        assert ht.size == 1


class TestRoaringBitmap:
    def test_basic(self):
        from structures.roaring_bitmap import RoaringBitmap
        rb = RoaringBitmap()
        for i in range(0, 10000, 3):
            rb.add(i)
        assert rb.cardinality() == len(range(0, 10000, 3))
        assert rb.contains(0)
        assert rb.contains(9999)
        assert not rb.contains(1)

    def test_and_or(self):
        from structures.roaring_bitmap import RoaringBitmap
        a = RoaringBitmap(); b = RoaringBitmap()
        for i in range(100): a.add(i)
        for i in range(50, 150): b.add(i)
        inter = a.and_op(b)
        assert inter.cardinality() == 50
        union = a.or_op(b)
        assert union.cardinality() == 150

    def test_remove(self):
        from structures.roaring_bitmap import RoaringBitmap
        rb = RoaringBitmap()
        rb.add(42)
        assert rb.contains(42)
        rb.remove(42)
        assert not rb.contains(42)


class TestSegmentTree:
    def test_sum(self):
        from structures.segment_tree import SegmentTree
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        st = SegmentTree(data)
        assert st.query(0, 9) == sum(data)
        assert st.query(2, 5) == 4 + 1 + 5 + 9

    def test_min_max(self):
        from structures.segment_tree import MinSegmentTree, MaxSegmentTree
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        assert MinSegmentTree(data).query(0, 9) == 1
        assert MaxSegmentTree(data).query(0, 9) == 9

    def test_update(self):
        from structures.segment_tree import SegmentTree
        st = SegmentTree([1, 2, 3])
        st.update(0, 10)
        assert st.query(0, 0) == 10
        assert st.query(0, 2) == 15

    def test_empty(self):
        from structures.segment_tree import SegmentTree
        st = SegmentTree([])
        assert st.query(0, 0) == 0


class TestSparseTable:
    def test_min_max(self):
        from structures.sparse_table import SparseTableMin, SparseTableMax
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        assert SparseTableMin(data).query(0, 9) == 1
        assert SparseTableMax(data).query(0, 9) == 9
        assert SparseTableMin(data).query(4, 7) == 2

    def test_single_element(self):
        from structures.sparse_table import SparseTableMin
        assert SparseTableMin([42]).query(0, 0) == 42


class TestFenwickTree:
    def test_prefix_sum(self):
        from structures.fenwick_tree import FenwickTree
        ft = FenwickTree.from_list([1, 2, 3, 4, 5])
        assert ft.prefix_sum(0) == 1
        assert ft.prefix_sum(4) == 15
        assert ft.range_sum(1, 3) == 9

    def test_update(self):
        from structures.fenwick_tree import FenwickTree
        ft = FenwickTree.from_list([1, 2, 3])
        ft.update(1, 10)
        assert ft.range_sum(1, 1) == 12


class TestLoserTree:
    def test_merge(self):
        from structures.tournament_tree import LoserTree
        s1 = iter([1, 4, 7, 10])
        s2 = iter([2, 5, 8, 11])
        s3 = iter([3, 6, 9, 12])
        lt = LoserTree([s1, s2, s3])
        result = lt.merge_all()
        assert result == list(range(1, 13))

    def test_single_source(self):
        from structures.tournament_tree import LoserTree
        lt = LoserTree([iter([3, 1, 4])])
        result = lt.merge_all()
        assert result == [3, 1, 4]  # Single source, no merge needed

    def test_empty_sources(self):
        from structures.tournament_tree import LoserTree
        lt = LoserTree([iter([]), iter([1, 2])])
        result = lt.merge_all()
        assert result == [1, 2]


class TestSortedContainer:
    def test_basic(self):
        from structures.sorted_container import SortedList
        sl = SortedList()
        vals = list(range(100))
        random.shuffle(vals)
        for v in vals:
            sl.add(v)
        assert sl.size == 100
        assert sl.kth(0) == 0
        assert sl.kth(99) == 99

    def test_remove(self):
        from structures.sorted_container import SortedList
        sl = SortedList()
        for v in range(10): sl.add(v)
        sl.remove(5)
        assert sl.size == 9
        assert sl.kth(5) == 6

    def test_median(self):
        from structures.sorted_container import SortedList
        sl = SortedList()
        for v in [1, 3, 5, 7, 9]: sl.add(v)
        assert sl.median() == 5


class TestXorFilter:
    def test_basic(self):
        from structures.xor_filter import XorFilter
        keys = list(range(100))
        xf = XorFilter(keys)
        for k in keys:
            assert xf.contains(k)
        fp = sum(1 for i in range(100, 200) if xf.contains(i))
        assert fp < 15  # Low FP rate


class TestCuckooFilter:
    def test_basic(self):
        from structures.cuckoo_filter import CuckooFilter
        cf = CuckooFilter(256)
        for i in range(50): cf.add(i)
        for i in range(50): assert cf.contains(i)
        cf.delete(25)
        assert not cf.contains(25)
        assert cf.count == 49


class TestWaveletTree:
    def test_quantile(self):
        from structures.wavelet_tree import WaveletTree
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        wt = WaveletTree(data, sigma=10)
        assert wt.quantile(0, 10, 0) == 1  # smallest

    def test_rank(self):
        from structures.wavelet_tree import WaveletTree
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        wt = WaveletTree(data, sigma=10)
        assert wt.rank(0, 5, 1) == 2  # two 1s in [0,5)


class TestART:
    def test_basic(self):
        from structures.art import AdaptiveRadixTree
        art = AdaptiveRadixTree()
        words = ['apple', 'app', 'banana', 'band']
        for i, w in enumerate(words):
            art.insert(w.encode(), i)
        for i, w in enumerate(words):
            assert art.search(w.encode()) == i
        assert art.search(b'xyz') is None

    def test_prefix_scan(self):
        from structures.art import AdaptiveRadixTree
        art = AdaptiveRadixTree()
        for w in ['apple', 'application', 'apply']:
            art.insert(w.encode(), w)
        results = art.prefix_scan(b'app')
        assert len(results) == 3


class TestSkipList:
    def test_basic(self):
        from structures.skip_list import SkipList
        sl = SkipList()
        for v in [5, 3, 8, 1, 9]: sl.insert(v, str(v))
        assert sl.search(3) == '3'
        assert sl.search(99) is None
        assert sl.size == 5

    def test_range_query(self):
        from structures.skip_list import SkipList
        sl = SkipList()
        for v in range(20): sl.insert(v, v)
        rng = sl.range_query(5, 10)
        assert len(rng) == 6  # 5..10 inclusive

    def test_delete(self):
        from structures.skip_list import SkipList
        sl = SkipList()
        sl.insert(1, 'a'); sl.insert(2, 'b')
        sl.delete(1)
        assert sl.search(1) is None
        assert sl.size == 1


class TestPerfectHash:
    def test_basic(self):
        from structures.perfect_hash import PerfectHashMap
        keys = list(range(50))
        values = [f"v{k}" for k in keys]
        phm = PerfectHashMap(keys, values)
        for k, v in zip(keys, values):
            assert phm.get(k) == v
        assert phm.get(999) is None


def run_structure_tests():
    classes = [TestRobinHood, TestRoaringBitmap, TestSegmentTree,
               TestSparseTable, TestFenwickTree, TestLoserTree,
               TestSortedContainer, TestXorFilter, TestCuckooFilter,
               TestWaveletTree, TestART, TestSkipList, TestPerfectHash]
    total = 0; passed = 0; failed = 0
    for cls in classes:
        obj = cls()
        for method_name in sorted(dir(obj)):
            if not method_name.startswith('test_'): continue
            total += 1
            try:
                getattr(obj, method_name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {cls.__name__}.{method_name}: {e}")
    print(f"\nStructures: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Structures Tests ===")
    run_structure_tests()
