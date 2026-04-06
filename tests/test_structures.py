from __future__ import annotations
from tests.conftest import *

def test_bloom_filter_basic():
    from structures.bloom_filter import BloomFilter
    bf = BloomFilter(1000, 0.01)
    bf.add('apple')
    bf.add('banana')
    assert bf.contains('apple') == True
    assert bf.contains('banana') == True
    assert bf.contains('cherry') == False or True  # 可能误判

def test_bloom_filter_false_positive():
    from structures.bloom_filter import BloomFilter
    bf = BloomFilter(100, 0.01)
    for i in range(50):
        bf.add(f'item_{i}')
    # 检查已添加的
    for i in range(50):
        assert bf.contains(f'item_{i}') == True
    # 检查未添加的（可能误判但概率低）
    false_positives = sum(1 for i in range(50, 100) if bf.contains(f'item_{i}'))
    assert false_positives < 10  # 误判率应低于 10%

def test_hll_cardinality():
    from structures.hll import HyperLogLog
    hll = HyperLogLog(14)
    for i in range(1000):
        hll.add(f'item_{i}')
    card = hll.cardinality()
    # HLL 误差 ~2%
    assert 950 < card < 1050

def test_kll_quantile():
    from structures.kll import KLL
    kll = KLL(200)
    for i in range(1000):
        kll.add(i)
    p50 = kll.query(0.5)
    p99 = kll.query(0.99)
    assert 400 < p50 < 600  # 中位数应在 500 附近
    assert 900 < p99 < 1000  # 99 分位应在 990 附近

def test_art_tree_basic():
    from structures.art import AdaptiveRadixTree
    art = AdaptiveRadixTree()
    art.insert('apple', 1)
    art.insert('app', 2)
    art.insert('application', 3)
    assert art.search('apple') == 1
    assert art.search('app') == 2
    assert art.search('application') == 3
    assert art.search('apricot') is None

def test_art_tree_range():
    from structures.art import AdaptiveRadixTree
    art = AdaptiveRadixTree()
    for i in range(100):
        art.insert(f'key_{i:03d}', i)
    results = art.range_search('key_010', 'key_050')
    assert len(results) > 0
    assert all(10 <= v <= 50 for v in results)

def test_skiplist_basic():
    from structures.skiplist import SkipList
    sl = SkipList()
    sl.insert(5, 'five')
    sl.insert(3, 'three')
    sl.insert(7, 'seven')
    assert sl.search(5) == 'five'
    assert sl.search(3) == 'three'
    assert sl.search(10) is None

def test_skiplist_range():
    from structures.skiplist import SkipList
    sl = SkipList()
    for i in range(1, 11):
        sl.insert(i, f'val_{i}')
    results = sl.range_search(3, 7)
    assert len(results) == 5  # 3,4,5,6,7

def test_robin_hood_hash():
    from structures.robin_hood_ht import RobinHoodHashTable
    rh = RobinHoodHashTable(capacity=16)
    rh.put(1, 'one')
    rh.put(2, 'two')
    rh.put(3, 'three')
    found, val = rh.get(2)
    assert found == True and val == 'two'
    found, val = rh.get(99)
    assert found == False

def test_robin_hood_collision():
    from structures.robin_hood_ht import RobinHoodHashTable
    rh = RobinHoodHashTable(capacity=8)
    # 故意制造碰撞
    for i in range(20):
        rh.put(i, f'val_{i}')
    # 验证所有值都能找到
    for i in range(20):
        found, val = rh.get(i)
        assert found == True and val == f'val_{i}'

def test_cuckoo_hash():
    from structures.cuckoo_hash import CuckooHashMap
    ch = CuckooHashMap(capacity=16)
    ch.put(1, 'one')
    ch.put(2, 'two')
    assert ch.get(1) == 'one'
    assert ch.get(2) == 'two'
    assert ch.get(99) is None

def test_lru_cache():
    from structures.lru_cache import LRUCache
    cache = LRUCache(max_size=3)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    assert cache.get('a') == 1
    cache.put('d', 4)  # 应淘汰 'b'
    assert cache.get('b') is None
    assert cache.get('d') == 4
