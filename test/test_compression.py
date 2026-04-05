from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRLE:
    def test_encode_decode(self):
        from storage.compression.rle import rle_encode, rle_decode
        data = [1, 1, 1, 2, 2, 3]
        vals, lens = rle_encode(data)
        assert vals == [1, 2, 3]
        assert lens == [3, 2, 1]
        assert rle_decode(vals, lens) == data

    def test_aggregate(self):
        from storage.compression.rle import rle_encode, rle_aggregate_sum
        data = [10, 10, 10, 20, 20]
        vals, lens = rle_encode(data)
        assert rle_aggregate_sum(vals, lens) == 70

    def test_empty(self):
        from storage.compression.rle import rle_encode
        assert rle_encode([]) == ([], [])


class TestDictCodec:
    def test_basic(self):
        # Note: file might be dict_code.py or dict_codec.py
        try:
            from storage.compression.dict_codec import DictEncoded
        except ImportError:
            from storage.compression.dict_codec import DictEncoded
        data = ['apple', 'banana', 'apple', 'cherry']
        de = DictEncoded.encode(data)
        assert de.ndv == 3
        assert de.decode_all() == data
        assert de.filter_eq('apple') == [0, 2]


class TestDelta:
    def test_encode_decode(self):
        from storage.compression.delta import delta_encode, delta_decode
        data = [100, 103, 106, 109]
        base, deltas = delta_encode(data)
        assert delta_decode(base, deltas) == data

    def test_delta_of_delta(self):
        from storage.compression.delta import delta_of_delta_encode, delta_of_delta_decode
        data = [100, 103, 106, 109, 112]
        b, fd, dod = delta_of_delta_encode(data)
        assert delta_of_delta_decode(b, fd, dod) == data


class TestBitPack:
    def test_encode_decode(self):
        from storage.compression.bitpack import bitpack_encode, bitpack_decode
        data = [0, 5, 3, 15, 7]
        packed, count = bitpack_encode(data, 4)
        assert bitpack_decode(packed, 4, count) == data

    def test_for(self):
        from storage.compression.bitpack import for_encode, for_decode
        data = [1000, 1003, 1001, 1005]
        min_v, packed, count, bw = for_encode(data)
        assert for_decode(min_v, packed, count, bw) == data


class TestGorilla:
    def test_encode_decode(self):
        from storage.compression.gorilla import gorilla_encode, gorilla_decode
        data = [20.0 + i * 0.01 for i in range(50)]
        encoded = gorilla_encode(data)
        decoded = gorilla_decode(encoded)
        assert len(decoded) == len(data)
        for a, b in zip(data, decoded):
            assert abs(a - b) < 1e-10

    def test_ratio(self):
        from storage.compression.gorilla import gorilla_compression_ratio
        # Slowly changing floats (not perfectly linear — add noise for realistic test)
        import random
        random.seed(42)
        data = [20.0 + i * 0.01 + random.uniform(-0.001, 0.001) for i in range(100)]
        ratio = gorilla_compression_ratio(data)
        assert ratio < 1.0, f"Gorilla should compress, got ratio={ratio:.2f}"
        # Linear data compresses less well than noisy-but-similar data
        # Just verify it produces valid output
        from storage.compression.gorilla import gorilla_encode, gorilla_decode
        decoded = gorilla_decode(gorilla_encode(data))
        assert len(decoded) == len(data)


class TestALP:
    def test_encode_decode(self):
        from storage.compression.alp import alp_encode, alp_decode
        data = [3.14 + i * 0.01 for i in range(50)]
        encoded = alp_encode(data)
        decoded = alp_decode(encoded)
        assert len(decoded) == len(data)
        for a, b in zip(data, decoded):
            assert abs(a - b) < 1e-6


class TestFSST:
    def test_encode_decode(self):
        from storage.compression.fsst import SymbolTable, fsst_encode, fsst_decode
        strings = ['http://example.com/p1', 'http://example.com/p2'] * 20
        st = SymbolTable.train(strings)
        for s in strings[:3]:
            assert fsst_decode(fsst_encode(s, st), st) == s


class TestAnalyzer:
    def test_choose(self):
        from storage.compression.analyzer import analyze_and_choose
        from storage.types import DataType
        assert analyze_and_choose(list(range(100)), DataType.INT) == 'DELTA'
        assert analyze_and_choose(['a', 'b', 'a'] * 100, DataType.VARCHAR) == 'DICT'


def run_compression_tests():
    classes = [TestRLE, TestDictCodec, TestDelta, TestBitPack,
               TestGorilla, TestALP, TestFSST, TestAnalyzer]
    total = 0; passed = 0; failed = 0
    for cls in classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith('test_'): continue
            total += 1
            try:
                getattr(obj, name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {cls.__name__}.{name}: {e}")
    print(f"\nCompression: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Compression Tests ===")
    run_compression_tests()
