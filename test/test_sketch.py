from __future__ import annotations
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHyperLogLog:
    def test_accuracy(self):
        from executor.sketch.hyperloglog import HyperLogLog
        hll = HyperLogLog(p=11)
        n = 10000
        for i in range(n): hll.add(i)
        est = hll.estimate()
        error = abs(est - n) / n
        assert error < 0.05, f"error {error:.2%}"

    def test_duplicates(self):
        from executor.sketch.hyperloglog import HyperLogLog
        hll = HyperLogLog(p=11)
        for i in range(100): hll.add(42)
        assert hll.estimate() <= 5

    def test_merge(self):
        from executor.sketch.hyperloglog import HyperLogLog
        a = HyperLogLog(p=11); b = HyperLogLog(p=11)
        for i in range(5000): a.add(i)
        for i in range(5000, 10000): b.add(i)
        a.merge(b)
        error = abs(a.estimate() - 10000) / 10000
        assert error < 0.05


class TestCountMinSketch:
    def test_frequency(self):
        from executor.sketch.count_min_sketch import CountMinSketch
        cms = CountMinSketch()
        for _ in range(100): cms.add('apple')
        for _ in range(50): cms.add('banana')
        assert cms.estimate('apple') >= 100
        assert cms.estimate('banana') >= 50
        assert cms.estimate('unknown') <= 5


class TestTDigest:
    def test_median(self):
        from executor.sketch.t_digest import TDigest
        td = TDigest()
        random.seed(42)
        data = sorted([random.gauss(0, 1) for _ in range(1000)])
        for v in data: td.add(v)
        actual = data[len(data) // 2]
        est = td.median()
        assert abs(est - actual) < 0.2

    def test_quantile(self):
        from executor.sketch.t_digest import TDigest
        td = TDigest()
        for i in range(100): td.add(float(i))
        p50 = td.quantile(0.5)
        assert 40 < p50 < 60


class TestKLLSketch:
    def test_basic(self):
        from executor.sketch.kll_sketch import KLLSketch
        kll = KLLSketch(k=200)
        for i in range(1000): kll.add(float(i))
        med = kll.median()
        # KLL with k=200 on 1000 values: median of [0,999] = 499.5
        # Allow wide range due to compaction randomness
        assert 300 < med < 700, f"KLL median={med}, expected ~500"
        assert kll.count == 1000

    def test_merge(self):
        from executor.sketch.kll_sketch import KLLSketch
        a = KLLSketch(k=200)
        b = KLLSketch(k=200)
        for i in range(500): a.add(float(i))
        for i in range(500, 1000): b.add(float(i))
        a.merge(b)
        assert a.count == 1000
        med = a.median()
        assert 300 < med < 700

    def test_extremes(self):
        from executor.sketch.kll_sketch import KLLSketch
        kll = KLLSketch(k=200)
        for i in range(100): kll.add(float(i))
        assert kll.quantile(0.0) == 0.0
        assert kll.quantile(1.0) == 99.0


def run_sketch_tests():
    classes = [TestHyperLogLog, TestCountMinSketch, TestTDigest, TestKLLSketch]
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
    print(f"\nSketches: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Sketch Tests ===")
    run_sketch_tests()
