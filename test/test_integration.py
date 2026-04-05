from __future__ import annotations
"""Integration tests — verify all optimizations work together."""
from engine import Engine


def _e():
    return Engine()


def test_dict_domain():
    e = _e()
    e.execute("CREATE TABLE t (city VARCHAR(20));")
    for c in ['Beijing']*4 + ['Shanghai']*3 + ['Guangzhou']*3:
        e.execute(f"INSERT INTO t VALUES ('{c}');")
    r = e.execute("SELECT COUNT(*) FROM t WHERE city = 'Beijing';")
    assert r.rows[0][0] == 4
    r = e.execute("SELECT COUNT(*) FROM t WHERE city = 'Nonexistent';")
    assert r.rows[0][0] == 0
    print("    Dict Domain ✓")
    e.execute("DROP TABLE t;")

def test_result_cache():
    e = _e()
    e.execute("CREATE TABLE t (id INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3);")
    r1 = e.execute("SELECT COUNT(*) FROM t;")
    r2 = e.execute("SELECT COUNT(*) FROM t;")
    assert r1.rows == r2.rows
    assert e._result_cache.size > 0
    e.execute("INSERT INTO t VALUES (4);")
    r3 = e.execute("SELECT COUNT(*) FROM t;")
    assert r3.rows[0][0] == 4
    print("    Result Cache ✓")
    e.execute("DROP TABLE t;")

def test_swar_upper():
    e = _e()
    e.execute("CREATE TABLE t (s VARCHAR(50));")
    e.execute("INSERT INTO t VALUES ('hello'),('world'),('Test 123');")
    r = e.execute("SELECT UPPER(s) FROM t ORDER BY s;")
    assert r.rows[0][0] == 'TEST 123'
    assert r.rows[1][0] == 'HELLO'
    assert r.rows[2][0] == 'WORLD'
    print("    SWAR UPPER ✓")
    e.execute("DROP TABLE t;")

def test_join_bloom():
    e = _e()
    e.execute("CREATE TABLE a (id INT);")
    e.execute("CREATE TABLE b (id INT);")
    for i in range(100): e.execute(f"INSERT INTO a VALUES ({i});")
    for i in range(50, 150): e.execute(f"INSERT INTO b VALUES ({i});")
    r = e.execute("SELECT COUNT(*) FROM a INNER JOIN b ON a.id = b.id;")
    assert r.rows[0][0] == 50
    print("    JOIN + Bloom ✓")
    e.execute("DROP TABLE a;"); e.execute("DROP TABLE b;")

def test_lazy_filter():
    e = _e()
    e.execute("CREATE TABLE t (id INT, val INT);")
    for i in range(100): e.execute(f"INSERT INTO t VALUES ({i}, {i*2});")
    r = e.execute("SELECT id, val FROM t WHERE id > 95 ORDER BY id;")
    assert r.row_count == 4
    assert r.rows[0][0] == 96
    print("    Lazy Filter ✓")
    e.execute("DROP TABLE t;")

def test_vectorized_cmp():
    e = _e()
    e.execute("CREATE TABLE t (x INT, y INT);")
    for i in range(50): e.execute(f"INSERT INTO t VALUES ({i}, {50-i});")
    r = e.execute("SELECT COUNT(*) FROM t WHERE x > y;")
    assert r.rows[0][0] == 24
    r = e.execute("SELECT COUNT(*) FROM t WHERE x = y;")
    assert r.rows[0][0] == 1
    print("    Vectorized CMP ✓")
    e.execute("DROP TABLE t;")

def test_topn():
    e = _e()
    e.execute("CREATE TABLE t (id INT, score INT);")
    for i in range(100): e.execute(f"INSERT INTO t VALUES ({i}, {1000-i*3});")
    r = e.execute("SELECT score FROM t ORDER BY score DESC LIMIT 3;")
    assert r.row_count == 3
    assert r.rows[0][0] == 1000
    print("    TopN ✓")
    e.execute("DROP TABLE t;")

def test_constant_folding():
    e = _e()
    r = e.execute("SELECT 1 + 2 * 3;")
    assert r.rows[0][0] == 7
    assert r.columns[0] == '1 + 2 * 3'
    r = e.execute("SELECT 'a' || 'b' || 'c';")
    assert r.rows[0][0] == 'abc'
    print("    Constant Folding ✓")

def test_full_pipeline():
    """CTE + JOIN + AGG + Window + LIKE + ORDER + LIMIT — all together."""
    e = _e()
    e.execute("CREATE TABLE emp (id INT, dept VARCHAR(20), salary INT);")
    depts = ['Eng', 'Sales', 'Mkt', 'Eng', 'Sales', 'Eng', 'Mkt', 'Sales', 'Eng', 'Eng']
    for i, d in enumerate(depts):
        e.execute(f"INSERT INTO emp VALUES ({i+1}, '{d}', {50000 + i * 5000});")

    r = e.execute("""
        WITH ds AS (
            SELECT dept, AVG(salary) AS avg_sal FROM emp GROUP BY dept
        )
        SELECT * FROM ds ORDER BY avg_sal DESC;
    """)
    assert r.row_count == 3

    r = e.execute("""
        SELECT dept, salary,
               RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk
        FROM emp ORDER BY dept, rnk LIMIT 5;
    """)
    assert r.row_count == 5

    r = e.execute("SELECT COUNT(*) FROM emp WHERE dept LIKE 'Eng%';")
    assert r.rows[0][0] == 5

    print("    Full Pipeline ✓")
    e.execute("DROP TABLE emp;")

def test_memory_budget():
    from executor.memory_budget import MemoryBudget
    mb = MemoryBudget(1000)
    assert mb.allocate('op1', 500)
    assert mb.remaining == 500
    assert mb.should_spill('op2', 500)  # 500 + 500 > 800
    mb.release('op1')
    assert mb.remaining == 1000
    print("    Memory Budget ✓")

def test_sketches():
    from executor.sketch.hyperloglog import HyperLogLog
    from executor.sketch.t_digest import TDigest
    from executor.sketch.count_min_sketch import CountMinSketch
    hll = HyperLogLog(p=11)
    for i in range(1000): hll.add(i)
    est = hll.estimate()
    assert 900 <= est <= 1100
    td = TDigest()
    for i in range(1000): td.add(float(i))
    assert 450 <= td.median() <= 550
    cms = CountMinSketch()
    for i in range(100): cms.add('x')
    assert cms.estimate('x') >= 100
    print(f"    Sketches ✓ (HLL={est})")

def test_compression():
    from storage.compression.rle import rle_encode, rle_decode
    from storage.compression.gorilla import gorilla_encode, gorilla_decode
    from storage.compression.delta import delta_encode, delta_decode
    data = [1, 1, 1, 2, 2, 3]
    v, l = rle_encode(data)
    assert rle_decode(v, l) == data
    floats = [20.0 + i * 0.01 for i in range(50)]
    assert gorilla_decode(gorilla_encode(floats)) == floats
    base, deltas = delta_encode(list(range(10)))
    assert delta_decode(base, deltas) == list(range(10))
    print("    Compression ✓")


ALL_TESTS = [
    ("Dict Domain", test_dict_domain),
    ("Result Cache", test_result_cache),
    ("SWAR UPPER", test_swar_upper),
    ("JOIN+Bloom", test_join_bloom),
    ("Lazy Filter", test_lazy_filter),
    ("Vectorized CMP", test_vectorized_cmp),
    ("TopN", test_topn),
    ("Constant Folding", test_constant_folding),
    ("Full Pipeline", test_full_pipeline),
    ("Memory Budget", test_memory_budget),
    ("Sketches", test_sketches),
    ("Compression", test_compression),
]
