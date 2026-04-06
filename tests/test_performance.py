from __future__ import annotations
import time
from tests.conftest import *

def test_perf_insert_1k():
    """插入 1000 行性能基准。"""
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, val VARCHAR);")
    t0 = time.perf_counter()
    for i in range(1000):
        e.execute(f"INSERT INTO t VALUES ({i}, 'val_{i}');")
    elapsed = time.perf_counter() - t0
    # 应在 1 秒内完成（内存引擎）
    assert elapsed < 1.0, f"插入 1000 行耗时 {elapsed:.2f}s"

def test_perf_select_1k():
    """扫描 1000 行性能基准。"""
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, val INT);")
    for i in range(1000):
        e.execute(f"INSERT INTO t VALUES ({i}, {i*2});")
    t0 = time.perf_counter()
    r = e.execute("SELECT COUNT(*) FROM t;")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"COUNT 耗时 {elapsed:.2f}s"
    assert_value(r, 1000)

def test_perf_where_filter():
    """WHERE 过滤性能。"""
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, val INT);")
    for i in range(1000):
        e.execute(f"INSERT INTO t VALUES ({i}, {i % 10});")
    t0 = time.perf_counter()
    r = e.execute("SELECT COUNT(*) FROM t WHERE val = 5;")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1
    assert r.rows[0][0] == 100

def test_perf_group_by():
    """GROUP BY 性能。"""
    e = make_engine()
    e.execute("CREATE TABLE t (g INT, v INT);")
    for i in range(1000):
        e.execute(f"INSERT INTO t VALUES ({i % 10}, {i});")
    t0 = time.perf_counter()
    r = e.execute("SELECT g, COUNT(*) FROM t GROUP BY g;")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2
    assert r.row_count == 10

def test_perf_join():
    """JOIN 性能。"""
    e = make_engine()
    e.execute("CREATE TABLE a (id INT, x INT);")
    e.execute("CREATE TABLE b (id INT, y INT);")
    for i in range(100):
        e.execute(f"INSERT INTO a VALUES ({i}, {i*2});")
        e.execute(f"INSERT INTO b VALUES ({i}, {i*3});")
    t0 = time.perf_counter()
    r = e.execute("SELECT COUNT(*) FROM a JOIN b ON a.id = b.id;")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2
    assert_value(r, 100)

def test_perf_order_by():
    """ORDER BY 性能。"""
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    for i in range(500, 0, -1):  # 逆序插入
        e.execute(f"INSERT INTO t VALUES ({i});")
    t0 = time.perf_counter()
    r = e.execute("SELECT a FROM t ORDER BY a LIMIT 10;")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2
    assert r.rows[0][0] == 1

def test_perf_distinct():
    """DISTINCT 性能。"""
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    for i in range(1000):
        e.execute(f"INSERT INTO t VALUES ({i % 100});")
    t0 = time.perf_counter()
    r = e.execute("SELECT DISTINCT a FROM t;")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2
    assert r.row_count == 100

def test_perf_string_functions():
    """字符串函数性能。"""
    e = make_engine()
    e.execute("CREATE TABLE t (s VARCHAR);")
    for i in range(100):
        e.execute(f"INSERT INTO t VALUES ('hello_world_{i}');")
    t0 = time.perf_counter()
    r = e.execute("SELECT COUNT(*) FROM t WHERE UPPER(s) LIKE 'HELLO%';")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1
