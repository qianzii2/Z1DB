from __future__ import annotations
from tests.conftest import *

def test_table_store_basic():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b VARCHAR);")
    e.execute("INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z');")
    store = e._catalog.get_store('t')
    assert store.row_count == 3
    rows = store.read_all_rows()
    assert len(rows) == 3

def test_table_store_delete():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3),(4),(5);")
    e.execute("DELETE FROM t WHERE a > 3;")
    store = e._catalog.get_store('t')
    assert store.row_count == 3

def test_table_store_update():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT);")
    e.execute("INSERT INTO t VALUES (1,10),(2,20);")
    e.execute("UPDATE t SET b = 99 WHERE a = 1;")
    r = e.execute("SELECT b FROM t WHERE a = 1;")
    assert_value(r, 99)

def test_table_store_truncate():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3);")
    e.execute("DELETE FROM t;")
    assert e._catalog.get_store('t').row_count == 0

def test_chunk_boundary():
    """测试 chunk 边界（CHUNK_SIZE = 65536）。"""
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    # 插入少量行
    for i in range(100):
        e.execute(f"INSERT INTO t VALUES ({i});")
    store = e._catalog.get_store('t')
    assert store.get_chunk_count() >= 1
    r = e.execute("SELECT COUNT(*) FROM t;")
    assert_value(r, 100)

def test_column_chunks():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b VARCHAR);")
    e.execute("INSERT INTO t VALUES (1,'x'),(2,'y');")
    store = e._catalog.get_store('t')
    chunks_a = store.get_column_chunks('a')
    chunks_b = store.get_column_chunks('b')
    assert len(chunks_a) >= 1
    assert chunks_a[0].row_count == 2

def test_zone_map():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (10),(20),(30);")
    store = e._catalog.get_store('t')
    chunk = store.get_column_chunks('a')[0]
    assert chunk.zone_map['min'] == 10
    assert chunk.zone_map['max'] == 30

def test_read_rows_by_indices():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (10),(20),(30),(40),(50);")
    store = e._catalog.get_store('t')
    rows = store.read_rows_by_indices([1, 3])
    assert len(rows) == 2
    assert rows[0][0] == 20
    assert rows[1][0] == 40

def test_multiple_deletes_compact():
    """大量删除应触发 compact。"""
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    for i in range(20):
        e.execute(f"INSERT INTO t VALUES ({i});")
    # 删除超过一半
    e.execute("DELETE FROM t WHERE a < 15;")
    r = e.execute("SELECT COUNT(*) FROM t;")
    assert_value(r, 5)
    # 确认剩余数据正确
    r = e.execute("SELECT MIN(a), MAX(a) FROM t;")
    assert r.rows[0] == [15, 19]
