from __future__ import annotations
from tests.conftest import *

def test_create_index():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, name VARCHAR);")
    e.execute("INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c');")
    r = e.execute("CREATE INDEX idx_id ON t (id);")
    assert 'CREATE INDEX' in r.message

def test_create_unique_index():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT);")
    e.execute("INSERT INTO t VALUES (1),(2),(3);")
    r = e.execute("CREATE UNIQUE INDEX idx ON t (id);")
    assert 'idx' in r.message

def test_create_index_if_not_exists():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT);")
    e.execute("CREATE INDEX idx ON t (id);")
    e.execute("CREATE INDEX IF NOT EXISTS idx ON t (id);")  # 不报错

def test_drop_index():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT);")
    e.execute("CREATE INDEX idx ON t (id);")
    e.execute("DROP INDEX idx;")

def test_drop_index_if_exists():
    e = make_engine()
    e.execute("DROP INDEX IF EXISTS nonexistent;")  # 不报错

def test_index_speeds_up_eq():
    """索引应使等值查询更快（功能测试，不计时）。"""
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, val VARCHAR);")
    for i in range(100):
        e.execute(f"INSERT INTO t VALUES ({i}, 'val_{i}');")
    e.execute("CREATE INDEX idx ON t (id);")
    r = e.execute("SELECT val FROM t WHERE id = 50;")
    assert_value(r, 'val_50')

def test_index_multi_column():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b INT, c VARCHAR);")
    e.execute("INSERT INTO t VALUES (1,10,'x'),(2,20,'y');")
    r = e.execute("CREATE INDEX idx ON t (a, b);")
    assert 'CREATE INDEX' in r.message

def test_index_after_insert():
    """INSERT 后索引应自动更新。"""
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, val VARCHAR);")
    e.execute("CREATE INDEX idx ON t (id);")
    e.execute("INSERT INTO t VALUES (1,'a'),(2,'b');")
    r = e.execute("SELECT val FROM t WHERE id = 2;")
    assert_value(r, 'b')
