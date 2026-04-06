from __future__ import annotations
import os
import shutil
import tempfile
from tests.conftest import *
from engine import Engine

def test_write_close_reopen():
    tmpdir = tempfile.mkdtemp(prefix='z1db_persist_')
    try:
        e = Engine(tmpdir)
        e.execute("CREATE TABLE t (id INT, name VARCHAR);")
        e.execute("INSERT INTO t VALUES (1, 'Alice');")
        e.execute("INSERT INTO t VALUES (2, 'Bob');")
        e.close()
        # 重新打开
        e2 = Engine(tmpdir)
        r = e2.execute("SELECT COUNT(*) FROM t;")
        assert_value(r, 2)
        r = e2.execute("SELECT name FROM t WHERE id = 1;")
        assert_value(r, 'Alice')
        e2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_multiple_tables():
    tmpdir = tempfile.mkdtemp(prefix='z1db_persist_')
    try:
        e = Engine(tmpdir)
        e.execute("CREATE TABLE users (id INT, name VARCHAR);")
        e.execute("CREATE TABLE orders (id INT, user_id INT, amount DOUBLE);")
        e.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');")
        e.execute("INSERT INTO orders VALUES (1, 1, 10.0), (2, 2, 20.5);")
        e.close()
        # 重新打开
        e2 = Engine(tmpdir)
        r1 = e2.execute("SELECT COUNT(*) FROM users;")
        r2 = e2.execute("SELECT COUNT(*) FROM orders;")
        assert_value(r1, 2)
        assert_value(r2, 2)
        e2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_update_persist():
    tmpdir = tempfile.mkdtemp(prefix='z1db_persist_')
    try:
        e = Engine(tmpdir)
        e.execute("CREATE TABLE t (id INT, val DOUBLE);")
        e.execute("INSERT INTO t VALUES (1, 10.5);")
        e.execute("UPDATE t SET val = 20.0 WHERE id = 1;")
        e.close()
        e2 = Engine(tmpdir)
        r = e2.execute("SELECT val FROM t WHERE id = 1;")
        assert_value(r, 20.0)
        e2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_delete_persist():
    tmpdir = tempfile.mkdtemp(prefix='z1db_persist_')
    try:
        e = Engine(tmpdir)
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (1),(2),(3);")
        e.execute("DELETE FROM t WHERE id = 2;")
        e.close()
        e2 = Engine(tmpdir)
        r = e2.execute("SELECT COUNT(*) FROM t;")
        assert_value(r, 2)
        e2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_persist_purge_on_drop():
    tmpdir = tempfile.mkdtemp(prefix='z1db_persist_')
    try:
        e = Engine(tmpdir)
        e.execute("CREATE TABLE t (id INT);")
        e.execute("INSERT INTO t VALUES (1), (2);")
        e.close()
        # 删除表后，数据目录清空
        e2 = Engine(tmpdir)
        e2.execute("DROP TABLE t;")
        path = os.path.join(tmpdir, 'lsm_t')
        assert not os.path.exists(path) or len(os.listdir(path)) == 0
        e2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
