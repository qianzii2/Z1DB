from __future__ import annotations
import threading
import time
from tests.conftest import *

def _worker_insert(engine: Engine, start: int, count: int):
    for i in range(start, start + count):
        engine.execute(f"INSERT INTO t VALUES ({i});")

def _worker_select(engine: Engine, count: int):
    for _ in range(count):
        r = engine.execute("SELECT COUNT(*) FROM t;")
        assert r.rows[0][0] >= 0

def test_concurrent_insert_select():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT);")

    t1 = threading.Thread(target=_worker_insert, args=(e, 0, 100))
    t2 = threading.Thread(target=_worker_select, args=(e, 100))

    t1.start(); t2.start()
    t1.join(); t2.join()

    r = e.execute("SELECT COUNT(*) FROM t;")
    assert r.rows[0][0] == 100
