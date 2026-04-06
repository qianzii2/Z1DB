from __future__ import annotations
"""共享工具。"""
import os, sys, shutil, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from engine import Engine

def make_engine():
    return Engine(':memory:')

def make_engine_with_data():
    e = make_engine()
    e.execute("CREATE TABLE users (id INT NOT NULL, name VARCHAR, age INT, salary DOUBLE, active BOOLEAN);")
    e.execute("INSERT INTO users VALUES (1,'Alice',30,75000.0,TRUE);")
    e.execute("INSERT INTO users VALUES (2,'Bob',25,60000.0,TRUE);")
    e.execute("INSERT INTO users VALUES (3,'Charlie',35,90000.0,FALSE);")
    e.execute("INSERT INTO users VALUES (4,'Diana',28,NULL,TRUE);")
    e.execute("INSERT INTO users VALUES (5,NULL,NULL,NULL,NULL);")
    return e

def make_persistent_engine():
    tmpdir = tempfile.mkdtemp(prefix='z1db_test_')
    return Engine(tmpdir), tmpdir

def cleanup_persistent(engine, tmpdir):
    engine.close()
    shutil.rmtree(tmpdir, ignore_errors=True)

def assert_rows(result, expected):
    actual = sorted([tuple(r) for r in result.rows])
    exp = sorted([tuple(r) for r in expected])
    assert actual == exp, f"行不匹配\n实际: {actual}\n期望: {exp}"

def assert_rows_ordered(result, expected):
    actual = [tuple(r) for r in result.rows]
    exp = [tuple(r) for r in expected]
    assert actual == exp, f"有序行不匹配\n实际: {actual}\n期望: {exp}"

def assert_value(result, expected):
    assert result.rows and result.rows[0], "空结果"
    v = result.rows[0][0]
    if isinstance(expected, float):
        assert abs(v - expected) < 1e-6, f"{v} != {expected}"
    else:
        assert v == expected, f"{v} != {expected}"

def assert_error(fn, *args):
    try: fn(*args); assert False, "应抛异常"
    except Exception: pass
