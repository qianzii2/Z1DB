from __future__ import annotations
import os, tempfile
from tests.conftest import *

def _write_csv(path, content):
    with open(path, 'w') as f:
        f.write(content)

def test_copy_from():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, name VARCHAR, score DOUBLE);")
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    try:
        tmpfile.write("id,name,score\n1,Alice,95.5\n2,Bob,88.0\n3,Charlie,72.3\n")
        tmpfile.close()
        r = e.execute(f"COPY t FROM '{tmpfile.name}';")
        assert r.affected_rows == 3
        r = e.execute("SELECT COUNT(*) FROM t;")
        assert_value(r, 3)
        r = e.execute("SELECT name FROM t WHERE id = 2;")
        assert_value(r, 'Bob')
    finally:
        os.unlink(tmpfile.name)

def test_copy_from_no_header():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b VARCHAR);")
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    try:
        tmpfile.write("1,hello\n2,world\n")
        tmpfile.close()
        r = e.execute(f"COPY t FROM '{tmpfile.name}' WITH (HEADER FALSE);")
        assert r.affected_rows == 2
    finally:
        os.unlink(tmpfile.name)

def test_copy_from_custom_delimiter():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b VARCHAR);")
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    try:
        tmpfile.write("a|b\n1|hello\n2|world\n")
        tmpfile.close()
        r = e.execute(f"COPY t FROM '{tmpfile.name}' WITH (DELIMITER '|');")
        assert r.affected_rows == 2
    finally:
        os.unlink(tmpfile.name)

def test_copy_to():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT, name VARCHAR);")
    e.execute("INSERT INTO t VALUES (1,'Alice'),(2,'Bob');")
    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmpfile.close()
    try:
        r = e.execute(f"COPY t TO '{tmpfile.name}';")
        assert r.affected_rows == 2
        with open(tmpfile.name, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows
        assert 'id' in lines[0]
    finally:
        os.unlink(tmpfile.name)

def test_copy_from_null_values():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b VARCHAR);")
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    try:
        tmpfile.write("a,b\n1,hello\n,\n3,world\n")
        tmpfile.close()
        r = e.execute(f"COPY t FROM '{tmpfile.name}';")
        assert r.affected_rows == 3
        r = e.execute("SELECT COUNT(*) FROM t WHERE a IS NULL;")
        assert_value(r, 1)
    finally:
        os.unlink(tmpfile.name)

def test_copy_nonexistent_table():
    e = make_engine()
    assert_error(lambda: e.execute("COPY nonexistent FROM '/tmp/x.csv';"))

def test_copy_roundtrip():
    """COPY FROM → COPY TO → COPY FROM 应数据一致。"""
    e = make_engine()
    e.execute("CREATE TABLE t1 (id INT, val DOUBLE);")
    e.execute("INSERT INTO t1 VALUES (1,1.5),(2,2.7),(3,NULL);")
    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmpfile.close()
    try:
        e.execute(f"COPY t1 TO '{tmpfile.name}';")
        e.execute("CREATE TABLE t2 (id INT, val DOUBLE);")
        e.execute(f"COPY t2 FROM '{tmpfile.name}';")
        r1 = e.execute("SELECT * FROM t1 ORDER BY id;")
        r2 = e.execute("SELECT * FROM t2 ORDER BY id;")
        assert r1.row_count == r2.row_count
    finally:
        os.unlink(tmpfile.name)
