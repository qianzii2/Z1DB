from __future__ import annotations
from tests.conftest import *

def test_create_table():
    e = make_engine()
    r = e.execute("CREATE TABLE t (id INT, name VARCHAR);")
    assert 'OK' in r.message
    tables = e.get_table_names()
    assert 't' in tables

def test_create_if_not_exists():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("CREATE TABLE IF NOT EXISTS t (a INT);")  # 不报错

def test_create_duplicate_error():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    assert_error(lambda: e.execute("CREATE TABLE t (a INT);"))

def test_drop_table():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("DROP TABLE t;")
    assert 't' not in e.get_table_names()

def test_drop_if_exists():
    e = make_engine()
    e.execute("DROP TABLE IF EXISTS nonexistent;")  # 不报错

def test_drop_nonexistent_error():
    e = make_engine()
    assert_error(lambda: e.execute("DROP TABLE nonexistent;"))

def test_alter_add_column():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("INSERT INTO t VALUES (1);")
    e.execute("ALTER TABLE t ADD COLUMN b VARCHAR;")
    r = e.execute("SELECT * FROM t;")
    assert len(r.columns) == 2
    assert r.rows[0][1] is None  # 新列默认 NULL

def test_alter_drop_column():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT, b VARCHAR);")
    e.execute("INSERT INTO t VALUES (1, 'x');")
    e.execute("ALTER TABLE t DROP COLUMN b;")
    r = e.execute("SELECT * FROM t;")
    assert len(r.columns) == 1
    assert r.columns == ['a']

def test_alter_rename_column():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT);")
    e.execute("ALTER TABLE t RENAME COLUMN a TO b;")
    schema = e.get_table_schema('t')
    assert schema.columns[0].name == 'b'

def test_column_types():
    e = make_engine()
    e.execute("""CREATE TABLE t (
        a INT, b BIGINT, c FLOAT, d DOUBLE,
        e BOOLEAN, f VARCHAR(50), g TEXT,
        h DATE, i TIMESTAMP
    );""")
    schema = e.get_table_schema('t')
    assert len(schema.columns) == 9

def test_primary_key():
    e = make_engine()
    e.execute("CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR);")
    schema = e.get_table_schema('t')
    assert schema.columns[0].primary_key == True

def test_not_null():
    e = make_engine()
    e.execute("CREATE TABLE t (a INT NOT NULL);")
    assert_error(lambda: e.execute("INSERT INTO t VALUES (NULL);"))
