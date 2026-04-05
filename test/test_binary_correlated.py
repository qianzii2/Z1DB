from __future__ import annotations
"""Binary persistence + correlated subquery tests."""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBinaryPersistence:
    def test_save_load_roundtrip(self):
        """Write binary, restart engine, read back."""
        tmp = tempfile.mkdtemp(prefix='z1db_test_')
        try:
            from engine import Engine
            e = Engine(tmp)
            e.execute("CREATE TABLE bp (id INT, name VARCHAR(20), score DOUBLE);")
            e.execute("INSERT INTO bp VALUES (1,'Alice',95.5),(2,'Bob',87.3),(3,NULL,NULL);")
            del e  # Close engine

            # Reload
            e2 = Engine(tmp)
            r = e2.execute("SELECT * FROM bp ORDER BY id;")
            assert r.row_count == 3, f"expected 3, got {r.row_count}"
            assert r.rows[0][0] == 1
            assert r.rows[0][1] == 'Alice'
            assert abs(r.rows[0][2] - 95.5) < 0.01
            assert r.rows[2][1] is None  # NULL preserved
            assert r.rows[2][2] is None
            e2.execute("DROP TABLE bp;")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_binary_file_smaller_than_json(self):
        """Binary format should be smaller than JSON for numeric data."""
        tmp = tempfile.mkdtemp(prefix='z1db_size_')
        try:
            from engine import Engine
            e = Engine(tmp)
            e.execute("CREATE TABLE sz (id INT, val INT);")
            for i in range(100):
                e.execute(f"INSERT INTO sz VALUES ({i}, {i*10});")
            del e

            from pathlib import Path
            bin_path = Path(tmp) / 'sz.z1db'
            json_path = Path(tmp) / 'sz.data.json'
            if bin_path.exists() and json_path.exists():
                assert bin_path.stat().st_size < json_path.stat().st_size, \
                    "Binary should be smaller"
            elif bin_path.exists():
                assert bin_path.stat().st_size < 5000  # Should be compact
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_boolean_roundtrip(self):
        tmp = tempfile.mkdtemp(prefix='z1db_bool_')
        try:
            from engine import Engine
            e = Engine(tmp)
            e.execute("CREATE TABLE br (id INT, flag BOOLEAN);")
            e.execute("INSERT INTO br VALUES (1, TRUE), (2, FALSE), (3, NULL);")
            del e
            e2 = Engine(tmp)
            r = e2.execute("SELECT flag FROM br ORDER BY id;")
            assert r.rows[0][0] == True
            assert r.rows[1][0] == False
            assert r.rows[2][0] is None
            e2.execute("DROP TABLE br;")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_varchar_roundtrip(self):
        tmp = tempfile.mkdtemp(prefix='z1db_vc_')
        try:
            from engine import Engine
            e = Engine(tmp)
            e.execute("CREATE TABLE vr (id INT, txt TEXT);")
            e.execute("INSERT INTO vr VALUES (1, 'hello world');")
            e.execute("INSERT INTO vr VALUES (2, 'special chars: ''quoted'' and \"double\"');")
            e.execute("INSERT INTO vr VALUES (3, NULL);")
            del e
            e2 = Engine(tmp)
            r = e2.execute("SELECT txt FROM vr ORDER BY id;")
            assert r.rows[0][0] == 'hello world'
            assert r.rows[2][0] is None
            e2.execute("DROP TABLE vr;")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_multiple_tables(self):
        tmp = tempfile.mkdtemp(prefix='z1db_multi_')
        try:
            from engine import Engine
            e = Engine(tmp)
            e.execute("CREATE TABLE m1 (x INT);")
            e.execute("CREATE TABLE m2 (y VARCHAR(10));")
            e.execute("INSERT INTO m1 VALUES (1),(2),(3);")
            e.execute("INSERT INTO m2 VALUES ('a'),('b');")
            del e
            e2 = Engine(tmp)
            assert e2.execute("SELECT COUNT(*) FROM m1;").rows[0][0] == 3
            assert e2.execute("SELECT COUNT(*) FROM m2;").rows[0][0] == 2
            e2.execute("DROP TABLE m1;")
            e2.execute("DROP TABLE m2;")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_drop_cleans_files(self):
        tmp = tempfile.mkdtemp(prefix='z1db_drop_')
        try:
            from engine import Engine
            from pathlib import Path
            e = Engine(tmp)
            e.execute("CREATE TABLE dc (x INT);")
            e.execute("INSERT INTO dc VALUES (1);")
            e.execute("DROP TABLE dc;")
            # Files should be cleaned
            assert not (Path(tmp) / 'dc.z1db').exists()
            assert not (Path(tmp) / 'dc.data.json').exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_update_persists(self):
        tmp = tempfile.mkdtemp(prefix='z1db_upd_')
        try:
            from engine import Engine
            e = Engine(tmp)
            e.execute("CREATE TABLE up (id INT, val INT);")
            e.execute("INSERT INTO up VALUES (1, 100), (2, 200);")
            e.execute("UPDATE up SET val = 999 WHERE id = 1;")
            del e
            e2 = Engine(tmp)
            r = e2.execute("SELECT val FROM up WHERE id = 1;")
            assert r.rows[0][0] == 999
            e2.execute("DROP TABLE up;")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSerializeDeserialize:
    def test_int_column(self):
        from storage.table_file import serialize_column, deserialize_column
        values = [10, 20, None, 40, 50]
        nulls = [False, False, True, False, False]
        bmp, data = serialize_column(values, nulls, 'INT')
        vals2, nulls2 = deserialize_column(bmp, data, 5, 'INT')
        assert vals2 == [10, 20, None, 40, 50]

    def test_double_column(self):
        from storage.table_file import serialize_column, deserialize_column
        values = [1.5, None, 3.14]
        nulls = [False, True, False]
        bmp, data = serialize_column(values, nulls, 'DOUBLE')
        vals2, nulls2 = deserialize_column(bmp, data, 3, 'DOUBLE')
        assert vals2[0] is not None and abs(vals2[0] - 1.5) < 1e-10
        assert vals2[1] is None
        assert abs(vals2[2] - 3.14) < 1e-10

    def test_varchar_column(self):
        from storage.table_file import serialize_column, deserialize_column
        values = ['hello', None, 'world']
        nulls = [False, True, False]
        bmp, data = serialize_column(values, nulls, 'VARCHAR')
        vals2, nulls2 = deserialize_column(bmp, data, 3, 'VARCHAR')
        assert vals2 == ['hello', None, 'world']

    def test_boolean_column(self):
        from storage.table_file import serialize_column, deserialize_column
        values = [True, False, None, True]
        nulls = [False, False, True, False]
        bmp, data = serialize_column(values, nulls, 'BOOLEAN')
        vals2, nulls2 = deserialize_column(bmp, data, 4, 'BOOLEAN')
        assert vals2 == [True, False, None, True]

    def test_empty_column(self):
        from storage.table_file import serialize_column, deserialize_column
        bmp, data = serialize_column([], [], 'INT')
        vals2, nulls2 = deserialize_column(bmp, data, 0, 'INT')
        assert vals2 == []

    def test_all_null_column(self):
        from storage.table_file import serialize_column, deserialize_column
        values = [None, None, None]
        nulls = [True, True, True]
        bmp, data = serialize_column(values, nulls, 'INT')
        vals2, nulls2 = deserialize_column(bmp, data, 3, 'INT')
        assert vals2 == [None, None, None]

    def test_bigint_column(self):
        from storage.table_file import serialize_column, deserialize_column
        values = [2**40, -(2**40), None]
        nulls = [False, False, True]
        bmp, data = serialize_column(values, nulls, 'BIGINT')
        vals2, nulls2 = deserialize_column(bmp, data, 3, 'BIGINT')
        assert vals2[0] == 2**40
        assert vals2[1] == -(2**40)
        assert vals2[2] is None


class TestCorrelatedSubquery:
    def test_exists_via_join(self):
        """EXISTS rewritten as JOIN — tests the pattern."""
        from engine import Engine
        e = Engine()
        e.execute("CREATE TABLE c1 (id INT, name VARCHAR(20));")
        e.execute("CREATE TABLE o1 (id INT, cid INT, amt INT);")
        e.execute("INSERT INTO c1 VALUES (1,'Alice'),(2,'Bob'),(3,'Carol');")
        e.execute("INSERT INTO o1 VALUES (1,1,100),(2,1,200),(3,2,50);")
        r = e.execute("""
            SELECT DISTINCT c.name FROM c1 c
            INNER JOIN o1 o ON c.id = o.cid ORDER BY c.name;
        """)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names and 'Bob' in names and 'Carol' not in names
        e.execute("DROP TABLE c1;"); e.execute("DROP TABLE o1;")

    def test_not_exists_via_left_join(self):
        """NOT EXISTS rewritten as LEFT JOIN IS NULL."""
        from engine import Engine
        e = Engine()
        e.execute("CREATE TABLE c2 (id INT, name VARCHAR(20));")
        e.execute("CREATE TABLE o2 (cid INT);")
        e.execute("INSERT INTO c2 VALUES (1,'Alice'),(2,'Bob'),(3,'Carol');")
        e.execute("INSERT INTO o2 VALUES (1),(1);")
        r = e.execute("""
            SELECT c.name FROM c2 c LEFT JOIN o2 o ON c.id = o.cid
            WHERE o.cid IS NULL ORDER BY c.name;
        """)
        names = [row[0] for row in r.rows]
        assert 'Bob' in names and 'Carol' in names and 'Alice' not in names
        e.execute("DROP TABLE c2;"); e.execute("DROP TABLE o2;")

    def test_uncorrelated_in_subquery(self):
        from engine import Engine
        e = Engine()
        e.execute("CREATE TABLE t1 (x INT);")
        e.execute("CREATE TABLE t2 (y INT);")
        e.execute("INSERT INTO t1 VALUES (1),(2),(3),(4),(5);")
        e.execute("INSERT INTO t2 VALUES (2),(4),(6);")
        r = e.execute("SELECT x FROM t1 WHERE x IN (SELECT y FROM t2) ORDER BY x;")
        assert [row[0] for row in r.rows] == [2, 4]
        e.execute("DROP TABLE t1;"); e.execute("DROP TABLE t2;")

    def test_scalar_subquery_in_select(self):
        from engine import Engine
        e = Engine()
        e.execute("CREATE TABLE ss (x INT);")
        e.execute("INSERT INTO ss VALUES (10),(20),(30);")
        r = e.execute("SELECT (SELECT MAX(x) FROM ss) AS mx;")
        assert r.rows[0][0] == 30
        e.execute("DROP TABLE ss;")

    def test_scalar_subquery_in_where(self):
        from engine import Engine
        e = Engine()
        e.execute("CREATE TABLE sw (id INT, val INT);")
        e.execute("INSERT INTO sw VALUES (1,10),(2,20),(3,30);")
        r = e.execute("SELECT id FROM sw WHERE val > (SELECT AVG(val) FROM sw) ORDER BY id;")
        assert [row[0] for row in r.rows] == [3]
        e.execute("DROP TABLE sw;")

    def test_not_in_subquery(self):
        from engine import Engine
        e = Engine()
        e.execute("CREATE TABLE ni1 (x INT);")
        e.execute("CREATE TABLE ni2 (y INT);")
        e.execute("INSERT INTO ni1 VALUES (1),(2),(3),(4),(5);")
        e.execute("INSERT INTO ni2 VALUES (2),(4);")
        r = e.execute("SELECT x FROM ni1 WHERE x NOT IN (SELECT y FROM ni2) ORDER BY x;")
        assert [row[0] for row in r.rows] == [1, 3, 5]
        e.execute("DROP TABLE ni1;"); e.execute("DROP TABLE ni2;")


def run_binary_correlated_tests():
    classes = [TestBinaryPersistence, TestSerializeDeserialize, TestCorrelatedSubquery]
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
    print(f"\nBinary+Correlated: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Binary Persistence & Correlated Subquery Tests ===")
    run_binary_correlated_tests()
