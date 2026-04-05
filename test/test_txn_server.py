from __future__ import annotations
"""Transaction and TCP server tests."""
import sys, os, time, threading, json, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLockManager:
    def test_read_read_parallel(self):
        from txn.lock import TableLockManager
        lm = TableLockManager()
        lm.acquire_read('t')
        lm.acquire_read('t')
        assert lm.reader_count('t') == 2
        lm.release_read('t')
        assert lm.reader_count('t') == 1
        lm.release_read('t')
        assert lm.reader_count('t') == 0

    def test_write_exclusive(self):
        from txn.lock import TableLockManager
        lm = TableLockManager()
        lm.acquire_write('t')
        assert lm.is_locked_for_write('t')
        lm.release_write('t')
        assert not lm.is_locked_for_write('t')

    def test_write_serialization(self):
        from txn.lock import TableLockManager
        lm = TableLockManager()
        results = []

        def writer(wid: int) -> None:
            lm.acquire_write('t')
            results.append(f'start_{wid}')
            time.sleep(0.01)
            results.append(f'end_{wid}')
            lm.release_write('t')

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start(); t2.start()
        t1.join(); t2.join()
        # Should be serialized: start_1,end_1,start_2,end_2 or vice versa
        assert len(results) == 4
        # No interleaving: start and end of same writer must be adjacent
        for i in range(0, 4, 2):
            wid = results[i].split('_')[1]
            assert results[i + 1] == f'end_{wid}'


class TestTransactionManager:
    def test_auto_commit(self):
        from txn.manager import TransactionManager
        tm = TransactionManager()
        assert tm.auto_commit
        assert not tm.in_transaction

    def test_begin_commit(self):
        from txn.manager import TransactionManager
        tm = TransactionManager()
        txn_id = tm.begin()
        assert tm.in_transaction
        assert not tm.auto_commit
        tm.commit()
        assert not tm.in_transaction

    def test_delete_rollback(self):
        """Full delete then rollback — verify via store directly."""
        from txn.manager import TransactionManager
        from engine import Engine
        e = Engine()
        tm = TransactionManager()

        e.execute("CREATE TABLE dr1 (id INT);")
        e.execute("INSERT INTO dr1 VALUES (1), (2), (3);")

        store = e._catalog.get_store('dr1')
        assert store.row_count == 3

        txn_id = tm.begin()
        tm.snapshot_table('dr1', e._catalog)

        e.execute("DELETE FROM dr1;")
        assert store.row_count == 0

        tm.rollback(e._catalog)

        # Verify via store directly (bypasses planner entirely)
        store = e._catalog.get_store('dr1')
        assert store.row_count == 3, f"store row_count: {store.row_count}"
        rows = store.read_all_rows()
        assert len(rows) == 3, f"read_all_rows: {len(rows)}"
        ids = sorted(r[0] for r in rows)
        assert ids == [1, 2, 3], f"row data: {ids}"

        e.execute("DROP TABLE dr1;")

    def test_partial_delete_rollback(self):
        from txn.manager import TransactionManager
        from engine import Engine
        e = Engine()
        tm = TransactionManager()

        e.execute("CREATE TABLE dr2 (id INT);")
        e.execute("INSERT INTO dr2 VALUES (10), (20), (30);")

        txn_id = tm.begin()
        tm.snapshot_table('dr2', e._catalog)

        e.execute("DELETE FROM dr2 WHERE id = 20;")
        store = e._catalog.get_store('dr2')
        assert store.row_count == 2

        tm.rollback(e._catalog)
        store = e._catalog.get_store('dr2')
        assert store.row_count == 3, f"after rollback: {store.row_count}"
        rows = store.read_all_rows()
        ids = sorted(r[0] for r in rows)
        assert ids == [10, 20, 30], f"data: {ids}"

        e.execute("DROP TABLE dr2;")

    def test_update_rollback(self):
        from txn.manager import TransactionManager
        from engine import Engine
        e = Engine()
        tm = TransactionManager()

        e.execute("CREATE TABLE dr3 (id INT, val INT);")
        e.execute("INSERT INTO dr3 VALUES (1, 100), (2, 200);")

        txn_id = tm.begin()
        tm.snapshot_table('dr3', e._catalog)

        # Verify snapshot is deep copy
        snap = tm._current_txn.snapshots['dr3']
        assert snap[0] == [1, 100]
        assert snap[1] == [2, 200]

        e.execute("UPDATE dr3 SET val = 999 WHERE id = 1;")

        # Verify snapshot not mutated by UPDATE
        assert snap[0] == [1, 100], f"snapshot mutated: {snap[0]}"

        tm.rollback(e._catalog)

        # Verify via store directly
        store = e._catalog.get_store('dr3')
        rows = store.read_all_rows()
        assert len(rows) == 2, f"row count: {len(rows)}"
        assert rows[0] == [1, 100], f"row 0: {rows[0]}"
        assert rows[1] == [2, 200], f"row 1: {rows[1]}"

        e.execute("DROP TABLE dr3;")

    def test_snapshot(self):
        from txn.manager import TransactionManager
        from engine import Engine
        e = Engine()
        tm = TransactionManager()
        e.execute("CREATE TABLE s (x INT);")
        e.execute("INSERT INTO s VALUES (10), (20);")

        txn_id = tm.begin()
        tm.snapshot_table('s', e._catalog)
        assert 's' in tm._current_txn.snapshots
        assert len(tm._current_txn.snapshots['s']) == 2
        tm.commit()
        e.execute("DROP TABLE s;")

    def test_multiple_tables(self):
        from txn.manager import TransactionManager
        from engine import Engine
        e = Engine()
        tm = TransactionManager()
        e.execute("CREATE TABLE ma (x INT);")
        e.execute("CREATE TABLE mb (y INT);")
        e.execute("INSERT INTO ma VALUES (1);")
        e.execute("INSERT INTO mb VALUES (2);")

        txn_id = tm.begin()
        tm.snapshot_table('ma', e._catalog)
        tm.snapshot_table('mb', e._catalog)

        e.execute("DELETE FROM ma;")
        e.execute("DELETE FROM mb;")

        tm.rollback(e._catalog)

        sa = e._catalog.get_store('ma')
        sb = e._catalog.get_store('mb')
        assert sa.row_count == 1, f"ma: {sa.row_count}"
        assert sb.row_count == 1, f"mb: {sb.row_count}"

        e.execute("DROP TABLE ma;")
        e.execute("DROP TABLE mb;")

    def test_commit_permanent(self):
        """After commit, changes are permanent — no rollback possible."""
        from txn.manager import TransactionManager
        from engine import Engine
        e = Engine()
        tm = TransactionManager()
        e.execute("CREATE TABLE cm (x INT);")
        e.execute("INSERT INTO cm VALUES (1), (2);")

        txn_id = tm.begin()
        tm.snapshot_table('cm', e._catalog)
        e.execute("DELETE FROM cm WHERE x = 1;")
        tm.commit()

        store = e._catalog.get_store('cm')
        assert store.row_count == 1, f"after commit: {store.row_count}"

        # Rollback after commit should be no-op
        result = tm.rollback(e._catalog)
        assert not result  # No active txn to rollback

        e.execute("DROP TABLE cm;")

    def test_snapshot_deep_copy(self):
        """Verify snapshot is truly independent of the store."""
        from txn.manager import TransactionManager
        from engine import Engine
        e = Engine()
        tm = TransactionManager()
        e.execute("CREATE TABLE dc (x INT, y VARCHAR(10));")
        e.execute("INSERT INTO dc VALUES (1, 'hello'), (2, 'world');")

        txn_id = tm.begin()
        tm.snapshot_table('dc', e._catalog)

        snap = tm._current_txn.snapshots['dc']

        # Modify store
        e.execute("UPDATE dc SET y = 'CHANGED' WHERE x = 1;")
        e.execute("INSERT INTO dc VALUES (3, 'new');")

        # Snapshot should be unchanged
        assert len(snap) == 2, f"snapshot length changed: {len(snap)}"
        assert snap[0] == [1, 'hello'], f"snapshot[0] changed: {snap[0]}"
        assert snap[1] == [2, 'world'], f"snapshot[1] changed: {snap[1]}"

        tm.rollback(e._catalog)

        store = e._catalog.get_store('dc')
        rows = store.read_all_rows()
        assert len(rows) == 2
        assert rows[0] == [1, 'hello']
        assert rows[1] == [2, 'world']

        e.execute("DROP TABLE dc;")


class TestTCPServer:
    def test_server_client(self):
        """Start server in thread, connect client, execute SQL."""
        from engine import Engine
        from server.tcp_server import Z1TCPServer
        from server.protocol import Z1Client

        engine = Engine()
        engine.execute("CREATE TABLE tcp_test (id INT, name VARCHAR(20));")
        engine.execute("INSERT INTO tcp_test VALUES (1, 'Alice'), (2, 'Bob');")

        port = 15433  # Use non-standard port for testing
        server = Z1TCPServer(engine, '127.0.0.1', port)

        # Start server in background
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.3)  # Wait for server to start

        try:
            with Z1Client('127.0.0.1', port) as client:
                # Test SELECT
                r = client.execute("SELECT * FROM tcp_test ORDER BY id;")
                assert r['status'] == 'ok'
                assert r['row_count'] == 2
                assert r['columns'] == ['id', 'name']

                # Test INSERT
                r = client.execute("INSERT INTO tcp_test VALUES (3, 'Carol');")
                assert r['status'] == 'ok'
                assert r['affected_rows'] == 1

                # Test COUNT after insert
                r = client.execute("SELECT COUNT(*) FROM tcp_test;")
                assert r['status'] == 'ok'
                assert r['rows'][0][0] == 3

                # Test error
                r = client.execute("SELECT * FROM nonexistent;")
                assert r['status'] == 'error'

                # Test aggregate
                r = client.execute("SELECT MAX(id) FROM tcp_test;")
                assert r['status'] == 'ok'
                assert r['rows'][0][0] == 3

        finally:
            server.stop()
            time.sleep(0.2)

    def test_protocol_format(self):
        """Verify wire protocol JSON format."""
        from server.protocol import Z1Client
        # Just test that the class exists and has correct interface
        client = Z1Client('127.0.0.1', 5433)
        assert hasattr(client, 'connect')
        assert hasattr(client, 'execute')
        assert hasattr(client, 'close')

    def test_multiple_queries(self):
        """Test sending multiple queries on same connection."""
        from engine import Engine
        from server.tcp_server import Z1TCPServer
        from server.protocol import Z1Client

        engine = Engine()
        port = 15434
        server = Z1TCPServer(engine, '127.0.0.1', port)
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.3)

        try:
            with Z1Client('127.0.0.1', port) as client:
                client.execute("CREATE TABLE multi (x INT);")
                for i in range(10):
                    r = client.execute(f"INSERT INTO multi VALUES ({i});")
                    assert r['status'] == 'ok'
                r = client.execute("SELECT COUNT(*) FROM multi;")
                assert r['rows'][0][0] == 10
                r = client.execute("SELECT SUM(x) FROM multi;")
                assert r['rows'][0][0] == 45
        finally:
            server.stop()
            time.sleep(0.2)


def run_txn_server_tests():
    classes = [TestLockManager, TestTransactionManager, TestTCPServer]
    total = 0; passed = 0; failed = 0
    for cls in classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith('test_'):
                continue
            total += 1
            try:
                getattr(obj, name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                import traceback
                print(f"  ✗ {cls.__name__}.{name}: {e}")
    print(f"\nTxn+Server: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Transaction & Server Tests ===")
    run_txn_server_tests()
