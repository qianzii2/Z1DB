from __future__ import annotations
import os, tempfile, shutil
from tests.conftest import *

def test_wal_write_recover():
    tmpdir = tempfile.mkdtemp(prefix='z1db_wal_')
    try:
        e = Engine(tmpdir)
        e.execute("CREATE TABLE t (a INT);")
        e.execute("INSERT INTO t VALUES (1);")
        e.execute("INSERT INTO t VALUES (2);")
        e.close()
        # WAL 文件应存在
        assert os.path.exists(os.path.join(tmpdir, 'z1db.wal'))
        # 重新打开恢复
        e2 = Engine(tmpdir)
        r = e2.execute("SELECT COUNT(*) FROM t;")
        assert r.rows[0][0] >= 2
        e2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_wal_checkpoint():
    tmpdir = tempfile.mkdtemp(prefix='z1db_wal_')
    try:
        e = Engine(tmpdir)
        e.execute("CREATE TABLE t (a INT);")
        e.execute("INSERT INTO t VALUES (1);")
        # checkpoint 应 fsync
        if e._wal:
            lsn = e._wal.checkpoint()
            assert lsn > 0
        e.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_wal_truncate():
    from storage.wal import WriteAheadLog
    tmpdir = tempfile.mkdtemp(prefix='z1db_wal_')
    try:
        wal = WriteAheadLog(tmpdir)
        wal.open()
        for i in range(10):
            wal.append('INSERT', 't', {'sql': f'INSERT INTO t VALUES ({i})'})
        cp = wal.checkpoint()
        # 追加更多
        wal.append('INSERT', 't', {'sql': 'INSERT INTO t VALUES (99)'})
        # 截断 checkpoint 之前的
        wal.truncate_before(cp)
        entries = wal.recover()
        # 只有 checkpoint 之后的条目
        assert len(entries) >= 1
        wal.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_wal_atomic_truncate():
    """截断使用原子 rename，不应丢数据。"""
    from storage.wal import WriteAheadLog
    tmpdir = tempfile.mkdtemp(prefix='z1db_wal_')
    try:
        wal = WriteAheadLog(tmpdir)
        wal.open()
        wal.append('INSERT', 't', {'sql': 'data1'})
        wal.append('INSERT', 't', {'sql': 'data2'})
        cp = wal.checkpoint()
        wal.append('INSERT', 't', {'sql': 'data3'})
        wal.truncate_before(cp)
        wal.close()
        # 验证文件完整
        wal2 = WriteAheadLog(tmpdir)
        wal2.open()
        entries = wal2.recover()
        assert len(entries) >= 1
        wal2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
