from __future__ import annotations
"""TCP 服务器。[M04] 每连接独立事务上下文。"""
import json
import selectors
import socket
from typing import Any, Optional


class Z1TCPServer:
    def __init__(self, engine: Any, host: str = '0.0.0.0',
                 port: int = 5433) -> None:
        self._engine = engine
        self._host = host
        self._port = port
        self._sel = selectors.DefaultSelector()
        self._running = False

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen(100)
        sock.setblocking(False)
        self._sel.register(sock, selectors.EVENT_READ, data=None)
        self._running = True
        print(f"Z1DB TCP 服务器监听 {self._host}:{self._port}")
        try:
            while self._running:
                events = self._sel.select(timeout=1.0)
                for key, mask in events:
                    if key.data is None:
                        self._accept(key.fileobj)
                    else:
                        self._handle(key, mask)
        except KeyboardInterrupt:
            print("\n服务器关闭。")
        finally:
            self._sel.close()
            sock.close()

    def stop(self) -> None:
        self._running = False

    def _accept(self, sock) -> None:
        conn, addr = sock.accept()
        conn.setblocking(False)
        # [M04] 每连接独立会话
        session = _Session(conn, addr, self._engine)
        self._sel.register(
            conn,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            data=session)

    def _handle(self, key, mask) -> None:
        session = key.data
        try:
            if mask & selectors.EVENT_READ:
                data = session.conn.recv(65536)
                if not data:
                    self._close_session(key)
                    return
                session.buffer += data
                self._process(session)
            if mask & selectors.EVENT_WRITE:
                if session.outgoing:
                    sent = session.conn.send(session.outgoing)
                    session.outgoing = session.outgoing[sent:]
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._close_session(key)

    def _process(self, session) -> None:
        while b'\n' in session.buffer:
            line, session.buffer = session.buffer.split(b'\n', 1)
            sql = line.decode('utf-8', errors='replace').strip()
            if not sql:
                continue
            try:
                # [M04] 会话级事务命令隔离
                upper = sql.strip().rstrip(';').strip().upper()
                if upper == 'BEGIN' and session.in_txn:
                    response = {'status': 'error',
                                'message': '会话已有活跃事务'}
                elif upper == 'BEGIN':
                    session.in_txn = True
                    result = session.engine.execute(sql)
                    session.txn_sql_count = 0
                    response = self._result_to_dict(result)
                elif upper in ('COMMIT', 'ROLLBACK'):
                    result = session.engine.execute(sql)
                    session.in_txn = False
                    session.txn_sql_count = 0
                    response = self._result_to_dict(result)
                else:
                    if session.in_txn:
                        session.txn_sql_count += 1
                    result = session.engine.execute(sql)
                    response = self._result_to_dict(result)
            except Exception as e:
                response = {
                    'status': 'error',
                    'message': str(e),
                }
            resp_bytes = (json.dumps(response, default=str)
                          .encode('utf-8') + b'\n')
            session.outgoing += resp_bytes

    @staticmethod
    def _result_to_dict(result) -> dict:
        return {
            'status': 'ok',
            'columns': result.columns,
            'column_types': ([dt.name for dt in result.column_types]
                             if result.column_types else []),
            'rows': result.rows,
            'row_count': result.row_count,
            'affected_rows': result.affected_rows,
            'message': result.message,
            'timing': result.timing,
        }

    def _close_session(self, key) -> None:
        """[M04] 关闭会话时如果有未提交事务，自动回滚。"""
        session = key.data
        if session.in_txn:
            try:
                session.engine.execute('ROLLBACK;')
            except Exception:
                pass
        self._sel.unregister(key.fileobj)
        key.fileobj.close()


class _Session:
    """[M04] 每连接会话状态。"""
    __slots__ = ('conn', 'addr', 'engine', 'buffer',
                 'outgoing', 'in_txn', 'txn_sql_count')

    def __init__(self, conn, addr, engine) -> None:
        self.conn = conn
        self.addr = addr
        self.engine = engine
        self.buffer = b''
        self.outgoing = b''
        self.in_txn = False       # [M04] 会话事务状态
        self.txn_sql_count = 0    # 当前事务中的语句数
