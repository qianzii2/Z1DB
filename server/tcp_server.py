from __future__ import annotations
"""Minimal TCP server for Z1DB. Single-threaded event loop."""
import json
import selectors
import socket
import traceback
from typing import Any, Dict, Optional


class Z1TCPServer:
    """Simple TCP server using selectors for async I/O."""

    def __init__(self, engine: Any, host: str = '0.0.0.0', port: int = 5433) -> None:
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
        print(f"Z1DB TCP server listening on {self._host}:{self._port}")
        try:
            while self._running:
                events = self._sel.select(timeout=1.0)
                for key, mask in events:
                    if key.data is None:
                        self._accept(key.fileobj)
                    else:
                        self._handle(key, mask)
        except KeyboardInterrupt:
            print("\nServer shutting down.")
        finally:
            self._sel.close()
            sock.close()

    def stop(self) -> None:
        self._running = False

    def _accept(self, sock: socket.socket) -> None:
        conn, addr = sock.accept()
        conn.setblocking(False)
        session = _Session(conn, addr, self._engine)
        self._sel.register(conn, selectors.EVENT_READ | selectors.EVENT_WRITE, data=session)

    def _handle(self, key: selectors.SelectorKey, mask: int) -> None:
        session: _Session = key.data
        try:
            if mask & selectors.EVENT_READ:
                data = session.conn.recv(65536)
                if not data:
                    self._close(key)
                    return
                session.buffer += data
                self._process(session)
            if mask & selectors.EVENT_WRITE:
                if session.outgoing:
                    sent = session.conn.send(session.outgoing)
                    session.outgoing = session.outgoing[sent:]
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._close(key)

    def _process(self, session: _Session) -> None:
        while b'\n' in session.buffer:
            line, session.buffer = session.buffer.split(b'\n', 1)
            sql = line.decode('utf-8', errors='replace').strip()
            if not sql:
                continue
            try:
                result = session.engine.execute(sql)
                response = {
                    'status': 'ok',
                    'columns': result.columns,
                    'column_types': [dt.name for dt in result.column_types] if result.column_types else [],
                    'rows': result.rows,
                    'row_count': result.row_count,
                    'affected_rows': result.affected_rows,
                    'message': result.message,
                    'timing': result.timing,
                }
            except Exception as e:
                response = {
                    'status': 'error',
                    'message': str(e),
                }
            resp_bytes = json.dumps(response, default=str).encode('utf-8') + b'\n'
            session.outgoing += resp_bytes

    def _close(self, key: selectors.SelectorKey) -> None:
        self._sel.unregister(key.fileobj)
        key.fileobj.close()


class _Session:
    __slots__ = ('conn', 'addr', 'engine', 'buffer', 'outgoing')

    def __init__(self, conn: socket.socket, addr: tuple, engine: Any) -> None:
        self.conn = conn
        self.addr = addr
        self.engine = engine
        self.buffer = b''
        self.outgoing = b''
