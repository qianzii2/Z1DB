from __future__ import annotations
"""Wire protocol for Z1DB TCP communication.
Format: each message is a JSON line terminated by \\n.

Client → Server: {"sql": "SELECT ..."}
Server → Client: {"status": "ok", "columns": [...], "rows": [...], ...}
                  or {"status": "error", "message": "..."}
"""
import json
import socket
from typing import Any, Dict, List, Optional


class Z1Client:
    """Simple TCP client for Z1DB server."""

    def __init__(self, host: str = '127.0.0.1', port: int = 5433) -> None:
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._host, self._port))

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def execute(self, sql: str) -> Dict[str, Any]:
        """Send SQL, receive result as dict."""
        if self._sock is None:
            raise ConnectionError("Not connected")
        # Send
        msg = sql.strip() + '\n'
        self._sock.sendall(msg.encode('utf-8'))
        # Receive until newline
        data = b''
        while b'\n' not in data:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("Server closed connection")
            data += chunk
        line = data.split(b'\n')[0]
        return json.loads(line.decode('utf-8'))

    def __enter__(self) -> Z1Client:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
