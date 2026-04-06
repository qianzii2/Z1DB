from __future__ import annotations
"""Z1DB 入口。支持 REPL 和 TCP 服务器模式。"""
import sys
from engine import Engine


def main() -> None:
    args = sys.argv[1:]
    server_mode = False
    host = '0.0.0.0'
    port = 5433
    data_dir = ':memory:'

    i = 0
    while i < len(args):
        if args[i] == '--server':
            server_mode = True
            if (i + 1 < len(args)
                    and not args[i + 1].startswith('-')):
                addr = args[i + 1]
                if ':' in addr:
                    host, port_str = addr.rsplit(':', 1)
                    port = int(port_str)
                else:
                    port = int(addr)
                i += 1
        elif not args[i].startswith('-'):
            data_dir = args[i]
        i += 1

    engine = Engine(data_dir)

    if server_mode:
        from server.tcp_server import Z1TCPServer
        server = Z1TCPServer(engine, host, port)
        server.start()
    else:
        from server.client import REPL
        REPL(engine).run()


if __name__ == '__main__':
    main()
