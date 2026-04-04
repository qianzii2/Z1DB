from __future__ import annotations
"""Z1DB entry point."""

import sys

from engine import Engine
from server.client import REPL


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else ':memory:'
    engine = Engine(data_dir)
    REPL(engine).run()


if __name__ == '__main__':
    main()
