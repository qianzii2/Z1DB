from __future__ import annotations
"""NULL three-valued-logic reference.

This module serves as living documentation.  Actual logic is implemented
inline in the evaluator and operators.

Rules
-----
- ``x OP NULL`` (arithmetic / comparison) → ``NULL``
- ``NULL = NULL`` → ``NULL`` (not TRUE)
- ``NULL AND FALSE`` → ``FALSE``; ``NULL AND TRUE`` → ``NULL``
- ``NULL OR TRUE`` → ``TRUE``; ``NULL OR FALSE`` → ``NULL``
- ``NOT NULL`` → ``NULL``
- ``IS NULL`` / ``IS NOT NULL`` → boolean (never NULL)
- ``COUNT(*)`` includes NULL rows; ``COUNT(col)`` skips NULLs; empty set → 0
- ``SUM/AVG/MIN/MAX`` on empty set → ``NULL``
- ``ORDER BY``: NULLs placement controlled by ``NULLS FIRST`` / ``NULLS LAST``
- ``GROUP BY``: NULL values go into the same group
- ``DISTINCT``: NULLs are considered equal
- ``hash_value(NULL)`` → ``NULL_HASH_SENTINEL`` (0)
"""
