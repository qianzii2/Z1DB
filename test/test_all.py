from __future__ import annotations
"""Z1DB complete test suite — run all tests."""
import sys
import time
import traceback


def run_test(name: str, fn: callable) -> bool:
    try:
        fn()
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()
        return False


def main():
    start = time.perf_counter()
    passed = 0
    failed = 0
    errors = []

    # Import all test modules
    from test import test_metal
    from test import test_structures
    from test import test_sql
    from test import test_advanced
    from test import test_integration

    suites = [
        ("Metal Layer", test_metal.ALL_TESTS),
        ("Data Structures", test_structures.ALL_TESTS),
        ("SQL Core", test_sql.ALL_TESTS),
        ("Advanced Features", test_advanced.ALL_TESTS),
        ("Integration", test_integration.ALL_TESTS),
    ]

    for suite_name, tests in suites:
        print(f"\n{'═' * 60}")
        print(f"  {suite_name}")
        print(f"{'═' * 60}")
        for name, fn in tests:
            ok = run_test(name, fn)
            if ok:
                passed += 1
            else:
                failed += 1
                errors.append(name)

    elapsed = time.perf_counter() - start
    print(f"\n{'═' * 60}")
    print(f"  Results: {passed} passed, {failed} failed ({elapsed:.2f}s)")
    if errors:
        print(f"  Failed: {', '.join(errors)}")
    print(f"{'═' * 60}")
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
