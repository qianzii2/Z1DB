from __future__ import annotations
"""Run all Z1DB tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add test dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    total_failed = 0

    print("=" * 60)
    print("Z1DB Complete Test Suite")
    print("=" * 60)

    from test_metal import run_metal_tests
    print("\n=== 1/7: Metal Layer ===")
    total_failed += run_metal_tests()

    from test_structures import run_structure_tests
    print("\n=== 2/7: Data Structures ===")
    total_failed += run_structure_tests()

    from test_compression import run_compression_tests
    print("\n=== 3/7: Compression ===")
    total_failed += run_compression_tests()

    from test_sketch import run_sketch_tests
    print("\n=== 4/7: Sketches ===")
    total_failed += run_sketch_tests()

    from test_planner import run_planner_tests
    print("\n=== 5/7: Planner ===")
    total_failed += run_planner_tests()

    from test_sql import run_sql_tests
    print("\n=== 6/7: SQL End-to-End ===")
    total_failed += run_sql_tests()

    from test_txn_server import run_txn_server_tests
    print("\n=== 7/7: Transactions & Server ===")
    total_failed += run_txn_server_tests()

    print("\n" + "=" * 60)
    if total_failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {total_failed} TESTS FAILED")
    print("=" * 60)
    sys.exit(1 if total_failed else 0)


if __name__ == '__main__':
    main()
