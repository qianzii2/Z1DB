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
    print("\n=== 1/9: Metal Layer ===")
    total_failed += run_metal_tests()

    from test_structures import run_structure_tests
    print("\n=== 2/9: Data Structures ===")
    total_failed += run_structure_tests()

    from test_compression import run_compression_tests
    print("\n=== 3/9: Compression ===")
    total_failed += run_compression_tests()

    from test_sketch import run_sketch_tests
    print("\n=== 4/9: Sketches ===")
    total_failed += run_sketch_tests()

    from test_planner import run_planner_tests
    print("\n=== 5/9: Planner ===")
    total_failed += run_planner_tests()

    from test_sql import run_sql_tests
    print("\n=== 6/9: SQL End-to-End ===")
    total_failed += run_sql_tests()

    from test_txn_server import run_txn_server_tests
    print("\n=== 7/9: Transactions & Server ===")
    total_failed += run_txn_server_tests()

    from test_recursive_correlated import run_recursive_correlated_tests
    print("\n=== 8/9: Recursive CTE & Correlated Subquery ===")
    total_failed += run_recursive_correlated_tests()

    from test_binary_correlated import run_binary_correlated_tests
    print("\n=== 9/9: Binary Persistence & Correlated Subquery ===")
    total_failed += run_binary_correlated_tests()


    print("\n" + "=" * 60)
    if total_failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {total_failed} TESTS FAILED")
    print("=" * 60)
    sys.exit(1 if total_failed else 0)


if __name__ == '__main__':
    main()
