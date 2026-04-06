#!/usr/bin/env python3
"""Z1DB 测试运行器 — 无需 pytest。
用法: python tests/run_tests.py [模块名] [用例名]
示例: python tests/run_tests.py test_select test_where_eq"""
from __future__ import annotations
import importlib, os, sys, time, traceback

# 确保项目根目录在 path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS = 0; FAIL = 0; ERROR = 0; SKIP = 0

def run_module(mod_name: str, filter_name: str = ''):
    global PASS, FAIL, ERROR, SKIP
    try:
        mod = importlib.import_module(f'tests.{mod_name}')
    except Exception as e:
        print(f"  ❌ 无法加载 {mod_name}: {e}")
        ERROR += 1; return
    funcs = [(n, getattr(mod, n)) for n in sorted(dir(mod))
             if n.startswith('test_') and callable(getattr(mod, n))]
    if filter_name:
        funcs = [(n, f) for n, f in funcs if filter_name in n]
    for name, fn in funcs:
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            print(f"  ✅ {name} ({elapsed*1000:.1f}ms)")
            PASS += 1
        except AssertionError as e:
            elapsed = time.perf_counter() - t0
            print(f"  ❌ {name} ({elapsed*1000:.1f}ms)")
            print(f"     {e}")
            FAIL += 1
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  💥 {name} ({elapsed*1000:.1f}ms)")
            traceback.print_exc()
            ERROR += 1

ALL_MODULES = [
    'test_lexer', 'test_parser', 'test_resolver', 'test_validator',
    'test_optimizer', 'test_evaluator', 'test_types',
    'test_ddl', 'test_dml', 'test_select', 'test_join',
    'test_agg', 'test_window', 'test_set_ops', 'test_subquery',
    'test_copy', 'test_index', 'test_storage', 'test_compression',
    'test_column_chunk', 'test_wal', 'test_txn', 'test_structures',
    'test_nanboxing', 'test_bitmap', 'test_performance',
    'test_null', 'test_edge_cases', 'test_persistence', 'test_concurrent',
]

if __name__ == '__main__':
    args = sys.argv[1:]
    mod_filter = args[0] if args else ''
    test_filter = args[1] if len(args) > 1 else ''
    modules = [m for m in ALL_MODULES if mod_filter in m] if mod_filter else ALL_MODULES
    print(f"Z1DB 测试套件 — {len(modules)} 模块\n{'='*50}")
    t_start = time.perf_counter()
    for mod in modules:
        print(f"\n📦 {mod}")
        run_module(mod, test_filter)
    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*50}")
    print(f"✅ {PASS} 通过  ❌ {FAIL} 失败  💥 {ERROR} 错误  ⏱ {elapsed:.2f}s")
    sys.exit(1 if FAIL + ERROR > 0 else 0)
