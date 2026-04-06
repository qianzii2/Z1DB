from __future__ import annotations
"""JIT 编译器 — AST → Python callable。
[D09] 安全措施：exec 命名空间不传 __builtins__。
注意：JIT 代码仅内部生成，不接受用户输入，安全风险可控。"""
import re
import threading
from typing import Any, Callable, Dict, List, Optional
from parser.ast import (
    AliasExpr, BinaryExpr, CaseExpr, CastExpr, ColumnRef,
    FunctionCall, InExpr, IsNullExpr, LikeExpr, Literal,
    UnaryExpr, BetweenExpr, AggregateCall, StarExpr)

_SAFE_COL_RE = re.compile(r'^[a-zA-Z0-9_.]+$')
_counter_lock = threading.Lock()
_counter = 0


def _next_counter() -> int:
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


def _is_safe_col(name: str) -> bool:
    """检查列名是否安全（仅字母数字和点号）。"""
    return bool(_SAFE_COL_RE.match(name))


# [D09] 安全命名空间：禁止访问内建函数
_SAFE_NS: dict = {'__builtins__': {}}


class ExprCompiler:
    """AST → Python 函数编译器。支持行式和列式两种目标。"""

    # ═══ 行式编译 ═══

    @classmethod
    def compile_predicate(cls, expr: Any) -> Optional[Callable]:
        """编译 WHERE → function(row_dict) → bool。"""
        try:
            code = cls._expr_to_code(expr)
            if code is None:
                return None
            fn_name = f'_jit_pred_{_next_counter()}'
            source = (
                f"def {fn_name}(r):\n"
                f"    try:\n"
                f"        _v = {code}\n"
                f"        return bool(_v) if _v is not None else False\n"
                f"    except:\n"
                f"        return False\n")
            ns = dict(_SAFE_NS)  # [D09]
            exec(compile(source, f'<jit:{fn_name}>', 'exec'), ns)
            return ns[fn_name]
        except Exception:
            return None

    @classmethod
    def compile_projection(cls, exprs: list,
                           names: list) -> Optional[Callable]:
        """编译 SELECT → function(row_dict) → list。"""
        try:
            codes = []
            for expr in exprs:
                code = cls._expr_to_code(expr)
                if code is None:
                    return None
                codes.append(code)
            fn_name = f'_jit_proj_{_next_counter()}'
            body = ', '.join(codes)
            source = (
                f"def {fn_name}(r):\n"
                f"    try:\n"
                f"        return [{body}]\n"
                f"    except:\n"
                f"        return [None] * {len(codes)}\n")
            ns = dict(_SAFE_NS)
            exec(compile(source, f'<jit:{fn_name}>', 'exec'), ns)
            return ns[fn_name]
        except Exception:
            return None

    # ═══ 列式编译 ═══

    @classmethod
    def compile_columnar_predicate(cls, expr: Any) -> Optional[Callable]:
        """编译 WHERE → function(columns, n, Bitmap) → Bitmap。
        直接在列向量上操作，无需逐行构建 row_dict。"""
        try:
            col_code = cls._expr_to_columnar_code(expr)
            if col_code is None:
                return None
            fn_name = f'_jit_col_pred_{_next_counter()}'
            source = (
                f"def {fn_name}(cols, n, Bitmap):\n"
                f"    bm = Bitmap(n)\n"
                f"    for _i in range(n):\n"
                f"        try:\n"
                f"            _v = {col_code}\n"
                f"            if _v is not None and _v:\n"
                f"                bm.set_bit(_i)\n"
                f"        except:\n"
                f"            pass\n"
                f"    return bm\n")
            ns = dict(_SAFE_NS)
            exec(compile(source, f'<jit:{fn_name}>', 'exec'), ns)
            return ns[fn_name]
        except Exception:
            return None

    @classmethod
    def compile_columnar_projection(cls, exprs: list,
                                     names: list) -> Optional[Callable]:
        """编译 SELECT → function(columns, n) → Dict[str, list]。"""
        try:
            codes = []
            for expr in exprs:
                code = cls._expr_to_columnar_code(expr)
                if code is None:
                    return None
                codes.append(code)
            fn_name = f'_jit_col_proj_{_next_counter()}'
            result_init = ', '.join(
                f'{repr(nm)}: []' for nm in names)
            appends = '\n'.join(
                f"            result[{repr(names[i])}].append({codes[i]})"
                for i in range(len(codes)))
            fallbacks = '\n'.join(
                f"            result[{repr(nm)}].append(None)"
                for nm in names)
            source = (
                f"def {fn_name}(cols, n):\n"
                f"    result = {{{result_init}}}\n"
                f"    for _i in range(n):\n"
                f"        try:\n"
                f"{appends}\n"
                f"        except:\n"
                f"{fallbacks}\n"
                f"    return result\n")
            ns = dict(_SAFE_NS)
            exec(compile(source, f'<jit:{fn_name}>', 'exec'), ns)
            return ns[fn_name]
        except Exception:
            return None

    # ═══ 列式代码生成 ═══

    @classmethod
    def _expr_to_columnar_code(cls, expr) -> Optional[str]:
        if isinstance(expr, Literal):
            if expr.value is None: return 'None'
            if isinstance(expr.value, str): return repr(expr.value)
            if isinstance(expr.value, bool):
                return 'True' if expr.value else 'False'
            return repr(expr.value)
        if isinstance(expr, ColumnRef):
            col = expr.column
            if expr.table: col = f"{expr.table}.{expr.column}"
            if not _is_safe_col(col): return None
            return f"cols[{col!r}].get(_i)"
        if isinstance(expr, AliasExpr):
            return cls._expr_to_columnar_code(expr.expr)
        if isinstance(expr, BinaryExpr):
            left = cls._expr_to_columnar_code(expr.left)
            right = cls._expr_to_columnar_code(expr.right)
            if left is None or right is None: return None
            op_map = {
                '+': '+', '-': '-', '*': '*', '/': '/',
                '%': '%', '=': '==', '!=': '!=', '<': '<',
                '>': '>', '<=': '<=', '>=': '>=',
                'AND': 'and', 'OR': 'or'}
            pyop = op_map.get(expr.op)
            if pyop is None:
                if expr.op == '||':
                    return (f"(str({left}) if {left} is not None else '')"
                            f" + (str({right}) if {right} is not None else '')")
                return None
            if expr.op in ('AND', 'OR'):
                return f"({left} {pyop} {right})"
            return (f"(None if ({left} is None or {right} is None)"
                    f" else ({left} {pyop} {right}))")
        if isinstance(expr, UnaryExpr):
            operand = cls._expr_to_columnar_code(expr.operand)
            if operand is None: return None
            if expr.op == '-':
                return f"(None if {operand} is None else (-{operand}))"
            if expr.op == '+': return operand
            if expr.op == 'NOT':
                return f"(None if {operand} is None else (not {operand}))"
            return None
        if isinstance(expr, IsNullExpr):
            inner = cls._expr_to_columnar_code(expr.expr)
            if inner is None: return None
            return (f"({inner} is not None)"
                    if expr.negated
                    else f"({inner} is None)")
        if isinstance(expr, BetweenExpr):
            e = cls._expr_to_columnar_code(expr.expr)
            lo = cls._expr_to_columnar_code(expr.low)
            hi = cls._expr_to_columnar_code(expr.high)
            if e is None or lo is None or hi is None:
                return None
            core = f"({lo} <= {e} <= {hi})"
            return f"(not {core})" if expr.negated else core
        if isinstance(expr, InExpr):
            e = cls._expr_to_columnar_code(expr.expr)
            if e is None: return None
            val_codes = []
            for v in expr.values:
                vc = cls._expr_to_columnar_code(v)
                if vc is None: return None
                val_codes.append(vc)
            vals = ', '.join(val_codes)
            core = f"({e} in ({vals},))"
            return f"(not {core})" if expr.negated else core
        return None

    # ═══ 行式代码生成 ═══

    @classmethod
    def _expr_to_code(cls, expr) -> Optional[str]:
        if isinstance(expr, Literal):
            if expr.value is None: return 'None'
            if isinstance(expr.value, str): return repr(expr.value)
            if isinstance(expr.value, bool):
                return 'True' if expr.value else 'False'
            return repr(expr.value)
        if isinstance(expr, ColumnRef):
            col = expr.column
            if expr.table: col = f"{expr.table}.{expr.column}"
            if not _is_safe_col(col): return None
            return f"r.get({col!r})"
        if isinstance(expr, AliasExpr):
            return cls._expr_to_code(expr.expr)
        if isinstance(expr, BinaryExpr):
            left = cls._expr_to_code(expr.left)
            right = cls._expr_to_code(expr.right)
            if left is None or right is None: return None
            op_map = {
                '+': '+', '-': '-', '*': '*', '/': '/',
                '%': '%', '=': '==', '!=': '!=', '<': '<',
                '>': '>', '<=': '<=', '>=': '>=',
                'AND': 'and', 'OR': 'or'}
            pyop = op_map.get(expr.op)
            if pyop is None:
                if expr.op == '||':
                    return (f"(str({left} if {left} is not None else '')"
                            f" + str({right} if {right} is not None else ''))")
                return None
            if expr.op in ('AND', 'OR'):
                return f"({left} {pyop} {right})"
            return (f"(None if ({left} is None or {right} is None)"
                    f" else ({left} {pyop} {right}))")
        if isinstance(expr, UnaryExpr):
            operand = cls._expr_to_code(expr.operand)
            if operand is None: return None
            if expr.op == '-':
                return f"(None if {operand} is None else (-{operand}))"
            if expr.op == '+': return operand
            if expr.op == 'NOT':
                return f"(None if {operand} is None else (not {operand}))"
            return None
        if isinstance(expr, IsNullExpr):
            inner = cls._expr_to_code(expr.expr)
            if inner is None: return None
            return (f"({inner} is not None)"
                    if expr.negated
                    else f"({inner} is None)")
        if isinstance(expr, BetweenExpr):
            e = cls._expr_to_code(expr.expr)
            lo = cls._expr_to_code(expr.low)
            hi = cls._expr_to_code(expr.high)
            if e is None or lo is None or hi is None:
                return None
            core = f"({lo} <= {e} <= {hi})"
            return f"(not {core})" if expr.negated else core
        if isinstance(expr, InExpr):
            e = cls._expr_to_code(expr.expr)
            if e is None: return None
            val_codes = []
            for v in expr.values:
                vc = cls._expr_to_code(v)
                if vc is None: return None
                val_codes.append(vc)
            vals = ', '.join(val_codes)
            core = f"({e} in ({vals},))"
            return f"(not {core})" if expr.negated else core
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            args_code = []
            for a in expr.args:
                ac = cls._expr_to_code(a)
                if ac is None: return None
                args_code.append(ac)
            if name == 'UPPER' and len(args_code) == 1:
                return (f"(str({args_code[0]}).upper() "
                        f"if {args_code[0]} is not None else None)")
            if name == 'LOWER' and len(args_code) == 1:
                return (f"(str({args_code[0]}).lower() "
                        f"if {args_code[0]} is not None else None)")
            if name == 'LENGTH' and len(args_code) == 1:
                return (f"(len(str({args_code[0]})) "
                        f"if {args_code[0]} is not None else None)")
            if name == 'ABS' and len(args_code) == 1:
                return (f"(abs({args_code[0]}) "
                        f"if {args_code[0]} is not None else None)")
            if name == 'COALESCE':
                result = args_code[-1]
                for i in range(len(args_code) - 2, -1, -1):
                    result = (f"({args_code[i]} "
                              f"if {args_code[i]} is not None "
                              f"else {result})")
                return result
            return None
        return None

    @classmethod
    def clear_cache(cls) -> None:
        global _counter
        with _counter_lock:
            _counter = 0
