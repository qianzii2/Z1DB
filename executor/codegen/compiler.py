from __future__ import annotations
"""JIT compiler: AST expressions → compiled Python callables.
Avoids per-row AST traversal. 5-20x speedup for simple predicates."""
from typing import Any, Callable, Dict, List, Optional
from parser.ast import (
    AliasExpr, BinaryExpr, CaseExpr, CastExpr, ColumnRef, FunctionCall,
    InExpr, IsNullExpr, LikeExpr, Literal, UnaryExpr, BetweenExpr,
    AggregateCall, StarExpr,
)


class ExprCompiler:
    """Compiles AST expressions into Python functions via compile()+exec()."""

    _cache: Dict[str, Callable] = {}
    _counter: int = 0

    @classmethod
    def compile_predicate(cls, expr: Any) -> Optional[Callable]:
        """Compile WHERE predicate → function(row_dict) → bool."""
        try:
            code = cls._expr_to_code(expr)
            if code is None:
                return None
            fn_name = f'_jit_pred_{cls._counter}'
            cls._counter += 1
            source = (
                f"def {fn_name}(r):\n"
                f"    try:\n"
                f"        _v = {code}\n"
                f"        return bool(_v) if _v is not None else False\n"
                f"    except:\n"
                f"        return False\n"
            )
            ns: dict = {}
            compiled = compile(source, f'<jit:{fn_name}>', 'exec')
            exec(compiled, ns)
            return ns[fn_name]
        except Exception:
            return None

    @classmethod
    def compile_projection(cls, exprs: list, names: list) -> Optional[Callable]:
        """Compile SELECT list → function(row_dict) → list of values."""
        try:
            codes = []
            for expr in exprs:
                code = cls._expr_to_code(expr)
                if code is None:
                    return None
                codes.append(code)
            fn_name = f'_jit_proj_{cls._counter}'
            cls._counter += 1
            body = ', '.join(codes)
            source = (
                f"def {fn_name}(r):\n"
                f"    try:\n"
                f"        return [{body}]\n"
                f"    except:\n"
                f"        return [None] * {len(codes)}\n"
            )
            ns: dict = {}
            compiled = compile(source, f'<jit:{fn_name}>', 'exec')
            exec(compiled, ns)
            return ns[fn_name]
        except Exception:
            return None

    @classmethod
    def compile_hash_key(cls, exprs: list) -> Optional[Callable]:
        """Compile GROUP BY key computation → function(row_dict) → tuple."""
        try:
            codes = []
            for expr in exprs:
                code = cls._expr_to_code(expr)
                if code is None:
                    return None
                codes.append(code)
            fn_name = f'_jit_key_{cls._counter}'
            cls._counter += 1
            body = ', '.join(codes)
            if len(codes) == 1:
                body += ','  # single-element tuple
            source = (
                f"def {fn_name}(r):\n"
                f"    try:\n"
                f"        return ({body})\n"
                f"    except:\n"
                f"        return (None,) * {len(codes)}\n"
            )
            ns: dict = {}
            compiled = compile(source, f'<jit:{fn_name}>', 'exec')
            exec(compiled, ns)
            return ns[fn_name]
        except Exception:
            return None

    @classmethod
    def _expr_to_code(cls, expr: Any) -> Optional[str]:
        if isinstance(expr, Literal):
            if expr.value is None:
                return 'None'
            if isinstance(expr.value, str):
                return repr(expr.value)
            if isinstance(expr.value, bool):
                return 'True' if expr.value else 'False'
            return repr(expr.value)

        if isinstance(expr, ColumnRef):
            col = expr.column
            if expr.table:
                col = f"{expr.table}.{expr.column}"
            return f"r.get({col!r})"

        if isinstance(expr, AliasExpr):
            return cls._expr_to_code(expr.expr)

        if isinstance(expr, BinaryExpr):
            left = cls._expr_to_code(expr.left)
            right = cls._expr_to_code(expr.right)
            if left is None or right is None:
                return None
            op_map = {
                '+': '+', '-': '-', '*': '*', '/': '/',
                '%': '%', '||': '+',
                '=': '==', '!=': '!=', '<': '<', '>': '>',
                '<=': '<=', '>=': '>=',
                'AND': 'and', 'OR': 'or',
            }
            pyop = op_map.get(expr.op)
            if pyop is None:
                return None
            if expr.op == '||':
                return f"(str({left} if {left} is not None else '') + str({right} if {right} is not None else ''))"
            if expr.op in ('AND', 'OR'):
                return f"({left} {pyop} {right})"
            # NULL-safe comparison
            return f"(None if ({left} is None or {right} is None) else ({left} {pyop} {right}))"

        if isinstance(expr, UnaryExpr):
            operand = cls._expr_to_code(expr.operand)
            if operand is None:
                return None
            if expr.op == '-':
                return f"(None if {operand} is None else (-{operand}))"
            if expr.op == '+':
                return operand
            if expr.op == 'NOT':
                return f"(None if {operand} is None else (not {operand}))"
            return None

        if isinstance(expr, IsNullExpr):
            inner = cls._expr_to_code(expr.expr)
            if inner is None:
                return None
            if expr.negated:
                return f"({inner} is not None)"
            return f"({inner} is None)"

        if isinstance(expr, BetweenExpr):
            e = cls._expr_to_code(expr.expr)
            lo = cls._expr_to_code(expr.low)
            hi = cls._expr_to_code(expr.high)
            if e is None or lo is None or hi is None:
                return None
            core = f"({lo} <= {e} <= {hi})"
            if expr.negated:
                core = f"(not {core})"
            return core

        if isinstance(expr, InExpr):
            e = cls._expr_to_code(expr.expr)
            if e is None:
                return None
            val_codes = []
            for v in expr.values:
                vc = cls._expr_to_code(v)
                if vc is None:
                    return None
                val_codes.append(vc)
            vals = ', '.join(val_codes)
            core = f"({e} in ({vals},))"
            if expr.negated:
                core = f"(not {core})"
            return core

        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            args_code = []
            for a in expr.args:
                ac = cls._expr_to_code(a)
                if ac is None:
                    return None
                args_code.append(ac)
            # Simple built-in functions
            if name == 'UPPER' and len(args_code) == 1:
                return f"(str({args_code[0]}).upper() if {args_code[0]} is not None else None)"
            if name == 'LOWER' and len(args_code) == 1:
                return f"(str({args_code[0]}).lower() if {args_code[0]} is not None else None)"
            if name == 'LENGTH' and len(args_code) == 1:
                return f"(len(str({args_code[0]})) if {args_code[0]} is not None else None)"
            if name == 'ABS' and len(args_code) == 1:
                return f"(abs({args_code[0]}) if {args_code[0]} is not None else None)"
            if name == 'COALESCE':
                parts = ' if '.join(f"({c})" for c in args_code)
                # Generate nested ternary
                result = args_code[-1]
                for i in range(len(args_code) - 2, -1, -1):
                    result = f"({args_code[i]} if {args_code[i]} is not None else {result})"
                return result
            return None  # Complex function — fallback to interpreter

        # Too complex
        return None

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        cls._counter = 0
