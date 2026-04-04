from __future__ import annotations
"""JIT compiler: AST expressions → compiled Python callables."""
import dataclasses
from typing import Any, Callable, Dict, Optional
from parser.ast import (
    AliasExpr, BinaryExpr, CaseExpr, CastExpr, ColumnRef, FunctionCall,
    InExpr, IsNullExpr, LikeExpr, Literal, UnaryExpr, BetweenExpr,
    AggregateCall, StarExpr,
)
from storage.types import DataType


class ExprCompiler:
    """Compiles AST expressions into Python functions via compile()+exec()."""

    _cache: Dict[str, Callable] = {}
    _counter: int = 0

    @classmethod
    def compile_predicate(cls, expr: Any) -> Optional[Callable]:
        """Compile a WHERE predicate into a function(row_dict) → bool.
        Returns None if expression is too complex to compile."""
        try:
            code = cls._expr_to_code(expr)
            if code is None:
                return None
            fn_name = f'_jit_pred_{cls._counter}'
            cls._counter += 1
            source = f"def {fn_name}(r):\n    try:\n        return bool({code})\n    except:\n        return False\n"
            ns: dict = {}
            compiled = compile(source, f'<jit:{fn_name}>', 'exec')
            exec(compiled, ns)
            fn = ns[fn_name]
            return fn
        except Exception:
            return None

    @classmethod
    def compile_projection(cls, exprs: list, names: list) -> Optional[Callable]:
        """Compile a set of projections into a function(row_dict) → list."""
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
            source = f"def {fn_name}(r):\n    try:\n        return [{body}]\n    except:\n        return [None] * {len(codes)}\n"
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
                return f"(str({left}) + str({right}))"
            return f"({left} {pyop} {right})"

        if isinstance(expr, UnaryExpr):
            operand = cls._expr_to_code(expr.operand)
            if operand is None:
                return None
            if expr.op == '-':
                return f"(-{operand})"
            if expr.op == '+':
                return operand
            if expr.op == 'NOT':
                return f"(not {operand})"
            return None

        if isinstance(expr, IsNullExpr):
            inner = cls._expr_to_code(expr.expr)
            if inner is None:
                return None
            if expr.negated:
                return f"({inner} is not None)"
            return f"({inner} is None)"

        # Too complex for JIT
        return None
