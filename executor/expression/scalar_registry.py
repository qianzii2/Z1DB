from __future__ import annotations
"""标量函数注册表 — 替代evaluator中的1000行if-elif链。
所有标量函数注册到此处，evaluator通过dict查找O(1)调用。"""
import datetime as _dt
import math as _math
import random as _random
import re as _re
from typing import Any, Callable, Dict, List, Optional, Tuple
from storage.types import DataType

# 函数签名: (args: List[DataVector], n: int, evaluator) -> DataVector
_FunctionEntry = Tuple[Callable, DataType]  # (fn, return_type)


class ScalarFunctionRegistry:
    """标量函数注册表。O(1)查找替代O(k)字符串比较。"""

    _instance: Optional[ScalarFunctionRegistry] = None
    _fns: Dict[str, _FunctionEntry] = {}
    _type_fns: Dict[str, Callable] = {}  # 返回类型推断函数

    @classmethod
    def instance(cls) -> ScalarFunctionRegistry:
        if cls._instance is None:
            cls._instance = ScalarFunctionRegistry()
        return cls._instance

    @classmethod
    def register(cls, name: str, return_type: DataType) -> Callable:
        """装饰器：注册标量函数。"""
        def decorator(fn: Callable) -> Callable:
            cls._fns[name.upper()] = (fn, return_type)
            return fn
        return decorator

    @classmethod
    def register_dynamic_type(cls, name: str,
                              type_fn: Callable) -> Callable:
        """注册返回类型依赖输入的函数。"""
        def decorator(fn: Callable) -> Callable:
            cls._fns[name.upper()] = (fn, DataType.UNKNOWN)
            cls._type_fns[name.upper()] = type_fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[_FunctionEntry]:
        return cls._fns.get(name.upper())

    @classmethod
    def has(cls, name: str) -> bool:
        return name.upper() in cls._fns

    @classmethod
    def get_return_type(cls, name: str, args: list,
                        schema: dict) -> DataType:
        upper = name.upper()
        if upper in cls._type_fns:
            return cls._type_fns[upper](args, schema)
        entry = cls._fns.get(upper)
        if entry:
            return entry[1]
        return DataType.UNKNOWN

    @classmethod
    def all_names(cls) -> List[str]:
        return list(cls._fns.keys())
