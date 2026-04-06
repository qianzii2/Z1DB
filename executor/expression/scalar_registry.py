from __future__ import annotations
"""标量函数注册表 — O(1) dict 分发替代 O(k) if-elif 链。
M4 完成：所有内置函数注册。evaluator 查此表优先。"""
import datetime as _dt
import math as _math
import random as _random
import re as _re
from typing import Any, Callable, Dict, List, Optional, Tuple
from storage.types import DataType

_FunctionEntry = Tuple[Callable, DataType]


class ScalarFunctionRegistry:
    _instance: Optional[ScalarFunctionRegistry] = None
    _fns: Dict[str, _FunctionEntry] = {}
    _type_fns: Dict[str, Callable] = {}
    _initialized = False

    @classmethod
    def instance(cls) -> ScalarFunctionRegistry:
        if cls._instance is None:
            cls._instance = ScalarFunctionRegistry()
        if not cls._initialized:
            cls._initialized = True
            cls._instance._register_all()
        return cls._instance

    @classmethod
    def register(cls, name: str,
                 return_type: DataType) -> Callable:
        def decorator(fn: Callable) -> Callable:
            cls._fns[name.upper()] = (fn, return_type)
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[_FunctionEntry]:
        if not cls._initialized:
            cls.instance()
        return cls._fns.get(name.upper())

    @classmethod
    def has(cls, name: str) -> bool:
        if not cls._initialized:
            cls.instance()
        return name.upper() in cls._fns

    @classmethod
    def get_return_type(cls, name: str, args: list,
                        schema: dict) -> DataType:
        upper = name.upper()
        if upper in cls._type_fns:
            return cls._type_fns[upper](args, schema)
        entry = cls._fns.get(upper)
        return entry[1] if entry else DataType.UNKNOWN

    @classmethod
    def all_names(cls) -> List[str]:
        if not cls._initialized:
            cls.instance()
        return list(cls._fns.keys())

    def _register_all(self) -> None:
        """注册全部内置标量函数。
        每个函数签名: fn(args: List[DataVector], n: int, evaluator) -> DataVector
        evaluator 的辅助方法通过第三参数访问。"""
        R = self._fns
        V = DataType.VARCHAR
        I = DataType.INT
        BI = DataType.BIGINT
        D = DataType.DOUBLE
        B = DataType.BOOLEAN

        # ═══ 字符串函数 ═══
        def _upper(a, n, ev): return ev._str1(a[0], n, str.upper)
        def _lower(a, n, ev): return ev._str1(a[0], n, str.lower)
        def _length(a, n, ev): return ev._num1(a[0], n, lambda s: len(str(s)), I)
        def _trim(a, n, ev): return ev._str1(a[0], n, str.strip)
        def _ltrim(a, n, ev): return ev._str1(a[0], n, str.lstrip)
        def _rtrim(a, n, ev): return ev._str1(a[0], n, str.rstrip)
        def _reverse(a, n, ev): return ev._str1(a[0], n, lambda s: s[::-1])
        def _initcap(a, n, ev): return ev._str1(a[0], n, str.title)
        def _replace(a, n, ev): return ev._str3(a[0], a[1], a[2], n, lambda s, x, y: s.replace(x, y))
        def _position(a, n, ev): return ev._num2(a[0], a[1], n, lambda sub, s: str(s).find(str(sub)) + 1, I)
        def _left(a, n, ev): return ev._str_int(a[0], a[1], n, lambda s, k: s[:k])
        def _right(a, n, ev): return ev._str_int(a[0], a[1], n, lambda s, k: s[-k:] if k > 0 else '')
        def _repeat(a, n, ev): return ev._str_int(a[0], a[1], n, lambda s, k: s * k)
        def _starts_with(a, n, ev): return ev._bool2(a[0], a[1], n, lambda s, p: str(s).startswith(str(p)))
        def _ends_with(a, n, ev): return ev._bool2(a[0], a[1], n, lambda s, p: str(s).endswith(str(p)))
        def _contains(a, n, ev): return ev._bool2(a[0], a[1], n, lambda s, p: str(p) in str(s))
        def _ascii(a, n, ev): return ev._num1(a[0], n, lambda s: ord(str(s)[0]) if str(s) else 0, I)
        def _chr(a, n, ev): return ev._str1(a[0], n, lambda x: chr(int(x)) if isinstance(x, (int, float)) else '')
        def _split(a, n, ev): return ev._str2(a[0], a[1], n, lambda s, d: str(str(s).split(str(d))))
        def _regexp_replace(a, n, ev): return ev._str3(a[0], a[1], a[2], n, lambda s, p, r: _re.sub(p, r, s))
        def _regexp_extract(a, n, ev): return ev._str2(a[0], a[1], n, lambda s, p: (m.group(0) if (m := _re.search(str(p), str(s))) else ''))

        R['UPPER'] = (_upper, V); R['LOWER'] = (_lower, V)
        R['LENGTH'] = (_length, I); R['TRIM'] = (_trim, V)
        R['LTRIM'] = (_ltrim, V); R['RTRIM'] = (_rtrim, V)
        R['REVERSE'] = (_reverse, V); R['INITCAP'] = (_initcap, V)
        R['REPLACE'] = (_replace, V); R['POSITION'] = (_position, I)
        R['LEFT'] = (_left, V); R['RIGHT'] = (_right, V)
        R['REPEAT'] = (_repeat, V)
        R['STARTS_WITH'] = (_starts_with, B)
        R['ENDS_WITH'] = (_ends_with, B)
        R['CONTAINS'] = (_contains, B)
        R['ASCII'] = (_ascii, I); R['CHR'] = (_chr, V)
        R['SPLIT'] = (_split, V)
        R['REGEXP_REPLACE'] = (_regexp_replace, V)
        R['REGEXP_EXTRACT'] = (_regexp_extract, V)

        # ═══ 数学函数 ═══
        def _ceil(a, n, ev): return ev._num1(a[0], n, _math.ceil, BI)
        def _floor(a, n, ev): return ev._num1(a[0], n, _math.floor, BI)
        def _sqrt(a, n, ev): return ev._num1(a[0], n, _math.sqrt, D)
        def _cbrt(a, n, ev): return ev._num1(a[0], n, lambda x: x ** (1/3) if x >= 0 else -((-x) ** (1/3)), D)
        def _sign(a, n, ev): return ev._num1(a[0], n, lambda x: (1 if x > 0 else -1 if x < 0 else 0), I)
        def _ln(a, n, ev): return ev._num1(a[0], n, _math.log, D)
        def _log2(a, n, ev): return ev._num1(a[0], n, _math.log2, D)
        def _log10(a, n, ev): return ev._num1(a[0], n, _math.log10, D)
        def _exp(a, n, ev): return ev._num1(a[0], n, _math.exp, D)
        def _trunc(a, n, ev): return ev._num1(a[0], n, lambda x: int(x), BI)
        def _power(a, n, ev): return ev._num2(a[0], a[1], n, _math.pow, D)
        def _log(a, n, ev): return ev._num2(a[0], a[1], n, lambda b, x: _math.log(x, b), D)
        def _mod(a, n, ev): return ev._num2(a[0], a[1], n, lambda x, y: x % y if y != 0 else None, a[0].dtype)
        def _random_fn(a, n, ev): return ev._list_to_vec([_random.random() for _ in range(n)], D, n)
        def _bit_count(a, n, ev): return ev._num1(a[0], n, lambda v: bin(int(v)).count('1'), I)

        R['CEIL'] = (_ceil, BI); R['CEILING'] = (_ceil, BI)
        R['FLOOR'] = (_floor, BI); R['SQRT'] = (_sqrt, D)
        R['CBRT'] = (_cbrt, D); R['SIGN'] = (_sign, I)
        R['LN'] = (_ln, D); R['LOG2'] = (_log2, D)
        R['LOG10'] = (_log10, D); R['EXP'] = (_exp, D)
        R['TRUNC'] = (_trunc, BI); R['TRUNCATE'] = (_trunc, BI)
        R['POWER'] = (_power, D); R['LOG'] = (_log, D)
        R['MOD'] = (_mod, I); R['RANDOM'] = (_random_fn, D)
        R['BIT_COUNT'] = (_bit_count, I)

        # ═══ 条件函数 ═══
        def _typeof(a, n, ev):
            from executor.core.vector import DataVector
            from metal.bitmap import Bitmap
            rd = ['NULL' if a[0].is_null(i) else a[0].dtype.name for i in range(n)]
            return DataVector(dtype=V, data=rd, nulls=Bitmap(n), _length=n)
        def _encode(a, n, ev): return ev._str2(a[0], a[1], n, lambda v, enc: str(str(v).encode(str(enc))))
        def _decode(a, n, ev): return ev._str2(a[0], a[1], n, lambda v, enc: str(v))

        R['TYPEOF'] = (_typeof, V)
        R['ENCODE'] = (_encode, V); R['DECODE'] = (_decode, V)

        # ═══ 日期函数 ═══
        def _epoch(a, n, ev): return ev._num1(a[0], n, lambda x: x, BI)
        def _date_add(a, n, ev): return ev._num2(a[0], a[1], n, lambda d, delta: d + int(delta), I)
        def _date_sub(a, n, ev): return ev._num2(a[0], a[1], n, lambda d, delta: d - int(delta), I)
        def _date_diff(a, n, ev): return ev._num2(a[0], a[1], n, lambda x, y: int(x) - int(y), I)

        R['EPOCH'] = (_epoch, BI)
        R['DATE_ADD'] = (_date_add, I); R['DATE_SUB'] = (_date_sub, I)
        R['DATE_DIFF'] = (_date_diff, I)

        # ═══ 数组函数 ═══
        def _array_length(a, n, ev):
            return ev._num1(a[0], n, lambda v: len(ev._parse_array(v)), I)
        def _array_contains(a, n, ev):
            return ev._bool2(a[0], a[1], n, lambda arr, val: val in ev._parse_array(arr))
        def _array_position(a, n, ev):
            return ev._num2(a[0], a[1], n, lambda arr, val: (ev._parse_array(arr).index(val) + 1 if val in ev._parse_array(arr) else 0), I)
        def _array_join(a, n, ev):
            return ev._str2(a[0], a[1], n, lambda arr, sep: str(sep).join(str(x) for x in ev._parse_array(arr)))

        R['ARRAY_LENGTH'] = (_array_length, I)
        R['ARRAY_CONTAINS'] = (_array_contains, B)
        R['ARRAY_POSITION'] = (_array_position, I)
        R['ARRAY_JOIN'] = (_array_join, V)

        # ═══ 相似度（从 batch 18 迁入）═══
        try:
            from executor.similarity.simhash import text_similarity as _sim
            def _simhash(a, n, ev):
                from executor.core.vector import DataVector
                from metal.bitmap import Bitmap
                from metal.typed_vector import TypedVector
                from storage.types import DTYPE_TO_ARRAY_CODE
                code = DTYPE_TO_ARRAY_CODE.get(D)
                rd = TypedVector(code) if code else []; rn = Bitmap(n)
                for i in range(n):
                    if a[0].is_null(i) or a[1].is_null(i): rn.set_bit(i); rd.append(0.0)
                    else: rd.append(_sim(str(a[0].get(i)), str(a[1].get(i))))
                return DataVector(dtype=D, data=rd, nulls=rn, _length=n)
            R['SIMHASH_SIMILARITY'] = (_simhash, D)
        except ImportError: pass

        try:
            from executor.similarity.minhash_lsh import MinHash
            def _minhash(a, n, ev):
                from executor.core.vector import DataVector
                from metal.bitmap import Bitmap
                from metal.typed_vector import TypedVector
                from storage.types import DTYPE_TO_ARRAY_CODE
                mh = MinHash(num_hashes=64); code = DTYPE_TO_ARRAY_CODE.get(D)
                rd = TypedVector(code) if code else []; rn = Bitmap(n)
                for i in range(n):
                    if a[0].is_null(i) or a[1].is_null(i): rn.set_bit(i); rd.append(0.0)
                    else:
                        sa = set(str(a[0].get(i)).split()); sb = set(str(a[1].get(i)).split())
                        rd.append(mh.jaccard_estimate(mh.signature(sa), mh.signature(sb)))
                return DataVector(dtype=D, data=rd, nulls=rn, _length=n)
            R['MINHASH_JACCARD'] = (_minhash, D)
        except ImportError: pass
