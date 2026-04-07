from __future__ import annotations
"""函数 dispatch dict — O(1) 查找。
[RP2] 每个 handler 包装参数数量校验。
[RP3] ARRAY_POSITION 等避免重复 _parse_array。"""
import math as _math, random as _random, re as _re
import datetime as _dt, time as _time
from typing import Any, Callable, Dict
from metal.bitmap import Bitmap
from metal.hash import z1hash64
from storage.types import DataType, DTYPE_TO_ARRAY_CODE
from executor.core.vector import DataVector
from utils.errors import ExecutionError


def _safe(min_args: int, handler: Callable) -> Callable:
    """[RP2] 参数数量校验包装器。"""
    def wrapped(s, a, n, e):
        if len(a) < min_args:
            fname = e.name if hasattr(e, 'name') else '?'
            raise ExecutionError(
                f"函数 {fname} 至少需要 {min_args} 个参数，"
                f"实际 {len(a)} 个")
        return handler(s, a, n, e)
    return wrapped


def build_dispatch() -> Dict[str, Any]:
    D: Dict[str, Any] = {}

    # 字符串（1 参数）
    D['UPPER'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,str.upper))
    D['LOWER'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,str.lower))
    D['LENGTH'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda x:len(str(x)),DataType.INT))
    D['TRIM'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,str.strip))
    D['LTRIM'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,str.lstrip))
    D['RTRIM'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,str.rstrip))
    D['REVERSE'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,lambda x:x[::-1]))
    D['INITCAP'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,str.title))

    # 字符串（多参数）
    D['REPLACE'] = _safe(3, lambda s,a,n,e: s._str3(a[0],a[1],a[2],n,lambda x,y,z:x.replace(y,z)))
    D['SUBSTR'] = D['SUBSTRING'] = _safe(1, lambda s,a,n,e: s._substr(a,n))
    D['CONCAT'] = _safe(1, lambda s,a,n,e: s._concat_fn('CONCAT',a,n))
    D['CONCAT_WS'] = _safe(2, lambda s,a,n,e: s._concat_fn('CONCAT_WS',a,n))
    D['POSITION'] = _safe(2, lambda s,a,n,e: s._num2(a[0],a[1],n,lambda sub,x:str(x).find(str(sub))+1,DataType.INT))
    D['LEFT'] = _safe(2, lambda s,a,n,e: s._str_int(a[0],a[1],n,lambda x,k:x[:k]))
    D['RIGHT'] = _safe(2, lambda s,a,n,e: s._str_int(a[0],a[1],n,lambda x,k:x[-k:] if k>0 else ''))
    D['REPEAT'] = _safe(2, lambda s,a,n,e: s._str_int(a[0],a[1],n,lambda x,k:x*k))
    D['STARTS_WITH'] = _safe(2, lambda s,a,n,e: s._bool2(a[0],a[1],n,lambda x,p:str(x).startswith(str(p))))
    D['ENDS_WITH'] = _safe(2, lambda s,a,n,e: s._bool2(a[0],a[1],n,lambda x,p:str(x).endswith(str(p))))
    D['CONTAINS'] = _safe(2, lambda s,a,n,e: s._bool2(a[0],a[1],n,lambda x,p:str(p) in str(x)))
    D['LPAD'] = _safe(2, lambda s,a,n,e: s._lpad_rpad(a,n,True))
    D['RPAD'] = _safe(2, lambda s,a,n,e: s._lpad_rpad(a,n,False))
    D['ASCII'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda x:ord(str(x)[0]) if str(x) else 0,DataType.INT))
    D['CHR'] = _safe(1, lambda s,a,n,e: s._str1(a[0],n,lambda x:chr(int(float(x))) if x is not None else ''))
    D['SPLIT_PART'] = _safe(3, lambda s,a,n,e: s._split_part(a,n))
    D['SPLIT'] = _safe(2, lambda s,a,n,e: s._str2(a[0],a[1],n,lambda x,d:str(str(x).split(str(d)))))
    D['REGEXP_REPLACE'] = _safe(3, lambda s,a,n,e: s._str3(a[0],a[1],a[2],n,lambda x,p,r:_re.sub(p,r,x)))
    D['REGEXP_MATCH'] = _safe(2, lambda s,a,n,e: s._bool2(a[0],a[1],n,lambda x,p:bool(_re.search(str(p),str(x)))))
    D['REGEXP_EXTRACT'] = _safe(2, lambda s,a,n,e: s._str2(a[0],a[1],n,lambda x,p:(m.group(0) if (m:=_re.search(str(p),str(x))) else '')))

    # 数学
    D['ABS'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,abs,a[0].dtype))
    D['CEIL'] = D['CEILING'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,_math.ceil,DataType.BIGINT))
    D['FLOOR'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,_math.floor,DataType.BIGINT))
    D['ROUND'] = _safe(1, lambda s,a,n,e: (s._num2(a[0],a[1],n,lambda x,d:round(x,int(d)),DataType.DOUBLE) if len(a)>=2 else s._num1(a[0],n,round,DataType.BIGINT)))
    D['TRUNC'] = D['TRUNCATE'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda x:int(x),DataType.BIGINT))
    D['POWER'] = _safe(2, lambda s,a,n,e: s._num2(a[0],a[1],n,_math.pow,DataType.DOUBLE))
    D['SQRT'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,_math.sqrt,DataType.DOUBLE))
    D['CBRT'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)),DataType.DOUBLE))
    D['MOD'] = _safe(2, lambda s,a,n,e: s._num2(a[0],a[1],n,lambda x,y:x%y if y!=0 else None,a[0].dtype))
    D['SIGN'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda x:(1 if x>0 else -1 if x<0 else 0),DataType.INT))
    D['LN'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,_math.log,DataType.DOUBLE))
    D['LOG2'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,_math.log2,DataType.DOUBLE))
    D['LOG10'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,_math.log10,DataType.DOUBLE))
    D['LOG'] = _safe(2, lambda s,a,n,e: s._num2(a[0],a[1],n,lambda b,x:_math.log(x,b),DataType.DOUBLE))
    D['EXP'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,_math.exp,DataType.DOUBLE))
    D['GREATEST'] = _safe(2, lambda s,a,n,e: s._variadic(a,n,lambda vs:max((v for v in vs if v is not None),default=None),a[0].dtype))
    D['LEAST'] = _safe(2, lambda s,a,n,e: s._variadic(a,n,lambda vs:min((v for v in vs if v is not None),default=None),a[0].dtype))
    D['RANDOM'] = lambda s,a,n,e: s._list_to_vec([_random.random() for _ in range(n)],DataType.DOUBLE,n)
    D['WIDTH_BUCKET'] = _safe(4, lambda s,a,n,e: s._width_bucket(a,n))
    D['BIT_COUNT'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda v:bin(int(v)).count('1'),DataType.INT))

    # 条件
    D['NULLIF'] = _safe(2, lambda s,a,n,e: s._nullif(a[0],a[1],n))
    D['TYPEOF'] = _safe(1, lambda s,a,n,e: DataVector(dtype=DataType.VARCHAR,data=['NULL' if a[0].is_null(i) else a[0].dtype.name for i in range(n)],nulls=Bitmap(n),_length=n))
    D['TRY_CAST'] = _safe(2, lambda s,a,n,e: s._try_cast_from_args(a,n,e))
    D['HASH'] = D['MURMUR_HASH'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda v:z1hash64(str(v).encode('utf-8')),DataType.BIGINT))
    D['ENCODE'] = _safe(2, lambda s,a,n,e: s._str2(a[0],a[1],n,lambda v,enc:str(str(v).encode(str(enc)))))
    D['DECODE'] = _safe(2, lambda s,a,n,e: s._str2(a[0],a[1],n,lambda v,enc:str(v)))

    # 日期
    D['NOW'] = D['CURRENT_TIMESTAMP'] = lambda s,a,n,e: s._list_to_vec([int(_time.time()*1_000_000)]*n,DataType.TIMESTAMP,n)
    D['CURRENT_DATE'] = lambda s,a,n,e: s._list_to_vec([(_dt.date.today()-_dt.date(1970,1,1)).days]*n,DataType.DATE,n)
    for _p in ('YEAR','MONTH','DAY','HOUR','MINUTE','SECOND','DAY_OF_WEEK','DAY_OF_YEAR','WEEK_OF_YEAR','QUARTER'):
        D[_p] = _safe(1, lambda s,a,n,e,__p=_p: s._date_part(__p,a[0],n))
    D['EXTRACT'] = _safe(2, lambda s,a,n,e: s._date_part(str(a[0].get(0)).upper() if not a[0].is_null(0) else '',a[1],n))
    D['EPOCH'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda x:x,DataType.BIGINT))
    D['DATE_ADD'] = _safe(2, lambda s,a,n,e: s._num2(a[0],a[1],n,lambda d,delta:d+int(delta),DataType.INT))
    D['DATE_SUB'] = _safe(2, lambda s,a,n,e: s._num2(a[0],a[1],n,lambda d,delta:d-int(delta),DataType.INT))
    D['DATE_DIFF'] = _safe(2, lambda s,a,n,e: s._num2(a[0],a[1],n,lambda x,y:int(x)-int(y),DataType.INT))
    D['DATE_TRUNC'] = _safe(2, lambda s,a,n,e: s._date_trunc(a[0],a[1],n))
    D['DATE_FORMAT'] = _safe(2, lambda s,a,n,e: s._date_format(a[0],a[1],n))
    D['TO_DATE'] = _safe(1, lambda s,a,n,e: s._to_date(a[0],n))
    D['TO_TIMESTAMP'] = _safe(1, lambda s,a,n,e: s._to_timestamp(a[0],n))

    # 数组 — [RP3] ARRAY_POSITION 预解析避免双重 _parse_array
    D['ARRAY'] = D['ARRAY_CREATE'] = _safe(1, lambda s,a,n,e: s._eval_array_create(a,n))
    D['ARRAY_LENGTH'] = _safe(1, lambda s,a,n,e: s._num1(a[0],n,lambda v:len(s._parse_array(v)),DataType.INT))
    D['ARRAY_CONTAINS'] = _safe(2, lambda s,a,n,e: s._bool2(a[0],a[1],n,lambda arr,val:val in s._parse_array(arr)))

    # [RP3] ARRAY_POSITION: 预解析一次，避免 _parse_array 被调用两次
    def _array_position(s, a, n, e):
        from metal.typed_vector import TypedVector
        code = DTYPE_TO_ARRAY_CODE.get(DataType.INT)
        rd = TypedVector(code) if code else []
        rn = Bitmap(n)
        for i in range(n):
            if a[0].is_null(i) or a[1].is_null(i):
                rn.set_bit(i)
                rd.append(0 if isinstance(rd, TypedVector) else None)
            else:
                parsed = s._parse_array(a[0].get(i))  # 只解析一次
                val = a[1].get(i)
                try:
                    idx = parsed.index(val) + 1
                except ValueError:
                    idx = 0
                rd.append(idx)
        return DataVector(dtype=DataType.INT, data=rd, nulls=rn, _length=n)
    D['ARRAY_POSITION'] = _safe(2, _array_position)

    D['ARRAY_SLICE'] = _safe(3, lambda s,a,n,e: s._eval_array_slice(a,n))
    D['ARRAY_APPEND'] = _safe(2, lambda s,a,n,e: s._eval_array_binary(a[0],a[1],n,lambda arr,val:arr+[val]))
    D['ARRAY_PREPEND'] = _safe(2, lambda s,a,n,e: s._eval_array_binary(a[0],a[1],n,lambda arr,val:[val]+arr))
    D['ARRAY_REMOVE'] = _safe(2, lambda s,a,n,e: s._eval_array_binary(a[0],a[1],n,lambda arr,val:[x for x in arr if x!=val]))
    D['ARRAY_CONCAT'] = _safe(2, lambda s,a,n,e: s._eval_array_binary2(a[0],a[1],n,lambda x,y:x+y))
    D['ARRAY_SORT'] = _safe(1, lambda s,a,n,e: s._eval_array_unary(a[0],n,sorted))
    D['ARRAY_REVERSE'] = _safe(1, lambda s,a,n,e: s._eval_array_unary(a[0],n,lambda x:list(reversed(x))))
    D['ARRAY_DISTINCT'] = _safe(1, lambda s,a,n,e: s._eval_array_unary(a[0],n,lambda x:list(dict.fromkeys(x))))
    D['ARRAY_FLATTEN'] = _safe(1, lambda s,a,n,e: s._eval_array_unary(a[0],n,s._flatten))
    D['ARRAY_INTERSECT'] = _safe(2, lambda s,a,n,e: s._eval_array_binary2(a[0],a[1],n,lambda x,y:[v for v in x if v in set(y)]))
    D['ARRAY_UNION'] = _safe(2, lambda s,a,n,e: s._eval_array_binary2(a[0],a[1],n,lambda x,y:list(dict.fromkeys(x+y))))
    D['ARRAY_EXCEPT'] = _safe(2, lambda s,a,n,e: s._eval_array_binary2(a[0],a[1],n,lambda x,y:[v for v in x if v not in set(y)]))
    D['ARRAY_JOIN'] = _safe(2, lambda s,a,n,e: s._str2(a[0],a[1],n,lambda arr,sep:str(sep).join(str(x) for x in s._parse_array(arr))))
    D['GENERATE_SERIES'] = _safe(2, lambda s,a,n,e: s._eval_generate_series(a,n))
    D['EXPLODE'] = D['UNNEST'] = _safe(1, lambda s,a,n,e: s._eval_array_unary(a[0],n,lambda x:x))
    D['JACCARD_SIMILARITY'] = _safe(2, lambda s,a,n,e: s._eval_jaccard(a[0],a[1],n))
    D['COSINE_SIMILARITY'] = _safe(2, lambda s,a,n,e: s._eval_cosine(a[0],a[1],n))
    return D
