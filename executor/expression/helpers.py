from __future__ import annotations
"""evaluator 辅助方法 — 字符串/数值/日期/数组的通用模板。
这些是 ExpressionEvaluator 的 mixin 方法，通过多继承混入。"""
import datetime as _dt
import json as _json
import math as _math
from typing import Any, List
from executor.core.vector import DataVector
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from storage.types import (
    DTYPE_TO_ARRAY_CODE, DataType, is_numeric, resolve_type_name)
from utils.errors import ExecutionError
from executor.expression.vec_ops import (
    map_unary_numeric, map_binary_numeric,
    map_unary_string, map_binary_string, map_ternary_string,
    map_str_int, map_binary_bool, values_to_vector)
try:
    from executor.string_algo.compiled_date import ISO_DATE_PARSER, parse_date_auto as _parse_date_auto
    _HAS_COMPILED_DATE = True
except ImportError:
    _HAS_COMPILED_DATE = False; _parse_date_auto = None; ISO_DATE_PARSER = None

try:
    from executor.similarity.minhash_lsh import jaccard_exact as _jaccard_exact, cosine_similarity as _cosine_similarity
    _HAS_SIMILARITY = True
except ImportError:
    _HAS_SIMILARITY = False


class EvalHelpers:
    """evaluator 辅助方法 mixin。"""

    # ═══ 字符串 ═══

    def _str1(self, v, n, fn):
        return map_unary_string(v, n, fn)

    def _str2(self, a, b, n, fn):
        return map_binary_string(a, b, n, fn)

    def _str3(self, a, b, c, n, fn):
        return map_ternary_string(a, b, c, n, fn)

    def _str_int(self, sv, iv, n, fn):
        return map_str_int(sv, iv, n, fn)

    def _substr(self, args, n):
        sv=args[0]; rd=[]; rn=Bitmap(n)
        for i in range(n):
            if sv.is_null(i): rn.set_bit(i); rd.append(''); continue
            s=str(sv.get(i)); start=(int(args[1].get(i))-1 if len(args)>1 and not args[1].is_null(i) else 0)
            if start<0: start=0
            if len(args)>2 and not args[2].is_null(i): rd.append(s[start:start+int(args[2].get(i))])
            else: rd.append(s[start:])
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _concat_fn(self, name, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if name=='CONCAT_WS':
                if args[0].is_null(i): rn.set_bit(i); rd.append(''); continue
                sep=str(args[0].get(i)); parts=[str(a.get(i)) for a in args[1:] if not a.is_null(i)]
                rd.append(sep.join(parts))
            else: rd.append(''.join(str(a.get(i)) for a in args if not a.is_null(i)))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _lpad_rpad(self, args, n, left):
        sv,lv=args[0],args[1]; rd=[]; rn=Bitmap(n)
        for i in range(n):
            if sv.is_null(i) or lv.is_null(i): rn.set_bit(i); rd.append('')
            else:
                s=str(sv.get(i)); w=int(lv.get(i)); pc=' '
                if len(args)>=3 and not args[2].is_null(i):
                    fill=str(args[2].get(i))
                    if fill: pc=fill[0]
                rd.append(s.rjust(w,pc) if left else s.ljust(w,pc))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _split_part(self, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if any(args[j].is_null(i) for j in range(3)): rn.set_bit(i); rd.append('')
            else:
                parts=str(args[0].get(i)).split(str(args[1].get(i))); idx=int(args[2].get(i))
                rd.append(parts[idx-1] if 1<=idx<=len(parts) else '')
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    # ═══ 数值 ═══

    def _num1(self, v, n, fn, dt):
        return map_unary_numeric(v, n, fn, dt)

    def _num2(self, a, b, n, fn, dt):
        return map_binary_numeric(a, b, n, fn, dt)

    def _bool2(self, a, b, n, fn):
        return map_binary_bool(a, b, n, fn)

    def _variadic(self, args, n, fn, dt):
        results=[fn([a.get(i) for a in args]) for i in range(n)]
        return self._list_to_vec(results, dt, n)

    def _nullif(self, a, b, n):
        results=[None]*n
        for i in range(n):
            if a.is_null(i): continue
            if b.is_null(i): results[i]=a.get(i); continue
            results[i]=None if a.get(i)==b.get(i) else a.get(i)
        return self._list_to_vec(results, a.dtype, n)

    def _width_bucket(self, args, n):
        results=[None]*n
        for i in range(n):
            if any(args[j].is_null(i) for j in range(4)): continue
            val=float(args[0].get(i)); lo=float(args[1].get(i))
            hi=float(args[2].get(i)); buckets=int(args[3].get(i))
            if hi==lo or buckets<=0: results[i]=0
            elif val<lo: results[i]=0
            elif val>=hi: results[i]=buckets+1
            else: results[i]=int((val-lo)/(hi-lo)*buckets)+1
        return self._list_to_vec(results, DataType.INT, n)

    # ═══ 日期 ═══

    def _date_part(self, part, vec, n):
        results=[None]*n
        for i in range(n):
            if vec.is_null(i): continue
            v=vec.get(i)
            try:
                if isinstance(v,int) and abs(v)<1_000_000:
                    d=_dt.date(1970,1,1)+_dt.timedelta(days=v)
                    results[i]=self._extract_part(part,d,None)
                elif isinstance(v,int):
                    dt_obj=_dt.datetime(1970,1,1)+_dt.timedelta(microseconds=v)
                    results[i]=self._extract_part(part,dt_obj.date(),dt_obj.time())
            except Exception: pass
        return self._list_to_vec(results, DataType.INT, n)

    @staticmethod
    def _extract_part(part, d, t):
        if part=='YEAR': return d.year
        if part=='MONTH': return d.month
        if part=='DAY': return d.day
        if part=='HOUR': return t.hour if t else 0
        if part=='MINUTE': return t.minute if t else 0
        if part=='SECOND': return t.second if t else 0
        if part=='DAY_OF_WEEK': return d.isoweekday()
        if part=='DAY_OF_YEAR': return d.timetuple().tm_yday
        if part=='WEEK_OF_YEAR': return d.isocalendar()[1]
        if part=='QUARTER': return (d.month-1)//3+1
        return None

    def _date_trunc(self, uv, vv, n):
        results=[None]*n
        for i in range(n):
            if uv.is_null(i) or vv.is_null(i): continue
            unit=str(uv.get(i)).upper(); v=int(vv.get(i))
            try:
                if abs(v)<1_000_000:
                    d=_dt.date(1970,1,1)+_dt.timedelta(days=v)
                    if unit=='YEAR': d=d.replace(month=1,day=1)
                    elif unit=='MONTH': d=d.replace(day=1)
                    elif unit=='QUARTER': d=d.replace(month=((d.month-1)//3)*3+1,day=1)
                    elif unit=='WEEK': d=d-_dt.timedelta(days=d.weekday())
                    results[i]=(d-_dt.date(1970,1,1)).days
                else: results[i]=v
            except Exception: pass
        return self._list_to_vec(results, DataType.INT, n)

    def _date_format(self, vv, fv, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if vv.is_null(i) or fv.is_null(i): rn.set_bit(i); rd.append(''); continue
            v=int(vv.get(i)); fmt=str(fv.get(i))
            try:
                d=(_dt.date(1970,1,1)+_dt.timedelta(days=v)) if abs(v)<1_000_000 else (_dt.datetime(1970,1,1)+_dt.timedelta(microseconds=v))
                py_fmt=fmt.replace('YYYY','%Y').replace('MM','%m').replace('DD','%d').replace('HH','%H').replace('MI','%M').replace('SS','%S')
                rd.append(d.strftime(py_fmt))
            except Exception: rn.set_bit(i); rd.append('')
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _to_date(self, vv, n):
        results=[None]*n
        for i in range(n):
            if vv.is_null(i): continue
            s=str(vv.get(i))
            if _HAS_COMPILED_DATE and _parse_date_auto: results[i]=_parse_date_auto(s)
            else:
                for fmt in ('%Y-%m-%d','%m/%d/%Y','%d.%m.%Y'):
                    try: d=_dt.datetime.strptime(s,fmt).date(); results[i]=(d-_dt.date(1970,1,1)).days; break
                    except ValueError: continue
        return self._list_to_vec(results, DataType.DATE, n)

    def _to_timestamp(self, vv, n):
        results=[None]*n
        for i in range(n):
            if vv.is_null(i): continue
            s=str(vv.get(i))
            if _HAS_COMPILED_DATE and ISO_DATE_PARSER: results[i]=ISO_DATE_PARSER.parse_timestamp(s)
            else:
                for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%d'):
                    try: dt_obj=_dt.datetime.strptime(s,fmt); results[i]=int((dt_obj-_dt.datetime(1970,1,1)).total_seconds()*1_000_000); break
                    except ValueError: continue
        return self._list_to_vec(results, DataType.TIMESTAMP, n)

    def _try_cast_from_args(self, args, n, expr):
        from parser.ast import Literal
        if len(args)<2: return self._list_to_vec([None]*n, DataType.VARCHAR, n)
        src=args[0]; type_str='VARCHAR'
        if hasattr(expr,'args') and len(expr.args)>=2 and isinstance(expr.args[1],Literal): type_str=str(expr.args[1].value)
        elif not args[1].is_null(0): type_str=str(args[1].get(0))
        target_dt,_=resolve_type_name(type_str,[])
        results=[None]*n
        for i in range(n):
            if src.is_null(i): continue
            v=src.get(i)
            try:
                if target_dt==DataType.INT: results[i]=int(v)
                elif target_dt==DataType.BIGINT: results[i]=int(v)
                elif target_dt in (DataType.FLOAT,DataType.DOUBLE): results[i]=float(v)
                elif target_dt in (DataType.VARCHAR,DataType.TEXT): results[i]=str(v)
                elif target_dt==DataType.BOOLEAN: results[i]=_cast_bool(v)
                else: results[i]=v
            except (ValueError,TypeError): results[i]=None
        return self._list_to_vec(results, target_dt, n)

    # ═══ 数组 ═══

    @staticmethod
    def _parse_array(val):
        if isinstance(val,list): return val
        if isinstance(val,str):
            s=val.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed=_json.loads(s)
                    if isinstance(parsed,list): return parsed
                except: pass
                s=s[1:-1].strip()
                if not s: return []
                parts=[]; depth=0; cur=[]
                for ch in s:
                    if ch=='[': depth+=1; cur.append(ch)
                    elif ch==']': depth-=1; cur.append(ch)
                    elif ch==',' and depth==0: parts.append(''.join(cur).strip()); cur=[]
                    else: cur.append(ch)
                if cur: parts.append(''.join(cur).strip())
                result=[]
                for p in parts:
                    p=p.strip().strip("'\"")
                    try: result.append(int(p))
                    except:
                        try: result.append(float(p))
                        except: result.append(p)
                return result
        return [val] if val is not None else []

    def _eval_array_create(self, args, n):
        return DataVector(dtype=DataType.VARCHAR, data=[str([a.get(i) for a in args]) for i in range(n)], nulls=Bitmap(n), _length=n)

    def _eval_array_unary(self, av, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if av.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(str(fn(self._parse_array(av.get(i)))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_array_binary(self, av, vv, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if av.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(str(fn(self._parse_array(av.get(i)), vv.get(i) if not vv.is_null(i) else None)))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_array_binary2(self, av, bv, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if av.is_null(i) or bv.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(str(fn(self._parse_array(av.get(i)), self._parse_array(bv.get(i)))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_array_slice(self, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if args[0].is_null(i): rn.set_bit(i); rd.append('')
            else:
                arr=self._parse_array(args[0].get(i))
                start=int(args[1].get(i))-1 if not args[1].is_null(i) else 0
                end=int(args[2].get(i)) if not args[2].is_null(i) else len(arr)
                rd.append(str(arr[start:end]))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    @staticmethod
    def _flatten(arr):
        result=[]
        for item in arr:
            if isinstance(item,list): result.extend(item)
            else: result.append(item)
        return result

    def _eval_generate_series(self, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if any(args[j].is_null(i) for j in range(min(len(args),2))): rn.set_bit(i); rd.append(''); continue
            start=int(args[0].get(i)); stop=int(args[1].get(i))
            step=int(args[2].get(i)) if len(args)>=3 and not args[2].is_null(i) else 1
            if step==0: rn.set_bit(i); rd.append('')
            else: rd.append(str(list(range(start,stop+(1 if step>0 else -1),step))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    # ═══ 相似度 ═══

    def _eval_jaccard(self, av, bv, n):
        code=DTYPE_TO_ARRAY_CODE.get(DataType.DOUBLE); rd=TypedVector(code) if code else []; rn=Bitmap(n)
        for i in range(n):
            if av.is_null(i) or bv.is_null(i): rn.set_bit(i); rd.append(0.0)
            else:
                sa=set(self._parse_array(av.get(i))); sb=set(self._parse_array(bv.get(i)))
                rd.append(_jaccard_exact(sa,sb) if _HAS_SIMILARITY else (len(sa&sb)/len(sa|sb) if (sa or sb) else 1.0))
        return DataVector(dtype=DataType.DOUBLE, data=rd, nulls=rn, _length=n)

    def _eval_cosine(self, av, bv, n):
        code=DTYPE_TO_ARRAY_CODE.get(DataType.DOUBLE); rd=TypedVector(code) if code else []; rn=Bitmap(n)
        for i in range(n):
            if av.is_null(i) or bv.is_null(i): rn.set_bit(i); rd.append(0.0)
            else:
                va=[float(x) for x in self._parse_array(av.get(i))]
                vb=[float(x) for x in self._parse_array(bv.get(i))]
                if _HAS_SIMILARITY: rd.append(_cosine_similarity(va,vb))
                else:
                    ml=min(len(va),len(vb))
                    if ml==0: rd.append(0.0)
                    else:
                        dot=sum(va[j]*vb[j] for j in range(ml))
                        ma=_math.sqrt(sum(x*x for x in va[:ml])); mb=_math.sqrt(sum(x*x for x in vb[:ml]))
                        rd.append(dot/(ma*mb) if ma>0 and mb>0 else 0.0)
        return DataVector(dtype=DataType.DOUBLE, data=rd, nulls=rn, _length=n)

    # ═══ 通用 ═══

    def _list_to_vec(self, values, dtype, n):
        return values_to_vector(values, dtype, n)

    def _to_bool(self, vec):
        if vec.dtype==DataType.BOOLEAN: return vec
        if is_numeric(vec.dtype):
            n=len(vec); d=Bitmap(n); nl=Bitmap(n)
            for i in range(n):
                if vec.is_null(i): nl.set_bit(i)
                elif vec.get(i)!=0: d.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=d, nulls=nl, _length=n)
        raise ExecutionError(f"无法转换 {vec.dtype.name} 为布尔")

    def _bool_to_bitmap(self, vec):
        n=len(vec); bm=Bitmap(n)
        for i in range(n):
            if not vec.is_null(i) and vec.data.get_bit(i): bm.set_bit(i)
        return bm


def _cast_bool(val):
    if isinstance(val,bool): return val
    if isinstance(val,(int,float)): return val!=0
    if isinstance(val,str):
        l=val.strip().lower()
        if l in ('true','1','t','yes','on'): return True
        if l in ('false','0','f','no','off'): return False
        raise ExecutionError(f"无法转换 '{val}' 为布尔")
    return bool(val)


# 导出给 evaluator 的 _cast_to_bool
_cast_to_bool = _cast_bool

