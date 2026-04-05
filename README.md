# Z1DB — 纯 Python 零依赖 OLAP 列式数据库引擎

---

## 一、项目概述

**Z1DB** 是一个完全用纯 Python 实现的列式 OLAP 数据库引擎。零外部依赖，仅使用 Python 3.8+ 标准库。

## 二、SQL 支持

```sql
-- DDL
CREATE TABLE / DROP TABLE / ALTER TABLE (ADD/DROP/RENAME COLUMN)

-- DML
INSERT / UPDATE / DELETE

-- 查询
SELECT / WHERE / ORDER BY / LIMIT / OFFSET / DISTINCT
GROUP BY / HAVING
JOIN (INNER / LEFT / RIGHT / FULL OUTER / CROSS)
UNION / INTERSECT / EXCEPT [ALL]
WITH ... AS (CTE 公用表表达式)
子查询 (IN subquery / EXISTS / 标量子查询)
CASE WHEN / CAST / BETWEEN / LIKE / IN
EXPLAIN
窗口函数 (ROW_NUMBER, RANK, LAG, LEAD, SUM OVER ...)
```

---

## 三、内置函数

| 类别   | 数量 | 代表                                                         |
| ------ | ---- | ------------------------------------------------------------ |
| 数学   | 19   | ABS, CEIL, FLOOR, ROUND, POWER, SQRT, CBRT, LN, EXP, RANDOM  |
| 字符串 | 28   | UPPER, LOWER, LENGTH, SUBSTR, CONCAT, REPLACE, REGEXP_MATCH  |
| 日期   | 22   | CURRENT_DATE, YEAR, MONTH, DATE_ADD, TO_DATE, TO_TIMESTAMP   |
| 条件   | 12   | CAST, COALESCE, NULLIF, IF, TYPEOF, HASH, MURMUR_HASH        |
| 聚合   | 22   | COUNT, SUM, AVG, STDDEV, MEDIAN, MODE, APPROX_COUNT_DISTINCT |
| 窗口   | 16   | ROW_NUMBER, RANK, LAG, LEAD, FIRST_VALUE, SUM/AVG OVER       |
| 数组   | 17   | ARRAY_LENGTH, ARRAY_CONTAINS, ARRAY_SORT, ARRAY_JOIN         |
| 表函数 | 4    | EXPLODE, UNNEST, GENERATE_SERIES                             |
| 相似度 | 2    | JACCARD_SIMILARITY, COSINE_SIMILARITY                        |

---

## 四、事务支持

```
模式:
  自动提交: 每条语句 = 隐式事务 (默认)
  显式事务: BEGIN → 多条语句 → COMMIT / ROLLBACK

锁机制:
  读-读: 并行
  读-写: 读不阻塞 (COW 快照读)
  写-写: 串行等待

快照回滚:
  BEGIN → snapshot_table() 保存行数据快照
  ROLLBACK → 从快照恢复全部修改
  COMMIT → 释放快照, 变更永久化
```

---

## 五、网络服务

```
启动方式:
  python main.py                    # REPL 模式
  python main.py --server 5433      # TCP 服务器模式
  python main.py /path/to/data      # 持久化 REPL

Wire 协议:
  Client → Server: SQL 文本 + \n
  Server → Client: JSON 结果 + \n
  {"status":"ok", "columns":[...], "rows":[...], "timing":0.001}
  {"status":"error", "message":"table not found"}

Python 客户端:
  from server.protocol import Z1Client
  with Z1Client('127.0.0.1', 5433) as c:
      result = c.execute("SELECT * FROM users;")
      print(result['rows'])
```

---

## 六、REPL 增强功能

```
z1db> .color on           # ANSI 彩色输出 (关键字蓝/数字青/字符串绿)
z1db> .profile on         # 逐阶段耗时 (parse/resolve/optimize/execute)
z1db> .estimate on        # 查询前行数预估
z1db> .timer on|off       # 查询计时开关
z1db> .benchmark 100 SQL  # 跑 100 次, 显示 Avg/P50/P95/P99/QPS
z1db> .memory             # 各表内存估算
z1db> .history 20         # 最近 20 条查询记录
z1db> .analyze <table>    # 计算列统计信息
z1db> .stats <table>      # 显示 NDV/min/max
z1db> .import file table  # CSV 导入
z1db> .export file table  # CSV 导出
```

---

## 七、快速开始

```bash
# 内存模式
python main.py

# 持久化模式
python main.py /path/to/data

# TCP 服务器
python main.py --server 5433
```

---

## 八、已知限制

```
1.  无 DECIMAL (用 float64)
2.  无 COLLATE (非 ASCII 按字节序)
3.  无 MVCC/行锁 (表级 COW 快照)
4.  无 WAL 崩溃恢复
5.  无 GRANT/REVOKE 权限
6.  UPDATE/DELETE 全表重建
7.  不支持相关子查询
8.  INT 溢出抛错 (int32 ±2³¹)
9.  FLOAT/0 → inf; INT/0 → ERROR
10. FLOAT 与 DOUBLE 内部均为 float64
```

---

