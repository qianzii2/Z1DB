# Z1DB

---

## 一、项目定位

Z1DB 是一个**纯 Python 零依赖**的列式 OLAP 数据库引擎。从位操作到查询优化器全部手写实现，覆盖数据库系统全栈。

---

## 二、功能清单

### 2.1 SQL 语句

| 类别   | 支持的语句                                                   |
| ------ | ------------------------------------------------------------ |
| DDL    | CREATE TABLE, DROP TABLE, ALTER TABLE (ADD/DROP/RENAME COLUMN) |
| DML    | INSERT, UPDATE, DELETE                                       |
| 查询   | SELECT, WHERE, ORDER BY, LIMIT, OFFSET, DISTINCT             |
| 聚合   | GROUP BY, HAVING, 22 种聚合函数                              |
| 连接   | INNER/LEFT/RIGHT/FULL/CROSS JOIN, 7 种算法                   |
| 集合   | UNION, INTERSECT, EXCEPT (ALL)                               |
| 窗口   | ROW_NUMBER, RANK, LAG, LEAD, SUM/AVG/MIN/MAX OVER, 帧定义    |
| CTE    | WITH ... AS, WITH RECURSIVE                                  |
| 子查询 | IN subquery, EXISTS, NOT IN, NOT EXISTS, 标量子查询, 相关子查询 |
| 表达式 | CASE/WHEN, CAST, BETWEEN, LIKE, IS NULL                      |
| 管理   | EXPLAIN, ALTER TABLE                                         |
| 事务   | BEGIN, COMMIT, ROLLBACK (表级锁)                             |

### 2.2 142 个内置函数

```
标量-数学(19):    ABS CEIL FLOOR ROUND TRUNC POWER SQRT CBRT MOD SIGN
                  LN LOG2 LOG10 LOG EXP GREATEST LEAST RANDOM WIDTH_BUCKET

标量-字符串(28):  UPPER LOWER LENGTH SUBSTR CONCAT CONCAT_WS
                  TRIM LTRIM RTRIM LPAD RPAD LEFT RIGHT
                  REPLACE REVERSE REPEAT POSITION
                  STARTS_WITH ENDS_WITH CONTAINS
                  SPLIT SPLIT_PART INITCAP ASCII CHR
                  REGEXP_MATCH REGEXP_REPLACE REGEXP_EXTRACT

标量-日期(22):    NOW CURRENT_DATE CURRENT_TIMESTAMP
                  YEAR MONTH DAY HOUR MINUTE SECOND EXTRACT
                  DAY_OF_WEEK DAY_OF_YEAR WEEK_OF_YEAR QUARTER
                  DATE_TRUNC DATE_DIFF DATE_ADD DATE_SUB
                  DATE_FORMAT TO_DATE TO_TIMESTAMP EPOCH

标量-条件(12):    CAST TRY_CAST COALESCE NULLIF IF TYPEOF
                  HASH ENCODE DECODE BIT_COUNT MURMUR_HASH (CASE/WHEN)

聚合(22):         COUNT(*) COUNT(col) COUNT(DISTINCT)
                  SUM SUM(DISTINCT) AVG AVG(DISTINCT) MIN MAX
                  STDDEV STDDEV_POP VARIANCE VAR_POP
                  MEDIAN PERCENTILE_CONT PERCENTILE_DISC MODE
                  APPROX_COUNT_DISTINCT APPROX_PERCENTILE APPROX_TOP_K
                  ARRAY_AGG STRING_AGG GROUPING

窗口(16):         ROW_NUMBER RANK DENSE_RANK NTILE PERCENT_RANK CUME_DIST
                  LAG LEAD FIRST_VALUE LAST_VALUE NTH_VALUE
                  SUM/AVG/COUNT/MIN/MAX OVER

数组(17):         ARRAY ARRAY_LENGTH ARRAY_CONTAINS ARRAY_POSITION
                  ARRAY_SLICE ARRAY_APPEND ARRAY_PREPEND ARRAY_REMOVE
                  ARRAY_CONCAT ARRAY_SORT ARRAY_REVERSE ARRAY_DISTINCT
                  ARRAY_FLATTEN ARRAY_INTERSECT ARRAY_UNION ARRAY_EXCEPT ARRAY_JOIN

表函数(4):        EXPLODE POSEXPLODE UNNEST GENERATE_SERIES
相似度(2):        JACCARD_SIMILARITY COSINE_SIMILARITY
```

---

## 三、REPL 功能

| 命令                   | 功能         | 默认 |
| ---------------------- | ------------ | ---- |
| `.tables`              | 列出所有表   | —    |
| `.schema <t>`          | 显示表结构   | —    |
| `.analyze <t>`         | 计算统计信息 | —    |
| `.stats <t>`           | 显示统计信息 | —    |
| `.memory`              | 内存使用统计 | —    |
| `.import <f> <t>`      | CSV 导入     | —    |
| `.export <f> <t>`      | CSV 导出     | —    |
| `.timer on/off`        | 计时显示     | on   |
| `.color on/off`        | ANSI 彩色    | off  |
| `.profile on/off`      | 分阶段耗时   | off  |
| `.estimate on/off`     | 行数预估     | off  |
| `.benchmark <n> <sql>` | 性能基准     | —    |
| `.history [n]`         | 查询历史     | —    |
| `.quit`                | 退出         | —    |

---

## 四、已知限制

```
1.  无 DECIMAL (用 float64)
2.  无 COLLATE (按字节序)
3.  无 WAL 崩溃恢复
4.  无 GRANT/REVOKE
5.  UPDATE/DELETE 全表重建
6.  不支持完全相关子查询 (支持 EXISTS/IN 的 JOIN 改写)
7.  递归 CTE 限制: 1000 次迭代 / 1,000,000 行
8.  INT 溢出抛错 (int32 ±2³¹)
9.  FLOAT/0 → ±inf; INT/0 → ERROR
10. FLOAT 与 DOUBLE 内部均为 float64
11. Python 3.8+ 要求
12. 单连接 TCP (无连接池)
```

---

## 五、使用方式

```bash
# REPL 模式 (内存)
python main.py

# REPL 模式 (持久化)
python main.py /path/to/data

# TCP 服务器模式
python main.py --server 5433

```
