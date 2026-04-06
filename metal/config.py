from __future__ import annotations
"""全局配置常量。所有可调阈值集中于此，避免散落在各模块中。"""

import os

# ═══ 存储层 ═══
CHUNK_SIZE = 65536              # 每个 ColumnChunk 的最大行数（64K）
BATCH_SIZE = 1024               # 向量化批处理大小
MINI_BLOCK_SIZE = 1024          # 压缩 mini-block 大小
SEGMENT_SIZE = 1024             # ColumnChunk 分段解压大小
MAX_CACHED_SEGMENTS = 8         # 每个 chunk 最多缓存段数

# ═══ 内存管理 ═══
DEFAULT_MEMORY_LIMIT = 256 * 1024 * 1024  # 算子内存预算（256 MB）
ARENA_BLOCK_SIZE = 1024 * 1024            # Arena 块大小（1 MB）
RAW_THRESHOLD = 4096            # TypedVector 迁移到 RawMemoryBlock 的阈值

# ═══ 自适应引擎分层阈值 ═══
NANO_THRESHOLD = 64             # < 64 行：列表遍历，零开销
MICRO_THRESHOLD = 1024          # < 1K 行：Python dict，最小化设置
STANDARD_THRESHOLD = 100_000    # < 100K 行：向量化 + JIT
TURBO_THRESHOLD = 10_000_000    # < 10M 行：JIT + 并行扫描
# > 10M 行：NUCLEAR 层级（多进程 + 外部排序）

# ═══ 执行器 ═══
MAX_MEMORY_GROUPS = 100_000     # HashAgg 内存中最大组数
BUDGET_CHECK_INTERVAL = 1000    # HashAgg 内存检查频率（每 N 组检查一次）
JIT_THRESHOLD = 512             # Filter 算子启用 JIT 编译的行数阈值
PROJ_JIT_THRESHOLD = 256        # Project 算子启用 JIT 编译的行数阈值

# ═══ 并行度 ═══
DEFAULT_PARALLELISM = os.cpu_count() or 4  # 默认并行度

# ═══ 文件格式 ═══
FILE_MAGIC = b'Z1DB'            # 二进制文件魔数
FILE_VERSION = 1                # 文件格式版本
NULL_HASH_SENTINEL = 0          # NULL 值的哈希值（约定为 0）
