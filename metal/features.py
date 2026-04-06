from __future__ import annotations
"""集中检测所有可选特性。全项目只检测一次，其他模块直接引用。

用法:
    from metal.features import HAS_BLOOM, HAS_ART, BloomFilter
    if HAS_BLOOM:
        bf = BloomFilter(1000)
"""

# ═══ 数据结构 ═══

try:
    from structures.bloom_filter import BloomFilter
    HAS_BLOOM = True
except ImportError:
    BloomFilter = None  # type: ignore
    HAS_BLOOM = False

try:
    from structures.roaring_bitmap import RoaringBitmap
    HAS_ROARING = True
except ImportError:
    RoaringBitmap = None  # type: ignore
    HAS_ROARING = False

try:
    from structures.art import AdaptiveRadixTree
    HAS_ART = True
except ImportError:
    AdaptiveRadixTree = None  # type: ignore
    HAS_ART = False

try:
    from structures.skip_list import SkipList
    HAS_SKIP = True
except ImportError:
    SkipList = None  # type: ignore
    HAS_SKIP = False

try:
    from structures.robin_hood_ht import RobinHoodHashTable
    HAS_ROBIN_HOOD = True
except ImportError:
    RobinHoodHashTable = None  # type: ignore
    HAS_ROBIN_HOOD = False

try:
    from structures.cuckoo_filter import CuckooFilter
    HAS_CUCKOO = True
except ImportError:
    CuckooFilter = None  # type: ignore
    HAS_CUCKOO = False

try:
    from structures.ribbon_filter import RibbonFilter
    HAS_RIBBON = True
except ImportError:
    RibbonFilter = None  # type: ignore
    HAS_RIBBON = False

try:
    from structures.xor_filter import XorFilter
    HAS_XOR = True
except ImportError:
    XorFilter = None  # type: ignore
    HAS_XOR = False

try:
    from structures.fenwick_tree import FenwickTree
    HAS_FENWICK = True
except ImportError:
    FenwickTree = None  # type: ignore
    HAS_FENWICK = False

try:
    from structures.segment_tree import SegmentTree
    HAS_SEGTREE = True
except ImportError:
    SegmentTree = None  # type: ignore
    HAS_SEGTREE = False

try:
    from structures.sparse_table import SparseTableMin, SparseTableMax
    HAS_SPARSE = True
except ImportError:
    SparseTableMin = SparseTableMax = None  # type: ignore
    HAS_SPARSE = False

try:
    from structures.wavelet_tree import WaveletTree
    HAS_WAVELET = True
except ImportError:
    WaveletTree = None  # type: ignore
    HAS_WAVELET = False

try:
    from structures.sorted_container import SortedList
    HAS_SORTED = True
except ImportError:
    SortedList = None  # type: ignore
    HAS_SORTED = False

try:
    from structures.tournament_tree import LoserTree
    HAS_LOSER = True
except ImportError:
    LoserTree = None  # type: ignore
    HAS_LOSER = False

# ═══ Metal 层 ═══

try:
    from metal.arena import Arena
    HAS_ARENA = True
except ImportError:
    Arena = None  # type: ignore
    HAS_ARENA = False

try:
    from metal.slab import SlabAllocator
    HAS_SLAB = True
except ImportError:
    SlabAllocator = None  # type: ignore
    HAS_SLAB = False

try:
    from metal.inline_string import InlineStringStore
    HAS_INLINE = True
except ImportError:
    InlineStringStore = None  # type: ignore
    HAS_INLINE = False

try:
    from metal.bitmagic import (
        nan_pack_float, nan_unpack_float,
        nan_pack_int, nan_pack_bool, nan_pack_null,
        nan_is_null, nan_unpack, NULL_TAG,
        nanbox_batch_eq, nanbox_batch_lt, nanbox_batch_gt,
        nanbox_batch_add, nanbox_batch_sub, nanbox_batch_mul,
        pdep, pext, bitmap_gather, select64, rank64,
    )
    HAS_NANBOX = True
    HAS_NANBOX_BATCH = True
except ImportError:
    HAS_NANBOX = False
    HAS_NANBOX_BATCH = False
    NULL_TAG = 0x7FF8000000000001
    # 占位函数，防止未检测直接调用时崩溃
    nan_pack_float = nan_unpack_float = None  # type: ignore
    nan_pack_int = nan_pack_bool = nan_pack_null = None  # type: ignore
    nan_is_null = nan_unpack = None  # type: ignore
    nanbox_batch_eq = nanbox_batch_lt = nanbox_batch_gt = None  # type: ignore
    nanbox_batch_add = nanbox_batch_sub = nanbox_batch_mul = None  # type: ignore
    pdep = pext = bitmap_gather = select64 = rank64 = None  # type: ignore

try:
    from metal.swar import batch_to_upper, batch_to_lower
    HAS_SWAR = True
except ImportError:
    batch_to_upper = batch_to_lower = None  # type: ignore
    HAS_SWAR = False

try:
    from metal.advanced_hash import (
        ZobristHasher, CuckooHashMap, WriteCombiningBuffer)
    HAS_ZOBRIST = True
    HAS_CUCKOO_HM = True
    HAS_WCB = True
except ImportError:
    ZobristHasher = CuckooHashMap = WriteCombiningBuffer = None  # type: ignore
    HAS_ZOBRIST = False
    HAS_CUCKOO_HM = False
    HAS_WCB = False

# ═══ 存储/压缩 ═══

try:
    from storage.compression.dict_codec import DictEncoded
    HAS_DICT = True
except ImportError:
    DictEncoded = None  # type: ignore
    HAS_DICT = False

try:
    from storage.compression.rle import rle_encode, rle_decode
    HAS_RLE = True
except ImportError:
    HAS_RLE = False

try:
    from storage.compression.delta import delta_encode, delta_decode
    HAS_DELTA_CODEC = True
except ImportError:
    HAS_DELTA_CODEC = False

try:
    from storage.compression.bitpack import for_encode, for_decode
    HAS_FOR = True
except ImportError:
    HAS_FOR = False

try:
    from storage.compression.gorilla import gorilla_encode, gorilla_decode
    HAS_GORILLA = True
except ImportError:
    HAS_GORILLA = False

try:
    from storage.compression.alp import alp_encode, alp_decode
    HAS_ALP = True
except ImportError:
    HAS_ALP = False

try:
    from storage.compression.fsst import SymbolTable, fsst_encode, fsst_decode
    HAS_FSST = True
except ImportError:
    HAS_FSST = False

try:
    from storage.compression.analyzer import analyze_and_choose
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False

try:
    from storage.lsm.lsm_store import LSMStore
    HAS_LSM = True
except ImportError:
    LSMStore = None  # type: ignore
    HAS_LSM = False

try:
    from storage.io.buffer_pool import BufferPool
    HAS_BUFFER_POOL = True
except ImportError:
    BufferPool = None  # type: ignore
    HAS_BUFFER_POOL = False

try:
    from storage.hybrid.delta_store import DeltaStore
    HAS_DELTA_STORE = True
except ImportError:
    DeltaStore = None  # type: ignore
    HAS_DELTA_STORE = False

try:
    from storage.hybrid.merge_worker import MergeWorker
    HAS_MERGE_WORKER = True
except ImportError:
    MergeWorker = None  # type: ignore
    HAS_MERGE_WORKER = False

# ═══ 执行器组件 ═══

try:
    from executor.codegen.compiler import ExprCompiler
    from executor.codegen.cache import CompileCache
    HAS_JIT = True
except ImportError:
    ExprCompiler = CompileCache = None  # type: ignore
    HAS_JIT = False

try:
    from executor.core.lazy_batch import LazyBatch
    HAS_LAZY = True
except ImportError:
    LazyBatch = None  # type: ignore
    HAS_LAZY = False

try:
    from executor.pipeline.morsel import MorselDriver
    HAS_MORSEL = True
except ImportError:
    MorselDriver = None  # type: ignore
    HAS_MORSEL = False

try:
    from executor.string_algo.boyer_moore import BoyerMoore
    HAS_BM = True
except ImportError:
    BoyerMoore = None  # type: ignore
    HAS_BM = False

try:
    from executor.string_algo.compiled_date import (
        ISO_DATE_PARSER, parse_date_auto)
    HAS_COMPILED_DATE = True
except ImportError:
    ISO_DATE_PARSER = parse_date_auto = None  # type: ignore
    HAS_COMPILED_DATE = False

try:
    from executor.vectorized_ops import (
        try_vectorized_arith, try_vectorized_cmp_scalar)
    HAS_VEC_OPS = True
except ImportError:
    HAS_VEC_OPS = False

try:
    from executor.similarity.minhash_lsh import (
        jaccard_exact, cosine_similarity)
    HAS_SIMILARITY = True
except ImportError:
    HAS_SIMILARITY = False

# ═══ 规划器 ═══

try:
    from planner.cost_model import CostModel, CostEstimate
    from planner.cardinality import CardinalityEstimator
    from planner.rules import (
        PredicateReorder, PredicatePushdown, TopNPushdown)
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False

try:
    from planner.join_reorder import JoinGraph, DPccp
    HAS_DPCCP = True
except ImportError:
    HAS_DPCCP = False

try:
    from planner.runtime_optimizer import RuntimeOptimizer
    HAS_RUNTIME_OPT = True
except ImportError:
    RuntimeOptimizer = None  # type: ignore
    HAS_RUNTIME_OPT = False

# ═══ 执行器算子 ═══

try:
    from executor.operators.sort.top_n import TopNOperator
    HAS_TOPN = True
except ImportError:
    TopNOperator = None  # type: ignore
    HAS_TOPN = False

try:
    from executor.operators.join.nested_loop_join import NestedLoopJoinOperator
    HAS_NL = True
except ImportError:
    NestedLoopJoinOperator = None  # type: ignore
    HAS_NL = False

try:
    from executor.operators.join.radix_join import RadixJoinOperator
    HAS_RADIX = True
except ImportError:
    RadixJoinOperator = None  # type: ignore
    HAS_RADIX = False

try:
    from executor.operators.join.grace_join import GraceHashJoinOperator
    HAS_GRACE = True
except ImportError:
    GraceHashJoinOperator = None  # type: ignore
    HAS_GRACE = False

try:
    from executor.operators.join.sort_merge_join import SortMergeJoinOperator
    HAS_SMJ = True
except ImportError:
    SortMergeJoinOperator = None  # type: ignore
    HAS_SMJ = False

try:
    from executor.operators.scan.index_scan import IndexScanOperator
    HAS_INDEX_SCAN = True
except ImportError:
    IndexScanOperator = None  # type: ignore
    HAS_INDEX_SCAN = False

try:
    from executor.operators.scan.parallel_scan import ParallelScanOperator
    HAS_PARALLEL = True
except ImportError:
    ParallelScanOperator = None  # type: ignore
    HAS_PARALLEL = False

try:
    from executor.operators.scan.zone_map_scan import ZoneMapScanOperator
    HAS_ZONEMAP_SCAN = True
except ImportError:
    ZoneMapScanOperator = None  # type: ignore
    HAS_ZONEMAP_SCAN = False

try:
    from executor.adaptive.micro_engine import MicroAdaptiveEngine
    HAS_ADAPTIVE = True
except ImportError:
    MicroAdaptiveEngine = None  # type: ignore
    HAS_ADAPTIVE = False

try:
    from executor.adaptive.strategy import StrategySelector
    HAS_STRATEGY = True
except ImportError:
    StrategySelector = None  # type: ignore
    HAS_STRATEGY = False

try:
    from executor.pipeline.fuser import try_fuse
    HAS_FUSER = True
except ImportError:
    HAS_FUSER = False

try:
    from executor.compressed_execution import (
        rle_aggregate_direct, can_use_rle_execution,
        can_use_dict_execution, dict_group_aggregate)
    HAS_COMPRESSED = True
except ImportError:
    HAS_COMPRESSED = False

try:
    from executor.query_coordinator import QueryCoordinator
    HAS_COORDINATOR = True
except ImportError:
    QueryCoordinator = None  # type: ignore
    HAS_COORDINATOR = False

try:
    from catalog.index_manager import IndexManager
    HAS_INDEX = True
except ImportError:
    IndexManager = None  # type: ignore
    HAS_INDEX = False
