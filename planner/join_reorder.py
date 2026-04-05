from __future__ import annotations

"""DPccp — Dynamic Programming over Connected Complement Pairs.
Paper: Moerkotte & Neumann, 2006.

Solves the NP-hard join ordering problem OPTIMALLY for ≤15 tables.
Complexity: O(3^n) — exponential but tractable for typical queries.

Why this matters:
  4 tables: 4! = 24 possible orders → brute force OK
  8 tables: 8! = 40320 → too many for brute force
  DPccp with 8 tables: 3^8 = 6561 subproblems → fast

The key insight: only consider CONNECTED subsets of the join graph.
Most subsets are disconnected → skip them → huge speedup over O(2^n * n^2).
"""
import math
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from planner.cost_model import CostEstimate, CostModel
from planner.cardinality import CardinalityEstimator


class JoinNode:
    """Represents a table or intermediate join result."""
    __slots__ = ('tables', 'cost', 'plan', 'rows')

    def __init__(self, tables: FrozenSet[str], cost: CostEstimate,
                 plan: Any, rows: float) -> None:
        self.tables = tables
        self.cost = cost
        self.plan = plan
        self.rows = rows

    def __repr__(self) -> str:
        return f"JoinNode({self.tables}, rows={self.rows:.0f}, cost={self.cost.total:.1f})"


class JoinGraph:
    """Hypergraph of join predicates between tables."""

    def __init__(self) -> None:
        self.tables: Set[str] = set()
        self.edges: List[Tuple[str, str, Any]] = []  # (table_a, table_b, predicate)
        self.table_rows: Dict[str, float] = {}

    def add_table(self, name: str, rows: float) -> None:
        self.tables.add(name)
        self.table_rows[name] = rows

    def add_edge(self, t1: str, t2: str, predicate: Any = None) -> None:
        self.edges.append((t1, t2, predicate))

    def neighbors(self, table_set: FrozenSet[str]) -> Set[str]:
        """Tables connected to table_set via join edges."""
        result: Set[str] = set()
        for t1, t2, _ in self.edges:
            if t1 in table_set and t2 not in table_set:
                result.add(t2)
            if t2 in table_set and t1 not in table_set:
                result.add(t1)
        return result

    def is_connected(self, table_set: FrozenSet[str]) -> bool:
        """Check if table_set forms a connected subgraph."""
        if len(table_set) <= 1:
            return True
        visited: Set[str] = set()
        start = next(iter(table_set))
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop()
            for t1, t2, _ in self.edges:
                if t1 == current and t2 in table_set and t2 not in visited:
                    visited.add(t2);
                    queue.append(t2)
                if t2 == current and t1 in table_set and t1 not in visited:
                    visited.add(t1);
                    queue.append(t1)
        return visited == table_set


class DPccp:
    """DPccp join order optimizer.

    Algorithm:
    1. Enumerate all connected subsets S of the join graph
    2. For each S, enumerate all connected complement pairs (S1, S2)
       where S = S1 ∪ S2, S1 ∩ S2 = ∅, both connected
    3. dp[S] = min over all (S1, S2) of cost(join(dp[S1], dp[S2]))

    Optimality: guaranteed to find the cheapest join order.
    """

    def __init__(self, graph: JoinGraph,
                 estimator: Optional[CardinalityEstimator] = None) -> None:
        self._graph = graph
        self._estimator = estimator or CardinalityEstimator()
        self._dp: Dict[FrozenSet[str], JoinNode] = {}

    def optimize(self) -> JoinNode:
        """Find optimal join order. Returns JoinNode with best plan."""
        tables = sorted(self._graph.tables)
        n = len(tables)

        if n == 0:
            return JoinNode(frozenset(), CostEstimate(), None, 0)
        if n == 1:
            t = tables[0]
            rows = self._graph.table_rows.get(t, 1000)
            cost = CostModel.seq_scan(int(rows))
            return JoinNode(frozenset({t}), cost, ('SCAN', t), rows)

        # Initialize single-table entries
        for t in tables:
            rows = self._graph.table_rows.get(t, 1000)
            cost = CostModel.seq_scan(int(rows))
            self._dp[frozenset({t})] = JoinNode(frozenset({t}), cost, ('SCAN', t), rows)

        # Enumerate connected subsets in order of increasing size
        all_tables = frozenset(tables)
        for size in range(2, n + 1):
            for subset in self._connected_subsets(all_tables, size):
                self._process_subset(subset)

        # Return the full join plan
        result = self._dp.get(all_tables)
        if result is None:
            # Disconnected graph — use cross joins
            result = self._handle_disconnected(tables)
        return result

    def _process_subset(self, s: FrozenSet[str]) -> None:
        """Find best join plan for subset s by trying all partitions."""
        best: Optional[JoinNode] = None
        for s1 in self._complement_pairs(s):
            s2 = s - s1
            if not s2: continue
            if s1 not in self._dp or s2 not in self._dp: continue
            left = self._dp[s1]
            right = self._dp[s2]
            # Try different join algorithms
            join_sel = self._estimate_join_selectivity(s1, s2)
            algo, cost = CostModel.select_join_algorithm(
                left.cost, right.cost, join_sel)
            output_rows = max(1, left.rows * right.rows * join_sel)
            plan = (algo, left.plan, right.plan)
            node = JoinNode(s, cost, plan, output_rows)
            if best is None or cost < best.cost:
                best = node
        if best is not None:
            if s not in self._dp or best.cost < self._dp[s].cost:
                self._dp[s] = best

    def _complement_pairs(self, s: FrozenSet[str]):
        """Enumerate connected subsets s1 of s where s-s1 is also connected."""
        tables = sorted(s)
        n = len(tables)
        # Enumerate proper non-empty subsets
        for mask in range(1, (1 << n) - 1):
            s1 = frozenset(tables[i] for i in range(n) if mask & (1 << i))
            s2 = s - s1
            if not s2: continue
            # Both must be connected
            if not self._graph.is_connected(s1): continue
            if not self._graph.is_connected(s2): continue
            # Avoid duplicate pairs (s1, s2) and (s2, s1)
            if sorted(s1) > sorted(s2): continue
            yield s1

    def _connected_subsets(self, universe: FrozenSet[str], size: int):
        """Generate all connected subsets of given size."""
        tables = sorted(universe)
        n = len(tables)
        for mask in range(1, 1 << n):
            if bin(mask).count('1') != size: continue
            subset = frozenset(tables[i] for i in range(n) if mask & (1 << i))
            if self._graph.is_connected(subset):
                yield subset

    def _estimate_join_selectivity(self, s1: FrozenSet[str],
                                   s2: FrozenSet[str]) -> float:
        """Estimate join selectivity between two table sets."""
        # Find join predicates connecting s1 and s2
        for t1, t2, pred in self._graph.edges:
            if (t1 in s1 and t2 in s2) or (t2 in s1 and t1 in s2):
                # Use NDV-based estimation
                ndv1 = max(1, self._graph.table_rows.get(t1, 1000) * 0.8)
                ndv2 = max(1, self._graph.table_rows.get(t2, 1000) * 0.8)
                return 1.0 / max(ndv1, ndv2)
        # No direct join predicate → cross join
        return 1.0

    def _handle_disconnected(self, tables: list) -> JoinNode:
        """Handle disconnected join graphs via cross joins."""
        components = self._find_components(tables)
        result: Optional[JoinNode] = None
        for comp in components:
            comp_set = frozenset(comp)
            if comp_set in self._dp:
                node = self._dp[comp_set]
            else:
                t = comp[0]
                rows = self._graph.table_rows.get(t, 1000)
                node = JoinNode(comp_set, CostModel.seq_scan(int(rows)),
                                ('SCAN', t), rows)
            if result is None:
                result = node
            else:
                cross_cost = CostModel.nested_loop_join(result.cost, node.cost, 1.0)
                result = JoinNode(
                    result.tables | node.tables,
                    cross_cost,
                    ('CROSS_JOIN', result.plan, node.plan),
                    result.rows * node.rows)
        return result if result else JoinNode(frozenset(), CostEstimate(), None, 0)

    def _find_components(self, tables: list) -> List[List[str]]:
        visited: Set[str] = set()
        components: List[List[str]] = []
        for t in tables:
            if t in visited: continue
            comp = []
            queue = [t]
            while queue:
                curr = queue.pop()
                if curr in visited: continue
                visited.add(curr)
                comp.append(curr)
                for t1, t2, _ in self._graph.edges:
                    if t1 == curr and t2 not in visited: queue.append(t2)
                    if t2 == curr and t1 not in visited: queue.append(t1)
            components.append(comp)
        return components

    def explain(self) -> str:
        """Return human-readable plan."""
        all_tables = frozenset(self._graph.tables)
        result = self._dp.get(all_tables)
        if result is None:
            return "No plan found"
        return self._explain_node(result.plan, 0)

    def _explain_node(self, plan: Any, indent: int) -> str:
        if plan is None: return ""
        prefix = "  " * indent
        if isinstance(plan, tuple):
            if plan[0] == 'SCAN':
                rows = self._graph.table_rows.get(plan[1], 0)
                return f"{prefix}Scan({plan[1]}, rows={rows:.0f})"
            algo = plan[0]
            left = self._explain_node(plan[1], indent + 1)
            right = self._explain_node(plan[2], indent + 1)
            node = self._dp.get(frozenset(self._collect_tables(plan)))
            cost_str = f", cost={node.cost.total:.1f}" if node else ""
            rows_str = f", rows={node.rows:.0f}" if node else ""
            return f"{prefix}{algo}{cost_str}{rows_str}\n{left}\n{right}"
        return f"{prefix}{plan}"

    def _collect_tables(self, plan: Any) -> Set[str]:
        if plan is None: return set()
        if isinstance(plan, tuple):
            if plan[0] == 'SCAN': return {plan[1]}
            return self._collect_tables(plan[1]) | self._collect_tables(plan[2])
        return set()
