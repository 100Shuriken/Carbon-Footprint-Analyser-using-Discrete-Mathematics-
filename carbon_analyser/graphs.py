"""Graph theory: finding low-carbon routes in a multimodal transport network.

The network is an undirected multigraph G = (V, E). Each edge is a
(u, v, mode, distance_km) tuple. Its weight is the emissions for one
passenger making that journey:

    w(e) = distance_km * f(mode)

Algorithms:
* Dijkstra: the lowest-carbon route (edge weights are non-negative).
* BFS: the route with the fewest legs (unweighted shortest path).
* Floyd–Warshall: an all-pairs lowest-carbon matrix.
* Kruskal with union–find: a minimum spanning tree, i.e. the cheapest-carbon
  set of links that still connects every node (for example, when planning a
  shuttle network).
* Degree sequence and connectivity checks.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .emission_factors import factor


@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    mode: str
    distance_km: float

    @property
    def emissions(self) -> float:
        return self.distance_km * factor(self.mode)

    def other(self, node: str) -> str:
        return self.v if node == self.u else self.u


@dataclass
class Route:
    nodes: list[str]
    edges: list[Edge]

    @property
    def emissions(self) -> float:
        return sum(e.emissions for e in self.edges)

    @property
    def distance_km(self) -> float:
        return sum(e.distance_km for e in self.edges)

    def describe(self) -> str:
        if not self.edges:
            return self.nodes[0] if self.nodes else "(empty)"
        parts = [self.nodes[0]]
        for e, n in zip(self.edges, self.nodes[1:]):
            parts.append(f" --{e.mode} {e.distance_km:g}km--> {n}")
        return "".join(parts)


class TransportGraph:
    def __init__(self) -> None:
        self.adj: dict[str, list[Edge]] = {}
        self.edges: list[Edge] = []

    # -- construction --------------------------------------------------------
    def add_node(self, n: str) -> None:
        self.adj.setdefault(n, [])

    def add_edge(self, u: str, v: str, mode: str, distance_km: float) -> Edge:
        factor(mode)  # validate the mode is in the domain of f
        if distance_km < 0:
            raise ValueError("distance must be non-negative")
        e = Edge(u, v, mode, distance_km)
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append(e)
        self.adj[v].append(e)
        self.edges.append(e)
        return e

    @classmethod
    def from_json(cls, path: str | Path) -> "TransportGraph":
        data = json.loads(Path(path).read_text())
        g = cls()
        for n in data.get("nodes", []):
            g.add_node(n)
        for e in data["edges"]:
            g.add_edge(e["from"], e["to"], e["mode"], e["km"])
        return g

    @property
    def nodes(self) -> list[str]:
        return sorted(self.adj)

    # -- basic properties ----------------------------------------------------
    def degree(self, n: str) -> int:
        return len(self.adj[n])

    def degree_sequence(self) -> list[int]:
        return sorted((self.degree(n) for n in self.adj), reverse=True)

    def handshake_holds(self) -> bool:
        """Handshaking lemma: sum of degrees = 2|E|."""
        return sum(self.degree(n) for n in self.adj) == 2 * len(self.edges)

    def is_connected(self) -> bool:
        if not self.adj:
            return True
        start = next(iter(self.adj))
        return len(self._reachable(start)) == len(self.adj)

    def _reachable(self, start: str) -> set[str]:
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for e in self.adj[u]:
                w = e.other(u)
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        return seen

    def _check(self, *nodes: str) -> None:
        for n in nodes:
            if n not in self.adj:
                raise KeyError(f"Unknown node: {n}")

    # -- paths ---------------------------------------------------------------
    def fewest_legs(self, src: str, dst: str, allowed_modes: set[str] | None = None) -> Route | None:
        """BFS. Among the routes with the fewest legs, it keeps the first one found."""
        self._check(src, dst)
        prev: dict[str, tuple[str, Edge] | None] = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for e in self.adj[u]:
                if allowed_modes is not None and e.mode not in allowed_modes:
                    continue
                w = e.other(u)
                if w not in prev:
                    prev[w] = (u, e)
                    q.append(w)
        return self._rebuild(prev, src, dst)

    def greenest_route(
        self,
        src: str,
        dst: str,
        allowed_modes: set[str] | None = None,
        weight: str = "emissions",
    ) -> Route | None:
        """Dijkstra's algorithm. Set ``weight='distance'`` to get the physically shortest route instead."""
        self._check(src, dst)

        def w_of(e: Edge) -> float:
            return e.emissions if weight == "emissions" else e.distance_km

        dist = {src: 0.0}
        prev: dict[str, tuple[str, Edge] | None] = {src: None}
        pq: list[tuple[float, str]] = [(0.0, src)]
        done: set[str] = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            if u == dst:
                break
            for e in self.adj[u]:
                if allowed_modes is not None and e.mode not in allowed_modes:
                    continue
                v = e.other(u)
                nd = d + w_of(e)
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, e)
                    heapq.heappush(pq, (nd, v))
        return self._rebuild(prev, src, dst)

    @staticmethod
    def _rebuild(prev: dict, src: str, dst: str) -> Route | None:
        if dst not in prev:
            return None
        nodes, edges = [dst], []
        cur = dst
        while cur != src:
            p, e = prev[cur]
            edges.append(e)
            nodes.append(p)
            cur = p
        return Route(nodes[::-1], edges[::-1])

    def all_pairs_emissions(self) -> tuple[list[str], list[list[float]]]:
        """Floyd–Warshall on the minimum-weight edge between each pair of nodes."""
        nodes = self.nodes
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        d = [[0.0 if i == j else math.inf for j in range(n)] for i in range(n)]
        for e in self.edges:
            i, j = idx[e.u], idx[e.v]
            if e.emissions < d[i][j]:
                d[i][j] = d[j][i] = e.emissions
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if d[i][k] + d[k][j] < d[i][j]:
                        d[i][j] = d[i][k] + d[k][j]
        return nodes, d

    def adjacency_matrix(self) -> tuple[list[str], list[list[int]]]:
        """Entry (i, j) counts the parallel edges (modes) between node i and node j."""
        nodes = self.nodes
        idx = {n: i for i, n in enumerate(nodes)}
        m = [[0] * len(nodes) for _ in nodes]
        for e in self.edges:
            m[idx[e.u]][idx[e.v]] += 1
            if e.u != e.v:
                m[idx[e.v]][idx[e.u]] += 1
        return nodes, m

    # -- spanning tree -------------------------------------------------------
    def minimum_spanning_tree(self) -> list[Edge]:
        """Kruskal's algorithm with union–find (path compression and union by rank).

        Returns a minimum spanning forest if the graph is disconnected.
        """
        parent = {n: n for n in self.adj}
        rank = {n: 0 for n in self.adj}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        tree = []
        for e in sorted(self.edges, key=lambda e: (e.emissions, e.u, e.v, e.mode)):
            if union(e.u, e.v):
                tree.append(e)
        return tree
