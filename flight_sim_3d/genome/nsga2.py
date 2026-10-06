"""NSGA-II building blocks (Deb et al. 2002): constrained non-dominated sort + crowding distance.

All objectives are minimized. Constraint handling (Deb's constrained domination):
a feasible solution (violation == 0) dominates any infeasible one; between two
infeasible ones the smaller violation wins; between feasible ones, normal
Pareto dominance. NaN objectives are treated as +inf.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def _clean(F) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return np.where(np.isnan(F), np.inf, F)


def dominates(a, b, va: float = 0.0, vb: float = 0.0) -> bool:
    if va > 0 or vb > 0:
        if va == 0:
            return True
        if vb == 0:
            return False
        return va < vb
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F, violation: Optional[Sequence[float]] = None) -> List[List[int]]:
    """Fronts as lists of indices, best front first."""
    F = _clean(F)
    n = F.shape[0]
    v = np.zeros(n) if violation is None else np.asarray(violation, float)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    for p in range(n):
        for q in range(p + 1, n):
            if dominates(F[p], F[q], v[p], v[q]):
                S[p].append(q)
                cnt[q] += 1
            elif dominates(F[q], F[p], v[q], v[p]):
                S[q].append(p)
                cnt[p] += 1
    fronts, cur = [], [i for i in range(n) if cnt[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for p in cur:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        cur = sorted(nxt)
    return fronts


def crowding_distance(F) -> np.ndarray:
    F = _clean(F)
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        col = F[order, j]
        d[order[0]] = d[order[-1]] = np.inf
        span = col[-1] - col[0]
        if not np.isfinite(span) or span <= 0:
            continue
        d[order[1:-1]] += (col[2:] - col[:-2]) / span
    return d


def rank_population(F, violation=None) -> np.ndarray:
    """Indices sorted best-first by (front, -crowding). Feed the reordered population to ga.next_generation."""
    fronts = non_dominated_sort(F, violation)
    F = _clean(F)
    order = []
    for fr in fronts:
        cd = crowding_distance(F[fr])
        order += [fr[i] for i in np.argsort(-cd, kind="stable")]
    return np.array(order, int)


def select(F, k: int, violation=None) -> np.ndarray:
    """NSGA-II environmental selection: the best k indices."""
    return rank_population(F, violation)[:k]
