import numpy as np

import nsga2


def test_fronts_simple():
    F = np.array([[1, 5], [2, 3], [3, 1], [2, 4], [4, 4], [5, 5]], float)
    fr = nsga2.non_dominated_sort(F)
    assert sorted(fr[0]) == [0, 1, 2]
    assert fr[1] == [3] and fr[2] == [4] and fr[3] == [5]  # (2,4) dominates (4,4)


def test_constrained_domination():
    F = np.array([[0, 0], [1, 1], [5, 5]], float)
    V = np.array([10.0, 0.0, 0.0])  # the 'best' point is infeasible
    fr = nsga2.non_dominated_sort(F, V)
    assert fr[0] == [1] and fr[1] == [2] and fr[2] == [0]
    assert nsga2.dominates([9, 9], [0, 0], 1.0, 2.0)  # both infeasible: smaller violation wins


def test_crowding_extremes_infinite_and_order():
    F = np.array([[0, 4], [1, 3], [1.1, 2.9], [4, 0]], float)
    cd = nsga2.crowding_distance(F)
    assert np.isinf(cd[0]) and np.isinf(cd[3]) and cd[1] > 0 and cd[2] > 0
    order = nsga2.rank_population(F)
    assert set(order[:2]) == {0, 3}


def test_nan_treated_as_worst():
    F = np.array([[np.nan, 1], [1, 1]], float)
    assert nsga2.non_dominated_sort(F)[0] == [1]


def test_select_returns_k():
    F = np.random.default_rng(0).random((30, 3))
    s = nsga2.select(F, 10)
    assert len(s) == 10 and len(set(s)) == 10
