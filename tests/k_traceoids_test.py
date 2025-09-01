import pandas as pd

import k_traceoids.algorithm as ktr


def test__best_cluster_assignment():
    k = 3
    fitness = pd.DataFrame(
        {
            "0": [0.8, 0.9, 0.9, 0.3],
            "1": [0.6, 0.9, 0.8, 0.3],
            "2": [0.4, 0.5, 0.9, 0.3],
        }
    )

    cluster_assignment = pd.DataFrame(
        {
            "cluster_assignment_init": [0, 1, 1, 2],
        }
    )

    expected = pd.Series(
        {
            0: 0,  # unique max in row
            1: 1,  # tie, prev match
            2: 2,  # tie, no prev match (rightmost)
            3: 2,  # all equal, prev match
        }
    )

    result = ktr.reassign_clusters(
        fitness,
        cluster_assignment,
        k,
    )

    pd.testing.assert_series_equal(result, expected)


def test__ensure_cluster_non_empty():
    k = 3

    best_cluster = pd.Series([0, 1, 2])
    expected = pd.Series([0, 1, 2])
    actual = ktr.ensure_cluster_non_empty(best_cluster, k)
    pd.testing.assert_series_equal(expected, actual)

    best_cluster = pd.Series([0, 1, 0])
    actual = ktr.ensure_cluster_non_empty(best_cluster, k)
    assert 2 in actual

    best_cluster = pd.Series([0, 0, 0])
    actual = ktr.ensure_cluster_non_empty(best_cluster, k)
    assert 2 in actual
    assert 1 in actual
