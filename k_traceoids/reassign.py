import pandas as pd
import numpy as np


def reassign_clusters(fitness, cluster_assignment, k):
    # Decide on the cluster re-assignment per trace depending on the fitness
    best_cluster = _determine_best_cluster(fitness, cluster_assignment)
    best_cluster = _ensure_cluster_non_empty(best_cluster, k)
    return best_cluster


def _determine_best_cluster(fitness, cluster_assignment):
    # 1. Find the column/cluster with the maximum fitness value per row
    # 2. If there are multiple maximum values, pick the rightmost column.
    # 3. If multiple clusters are tied for max fitness,
    #    and the cluster assignment of the previous iteration is one of them,
    #    keep the old cluster assignment.
    row_max = fitness.max(axis=1)
    is_max = fitness.eq(row_max, axis=0)
    best_cluster = pd.Series(index=range(len(fitness)))

    prev_cluster_col = cluster_assignment.columns[-1]

    for trace, row in is_max.iterrows():
        tied_clusters = row[row == True].index  # noqa
        prev_cluster = cluster_assignment.loc[trace][prev_cluster_col]

        if prev_cluster in tied_clusters:
            best_cluster[trace] = prev_cluster
        else:
            best_cluster[trace] = tied_clusters[-1]  # rightmost cluster
    best_cluster = best_cluster.astype("int")
    return best_cluster


def _ensure_cluster_non_empty(best_cluster, k):
    # Idea: instead of re-assigning, just prevent one trace to leave?
    # Advantage: information about the "least fitting trace"?
    # Intuition: k-means would prevent this
    expected_clusters = np.arange(k)
    actual_clusters = best_cluster.unique()

    empty_clusters = np.setdiff1d(expected_clusters, actual_clusters)
    if len(empty_clusters) == 0:
        # All k clusters are not empty in current configuration
        return best_cluster
    else:
        # Detected empty clusters after re-assignment
        # --> Randomly assign new traces to this cluster to be the new centroid in the next iteration
        # TODO this strategy is probably too simple?
        ixs = np.random.choice(
            best_cluster.index.values,
            size=len(empty_clusters),
            replace=False,
        )
        best_cluster[ixs] = empty_clusters
        return best_cluster
