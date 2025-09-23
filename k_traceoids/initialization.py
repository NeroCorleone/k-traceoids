import numpy as np


def initialize_clusters(log, k):
    trace_to_variant = log[["case:concept:name", "variant_id", "variant"]]
    variants_counts = (
        trace_to_variant.groupby(
            by=["variant_id", "variant"],
        )
        .count()["case:concept:name"]
        .reset_index()
    )
    variants_counts = variants_counts.rename(
        columns={"case:concept:name": "count"},
    )
    # Assign varaints to k groups and track
    # the number of traces in each group
    # Iteratively assigning each variant (sorted by amount of traces in there)
    # into the group with the least nb of traces
    groups = [[] for _ in range(k)]
    group_sizes = [0] * k

    for i, row in variants_counts.iterrows():
        # Find the group with the least total size
        group_index = min(range(k), key=lambda i: group_sizes[i])
        vid = row["variant_id"]
        groups[group_index].append(vid)
        group_sizes[group_index] += row["count"]

    # cluster_assignment = pd.DataFrame(data={"case_index": log["@@case_index"].unique()})
    cluster_assignment = log[["@@case_index", "variant_id"]].copy()
    cluster_assignment["cluster_assignment_init"] = np.nan

    for i, g in enumerate(groups):
        cluster_assignment.loc[
            cluster_assignment["variant_id"].isin(g),
            "cluster_assignment_init",
        ] = i

    cluster_assignment = cluster_assignment.drop_duplicates(
        subset="@@case_index",
    )
    cluster_assignment = cluster_assignment.rename(
        columns={"@@case_index": "case_index"},
    )
    cluster_assignment = cluster_assignment.drop(columns="variant_id")
    cluster_assignment = cluster_assignment.reset_index(drop=True)

    return cluster_assignment
