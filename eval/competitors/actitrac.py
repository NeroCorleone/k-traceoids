from collections import Counter
from typing import List, Dict, Set, Callable
import math
import pandas as pd

import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

PARAMS_HM = {
    heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99,
}

Trace = tuple[str]          # e.g. ('A', 'B', 'C')
GroupedLog = dict[Trace, int]  # trace -> frequency
Cluster = set[Trace]


def group_event_log(log: pd.DataFrame) -> Dict[Trace, int]:
    variant_count = pm4py.get_variants_as_tuples(log)
    return variant_count


def select_window(
    remaining: Dict[Trace, int],
    w: float
) -> List[Trace]:
    """
    Select top-w fraction of remaining traces by frequency.
    """
    if w == 0:
        return [max(remaining, key=remaining.get)]

    sorted_traces = sorted(
        remaining.items(),
        key=lambda x: x[1],
        reverse=True
    )
    cutoff = max(1, math.ceil(w * len(sorted_traces)))
    return [t for t, _ in sorted_traces[:cutoff]]

# Distance-based selection (optional, ActiTraCMRA)
def select_candidate(
    window: List[Trace],
    current_cluster: Set[Trace],
    dist: Callable[[Trace, Trace], float] | None
) -> Trace:
    if not current_cluster or dist is None:
        return window[0]

    def avg_distance(trace):
        return sum(dist(trace, c) for c in current_cluster) / len(current_cluster)

    return min(window, key=avg_distance)


def model_fitness(model, sublog) -> float:
    net, initial_marking, final_marking = model
    alignments = pm4py.conformance_diagnostics_alignments(sublog, net, initial_marking, final_marking)
    alignments = pd.DataFrame(alignments)
    fitness = alignments["fitness"].mean()
    return fitness

# Phase 3
def resolve_residuals(
    clusters: List[Set[Trace]],
    remaining: Dict[Trace, int],
    log: pd.DataFrame,
) -> List[Set[Trace]]:
    # Resolve the last traces by adding them to the best fitting cluster
    models = []
    for c in clusters:
        sublog = log[log["variant"].isin(c)]
        m = discover_model(sublog)
        models.append(m)

    for trace in remaining:
        sublog = log[log["variant"].isin([trace])]
        best_cluster = max(
            range(len(clusters)),
            key=lambda i: model_fitness(models[i], sublog)
        )
        clusters[best_cluster].add(trace)
        print(f"Added trace during resolution {len(remaining)}")

    return clusters

def discover_model(sublog):
    model = heuristics_miner.apply(sublog, parameters=PARAMS_HM)
    return model

# Phase 2
def look_ahead_phase(
    cluster: Set[Trace],
    remaining: Dict[Trace, int],
    tf: float,
    log: pd.DataFrame,
) -> Set[Trace]:

    sublog = log[log['variant'].isin(list(cluster))]
    model = discover_model(sublog)
    fitting = set()

    for trace in list(remaining):
        sublog = log[log['variant'].isin(list(trace))]
        if model_fitness(model, trace) >= tf:
            fitting.add(trace)
            remaining.pop(trace)
            print(f"In look ahead: {len(remaining)}")
    return cluster | fitting


# Phase 1
def selection_phase(
    remaining: Dict[Trace, int],
    tf: float,
    mcs: float,
    w: float,
    log: pd.DataFrame,
    dist=None,
) -> tuple: #(Set[Trace], Dict[Trace, int]):

    C = set() # current cluster
    skipped = set()
    print(remaining)
    print(f"Starting with {len(remaining)} variants.")

    while remaining:
        window = select_window(remaining, w)
        candidate = select_candidate(window, C, dist)

        tentative_traces = list(C) + [candidate]
        sublog = log[log['variant'].isin(tentative_traces)]
        model = discover_model(sublog)
        fitness = model_fitness(model, sublog)

        if fitness >= tf:
            C.add(candidate)
            remaining.pop(candidate)
            print(f"Here: {len(remaining)}")
            return C
        else:
            cluster_size = len(C)
            if cluster_size >= mcs * len(remaining):
                print(f"cluster_size: {cluster_size}")
                print(f"remaining: {len(remaining)}")
                # Phase 2: desired cluster is reached
                C = look_ahead_phase(C, remaining, tf, log) 
                # TODO check!
                return C
            else:
                skipped.add(candidate)
                remaining.pop(candidate)
                print(f"skipped: {len(remaining)}")
    return C


# TODO the distance needs implementation
def actitrac(
    log: pd.DataFrame,
    k_max: int,
    tf: float,
    mcs: float,
    w: float = 0.0,
    dist=None
) -> List[Set[Trace]]:

    grouped_log = group_event_log(log)
    remaining = dict(grouped_log)
    clusters = []

    while remaining and len(clusters) < k_max:
        cluster = selection_phase(remaining, tf, mcs, w, log, dist)
        clusters.append(cluster)

    if remaining:
        clusters = resolve_residuals(clusters, remaining, log)

    return clusters
