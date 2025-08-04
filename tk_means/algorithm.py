import pandas as pd
import pm4py
import numpy as np
import time
import traceback

from pm4py.algo.discovery.inductive.variants.imf import IMFUVCL
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.util.compression.dtypes import UVCL
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

import multiprocessing as mp
import multiprocessing.pool as mp_pool


import concurrent.futures

import warnings

# annoying pandas warnings from pm4py, silence them
warnings.filterwarnings("ignore", category=Warning, module="pm4py")
warnings.filterwarnings("ignore", category=Warning, module="pd")
warnings.filterwarnings("ignore", category=Warning, module="pandas")

# TODO check those parameters?
PARAMS_HM = {heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99}
# Removing infrequencies, e.g. edges between infrequent activities are removed
PARAMS_IMF = {"noise_threshold": 0.2}
PARAMS_DFG = {"noise_threshold": 0.2}

CONFORMANCES = {
    "tbr": pm4py.conformance_diagnostics_token_based_replay,
    "al": pm4py.conformance_diagnostics_alignments,
}
CONFORMANCE_TIMEOUT = 2 * 60

np.random.seed(42)

def tk_means(params):
    log, k, pm, cc, max_iterations, ds = params
    print(f"Starting tk-means for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}")

    iteration = 0
    cluster_assignment = initialize_clusters(log, k)
    all_models = []
    all_fitness = []
    all_times = []
    while True:
        try:
            start_time = time.time()

            print(f"Discovering process models for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}")
            models = discover_process_model(log, pm, cluster_assignment)
            print(f"Checkign conformance for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}")
            fitness = check_conformance(log, models, cc)
            ca_col = f"cluster_assignment_{iteration}"
            
            print(f"Reassining clusters for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}")
            cluster_assignment[ca_col] = reassign_clusters(fitness, cluster_assignment, k)
            all_models.append(models)
            all_fitness.append(fitness)

            end_time = time.time()
            execution_seconds = end_time - start_time
            all_times.append(execution_seconds)
            
            if check_convergence(cluster_assignment, iteration, max_iterations):
                print(f"Tk-means run complete with parameters k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}")
                return (cluster_assignment, all_fitness, all_models, all_times) # df, list[df], list[models]
            iteration += 1
        except Exception as e:
            print(f"Tk-means run failed with exceptions with parameters k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}")
            print(traceback.format_exc())
            return None


def check_convergence(cluster_assignment, iteration, max_iterations):
    # Reached maximum number of iterations
    print(f"Checking conformance for iteration {iteration} of {max_iterations}....")
    if iteration >= max_iterations:
        return True
    # Check if the last two columns are the same
    before_last_col, last_col = cluster_assignment.columns[-2:]
    cond = cluster_assignment[before_last_col].equals(cluster_assignment[last_col])
    if cond:
        return True
    return False


def reassign_clusters(fitness, cluster_assignment, k):
    # Decide on the cluster re-assignment per trace depending on the fitness
    best_cluster = determine_best_cluster(fitness, cluster_assignment)
    best_cluster = ensure_cluster_non_empty(best_cluster, k)
    return best_cluster

         
def ensure_cluster_non_empty(best_cluster, k):
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
        

def determine_best_cluster(fitness, cluster_assignment):
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
        tied_clusters = row[row == True].index# .tolist()
        prev_cluster = cluster_assignment.loc[trace][prev_cluster_col]
        
        if prev_cluster in tied_clusters:
            best_cluster[trace] = prev_cluster
        else:
            best_cluster[trace] = tied_clusters[-1]  # rightmost cluster
    best_cluster = best_cluster.astype("int")
    return best_cluster


def _calculate_fitness(log, model, cc_func, f_col="fitness"):
    net, initial_marking, final_marking = model
    # Start timining
    conformance = cc_func(log, net, initial_marking, final_marking) 
    conformance = pd.DataFrame(conformance)
    if f_col not in conformance.columns:  # Different conformance check methods return different column names for fitnesss
        f_col = "trace_fitness"
        assert f_col in conformance.columns, "Conformance check did not return a fitness column"
    conformance_value = conformance[f_col].values[0]
    return conformance_value


def check_conformance(log, models, cc):
    # Calculate fitness on variant level as this saves time:
    # fitnes is the same for all traces of the same variant
    trace_to_variant = log[['@@case_index', 'variant_id']]
    cc_func = CONFORMANCES[cc]

    vid_to_fitness = {}
    for vid, df_ in log.groupby("variant_id"):
        case_index = np.random.choice(df_["@@case_index"].values)
        log_select = df_[df_["@@case_index"] == case_index]
        fitnesses = []
        for model in models:
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(_calculate_fitness, log_select, model, cc_func)
                    conformance_value = future.result(timeout=CONFORMANCE_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print("Timeout for conformance check")
                conformance_value = 0
            fitnesses.append(conformance_value)
        vid_to_fitness[vid] = fitnesses
    variants_fitness = pd.DataFrame(vid_to_fitness).T

    variants_fitness = variants_fitness.reset_index()
    variants_fitness = variants_fitness.rename(columns={"index": "variant_id"})
    fitness = variants_fitness.merge(trace_to_variant).sort_values("@@case_index", ascending=True)
    fitness = fitness.drop_duplicates(subset="@@case_index")
    fitness = fitness.drop(columns=["@@case_index", "variant_id"])
    fitness = fitness.reset_index(drop=True)
    return fitness


def discover_process_model(log, pm, cluster_assignment):
    if pm not in {"imf", "hm", "dfg"}:
        raise Exception(f"Unknown process discovery model {pm}")
    if pm == "imf":
        models = discover_process_model_imf(log, cluster_assignment)
    if pm == "hm":
        models = discover_process_model_hm(log, cluster_assignment)
    if pm == "dfg":
        models = discover_process_model_dfg(log, cluster_assignment)
    return models


def discover_process_model_dfg(log, cluster_assignment):
    models = []
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        dfg, sa, ea = pm4py.discover_directly_follows_graph(traces)
        activities_count = pm4py.get_event_attribute_values(log, "concept:name")
        activities = list(activities_count.keys())

        dfg_cleaned =  pm4py.algo.filtering.dfg.dfg_filtering.clean_dfg_based_on_noise_thresh(
            dfg,
            activities,
            noise_threshold=PARAMS_DFG["noise_threshold"],
        )
        model = pm4py.convert_to_petri_net(dfg_cleaned, sa, ea)
        models.append(model)
    return models


def discover_process_model_imf(log, cluster_assignment):
    # Discover pm with inductive miner infrequent
    models = []
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        # TODO why necessary?
        # TODO replace this
        variants = pm4py.get_variants(traces)
        uvcl = UVCL()

        for var, occ in variants.items():
            uvcl[var] = occ

        imfuvcl = IMFUVCL(PARAMS_IMF)
        tree = imfuvcl.apply(IMDataStructureUVCL(uvcl), parameters=PARAMS_IMF)
        model = pm4py.convert_to_petri_net(tree)
        models.append(model)
    return models


def discover_process_model_hm(log, cluster_assignment):
    # Discover pm with heuristic miner algorithm
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column

    models = []
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        models.append(
            heuristics_miner.apply(traces, parameters=PARAMS_HM)
        )
    return models


def initialize_clusters(log,  k):
    trace_to_variant = log[
        ['case:concept:name', 'variant_id', 'variant']
    ]
    variants_counts = trace_to_variant.groupby(
        by=["variant_id", "variant",]
    ).count()["case:concept:name"].reset_index()
    variants_counts = variants_counts.rename(
        columns={"case:concept:name": "count"}
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
            "cluster_assignment_init"
        ] = i

    cluster_assignment = cluster_assignment.drop_duplicates(subset="@@case_index")
    cluster_assignment = cluster_assignment.rename(
        columns={"@@case_index": "case_index"},
    )
    cluster_assignment = cluster_assignment.drop(columns="variant_id")
    cluster_assignment = cluster_assignment.reset_index(drop=True)

    return cluster_assignment
