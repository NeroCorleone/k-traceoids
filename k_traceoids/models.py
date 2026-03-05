import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.variants.imf import IMFUVCL
from pm4py.util.compression.dtypes import UVCL
from pm4py.algo.discovery.correlation_mining import algorithm as correlation_miner
from pm4py.statistics.start_activities.pandas import get as sa_get
from pm4py.statistics.end_activities.pandas import get as ea_get


# TODO check those parameters?
PARAMS_HM = {
    heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99,
}
# Removing infrequencies, e.g. edges between infrequent activities are removed
PARAMS_IMF = {"noise_threshold": 0.2}
PARAMS_DFG = {"noise_threshold": 0.2}
PARAMS_ILP = {"noise_threshold": 0.8} # This is inverse to the other parameters, i.e. 1.0 is the full log


def _calculate_dfg(log, cluster_assignment, params):
    if params is None:
        params = PARAMS_DFG
    models = []
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        dfg, sa, ea = pm4py.discover_directly_follows_graph(traces)
        activities_count = pm4py.get_event_attribute_values(
            log,
            "concept:name",
        )
        activities = list(activities_count.keys())

        dfg_cleaned = (
            pm4py.algo.filtering.dfg.dfg_filtering.clean_dfg_based_on_noise_thresh(
                dfg,
                activities,
                noise_threshold=params["noise_threshold"],
            )
        )
        model = pm4py.convert_to_petri_net(dfg_cleaned, sa, ea)
        models.append(model)
    return models


def _calculate_imf(log, cluster_assignment, params):
    # Discover pm with inductive miner infrequent
    if params is None:
        params = PARAMS_IMF
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

        imfuvcl = IMFUVCL(params)
        tree = imfuvcl.apply(IMDataStructureUVCL(uvcl), parameters=params)
        model = pm4py.convert_to_petri_net(tree)
        models.append(model)
    return models


def _calculate_hm(log, cluster_assignment, params):
    if params is None:
        params = PARAMS_HM
    # Discover pm with heuristic miner algorithm
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column

    models = []
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        models.append(
            heuristics_miner.apply(traces, parameters=params),
        )
    return models


def _calculate_declare(log, cluster_assignment, params):
    # No noise_threshold parameter here?
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column

    models = []
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        models.append(
            pm4py.discover_declare(traces)
        )
    return models

def _calculate_correlation(log, cluster_assignment, params):
    # No noise_threshold parameter here
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column

    models = []
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        dfg, _ = correlation_miner.apply(traces)  # also returns "performance_dfg" which contains estimations for the edges/arcs
        sa = sa_get.get_start_activities(log)
        ea = ea_get.get_end_activities(log)
        model = pm4py.convert_to_petri_net(dfg, sa, ea)
        models.append(model)
    return models

def _calculate_ilp(log, cluster_assignment, params):
    if params is None:
        params = PARAMS_ILP
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column

    models = []
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        models.append(
            pm4py.discover_petri_net_ilp(traces, alpha=params["noise_threshold"])
        )
    return models

MODELS = {
    "imf": _calculate_imf,
    "hm": _calculate_hm,
    "dfg": _calculate_dfg,
    "ilp": _calculate_ilp,
    "correlation": _calculate_correlation,
    "declare": _calculate_declare,
}

def calculate_model(log, pm, cluster_assignment, params=None):
    model_func = MODELS.get(pm)
    if model_func is None:
        raise Exception(f"Unknown process discovery model {pm}.")
    models = model_func(log, cluster_assignment, params)
    return models
