import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.variants.imf import IMFUVCL
from pm4py.util.compression.dtypes import UVCL


# TODO check those parameters?
PARAMS_HM = {
    heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99,
}
# Removing infrequencies, e.g. edges between infrequent activities are removed
PARAMS_IMF = {"noise_threshold": 0.2}
PARAMS_DFG = {"noise_threshold": 0.2}


def _calculate_dfg(log, cluster_assignment):
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
                noise_threshold=PARAMS_DFG["noise_threshold"],
            )
        )
        model = pm4py.convert_to_petri_net(dfg_cleaned, sa, ea)
        models.append(model)
    return models


def _calculate_imf(log, cluster_assignment):
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


def _calculate_hm(log, cluster_assignment):
    # Discover pm with heuristic miner algorithm
    ca_col = cluster_assignment.columns[-1]  # Latest cluster assignment column

    models = []
    for _, df_ in cluster_assignment.groupby(ca_col):
        traces = log[log["@@case_index"].isin(df_["case_index"])]
        models.append(
            heuristics_miner.apply(traces, parameters=PARAMS_HM),
        )
    return models


MODELS = {
    "imf": _calculate_imf,
    "hm": _calculate_hm,
    "dfg": _calculate_dfg,
}


def calculate_model(log, pm, cluster_assignment):
    model_func = MODELS.get(pm)
    if model_func is None:
        raise Exception(f"Unknown process discovery model {pm}")
    models = model_func(log, cluster_assignment)
    return models
