import pm4py
import pandas as pd
import numpy as np
import concurrent.futures


CONFORMANCES = {
    "tbr": pm4py.conformance_diagnostics_token_based_replay,
    "al": pm4py.conformance_diagnostics_alignments,
    "al-declare": pm4py.conformance_declare,
}
CONFORMANCE_TIMEOUT = 2 * 60


def check_conformance(log, models, cc):
    # Calculate fitness on variant level as this saves time:
    # fitnes is the same for all traces of the same variant
    trace_to_variant = log[["@@case_index", "variant_id"]]

    vid_to_fitness = {}
    for vid, df_ in log.groupby("variant_id"):
        case_index = np.random.choice(df_["@@case_index"].values)
        log_select = df_[df_["@@case_index"] == case_index]
        fitnesses = []
        for model in models:
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        _calculate_fitness,
                        log_select,
                        model,
                        cc,
                    )
                    conformance_value = future.result(
                        timeout=CONFORMANCE_TIMEOUT,
                    )
            except concurrent.futures.TimeoutError:
                print("Timeout for conformance check")
                conformance_value = 0
            fitnesses.append(conformance_value)
        vid_to_fitness[vid] = fitnesses
    variants_fitness = pd.DataFrame(vid_to_fitness).T

    variants_fitness = variants_fitness.reset_index()
    variants_fitness = variants_fitness.rename(columns={"index": "variant_id"})
    fitness = variants_fitness.merge(trace_to_variant).sort_values(
        "@@case_index",
        ascending=True,
    )
    fitness = fitness.drop_duplicates(subset="@@case_index")
    fitness = fitness.drop(columns=["@@case_index", "variant_id"])
    fitness = fitness.reset_index(drop=True)
    return fitness


def _calculate_fitness(log, model, cc, f_col="fitness"):
    if cc == "al-declare":
        conformance_value = _calculate_fitness_declare(log, model)
    else:
        cc_func = CONFORMANCES[cc]
        net, initial_marking, final_marking = model
        # Start timining
        conformance = cc_func(log, net, initial_marking, final_marking)
        conformance = pd.DataFrame(conformance)
        # Different conformance check methods return different column names for fitnesss
        if f_col not in conformance.columns:
            f_col = "trace_fitness"
            assert (
                f_col in conformance.columns
            ), "Conformance check did not return a fitness column"
        conformance_value = conformance[f_col].values[0]
    return conformance_value


def _calculate_fitness_declare(log, model):
    conformance = pm4py.conformance_declare(log, model)
    conformance = pd.DataFrame(conformance)
    conformance_value = conformance["dev_fitness"].values[0]
    return conformance_value

