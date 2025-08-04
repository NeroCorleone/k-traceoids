import pandas as pd
import os
import pm4py
import datetime

import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def store_time(result_dir, all_times):
    df = pd.DataFrame({"execution-times-seconds": all_times})
    time_file = os.path.join(result_dir, "execution-times.csv")
    df.to_csv(time_file)


def store_intermediate_results(models, fitness, iteration, result_dir):
    model_dir = os.path.join(result_dir, f"models/iteration_{iteration}")
    fitness_dir = os.path.join(result_dir, f"fitness/iteration_{iteration}")

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(fitness_dir):
        os.makedirs(fitness_dir)

    # Store fitness results
    fitness_file = os.path.join(
        fitness_dir,
        f"fitness.csv"
    )
    fitness.to_csv(fitness_file)

    for model_nb, model in enumerate(models):
        # Store model
        m_file = os.path.join(
            model_dir,
            f"model_{model_nb}.json"  # Model nb is the cluster nb
        )
        pnet, im, fm = model
        pm4py.write_pnml(pnet, im, fm, m_file)

        # Plot models
        gviz = pn_visualizer.apply(pnet, im, fm)
        m_plot = os.path.join(
            model_dir,
            f"model_plot{model_nb}.png"
        )
        pn_visualizer.save(gviz, m_plot)

def make_result_dir():
    runtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    result_dir = os.path.abspath(f"results/{runtime}")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    return result_dir


def read_log(event_file):
    df = pm4py.read_xes(event_file)
    log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name", 
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    return log


def get_variants(log):
    trace_to_variant = []
    vid = 0
    for v, traces in pm4py.stats.split_by_process_variant(log):
        for tid in traces["case:concept:name"].unique():
            trace_to_variant.append([tid, vid, v])
        vid += 1
    
    trace_to_variant = pd.DataFrame(
        trace_to_variant,
        columns=["case:concept:name", "variant_id", "variant"]
    )
    return trace_to_variant

