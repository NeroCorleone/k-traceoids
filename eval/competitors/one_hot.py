import numpy as np
import pandas as pd

def encode(log):
    categories = log["concept:name"].unique()
    event_to_index = {event: idx for idx, event in enumerate(categories)}
    traces = []
    encoded_traces = []
    trace_ids = []
    for tid, df_ in log.groupby("case:concept:name"):
        trace = df_["concept:name"].values
        ohv = np.zeros(len(categories))
        for event in trace:
            ohv[event_to_index[event]] += 1
        # Normalizing one hot encoded vector by trace length
        # Could also be the number of unique events in the trace?
        ohv = ohv / len(trace)
        encoded_traces.append(ohv)
        trace_ids.append(tid)
        traces.append(trace)

    data = pd.DataFrame({
        "ix": list(range(len(traces))),
        "trace_ids": trace_ids,
        "traces": traces,
        "encoded_traces": encoded_traces,
    })
    return data