from sklearn.cluster import KMeans
from entroclus import entropic_clustering
from utils import get_encoding, retrieve_traces
import pm4py
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def cluster(data, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(list(data["encoded_traces"].values))
    data["labels"] = kmeans.labels_
    return data 


def run_entroclus(k, log):
    result = entropic_clustering.cluster(log, num_clusters=k)
    dfs = []
    for k, df_ in enumerate(result):
        df_.loc[:, "label"] = k
        dfs.append(df_)
    res = pd.concat(dfs)
    res = res.sort_index()

    trace_ids, traces = retrieve_traces(res)
    data = pd.DataFrame({
        "ix": list(range(len(traces))),
        "trace_ids": trace_ids,
        "traces": traces,
        "encoded_traces": np.nan,
    })
    return data

def run_other(k, log, competitor_name):
    encode = get_encoding(competitor_name)
    data = encode(log)
    data = cluster(data, k)
    return data


def run(k, ds_path, competitor_name):
    log = pm4py.read_xes(ds_path)
    if competitor_name == "entroclus":
        data = run_entroclus(k, log)
    else:
        data = run_other(k, log, competitor_name)
    return data


