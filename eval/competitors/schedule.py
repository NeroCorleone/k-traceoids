from sklearn.cluster import KMeans
from entroclus import entropic_clustering
from actitrac import actitrac
from utils import get_encoding, retrieve_traces
import pm4py
import warnings
import pandas as pd
import numpy as np
import k_traceoids as ktr

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

    data = []
    for ix_, df_ in res.groupby(by=["case:concept:name", "label"]):
        ccn, label = ix_
        data.append((ccn, label, list(df_["concept:name"])))

    data = pd.DataFrame(data, columns=["trace_ids", "labels", "traces"])
    data["encoded_traces"] = np.nan
    return data


def run_actitrac(k, log):
    var_to_trace = ktr.data.get_variants(log)
    log = log.merge(var_to_trace)
    clusters = actitrac(log, k, tf=0.9, mcs=10, w=0.0, dist=None)
    data = []
    for label, cluster in enumerate(clusters):
        for trace in cluster:
            data.append((label, trace))

    data = pd.DataFrame(data, columns=["label", "traces"])
    data["encoded_traces"] = np.nan


    l = log.groupby(by=["case:concept:name", "variant"]).count().reset_index()
    l = l[["case:concept:name", "variant"]]
    data = data.merge(
        l,
        right_on="variant", 
        left_on="traces",
    )
    data = data.drop(columns="variant")
    data = data.rename(columns={"case:concept:name": "trace_id"})
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
    elif competitor_name == "actitrac":
        data = run_actitrac(k, log)
    else:
        data = run_other(k, log, competitor_name)
    return data


