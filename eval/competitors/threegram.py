import nltk
from collections import Counter
import numpy as np
import pandas as pd
import os


def extract_trigrams(trace, pad_token="<PAD>"):
    trigrams = nltk.ngrams(trace, 3, pad_right=True, right_pad_symbol=pad_token)
    return trigrams


def encode_trace(trace, trigram_vocab, pad_token="<PAD>"):
    trigrams = extract_trigrams(trace, pad_token)
    trigram_counts = Counter(trigrams)
    feature_vector = np.zeros(len(trigram_vocab))
    for i, trigram in enumerate(trigram_vocab):
        feature_vector[i] = trigram_counts.get(trigram, 0)
    return feature_vector


def encode(log):
    pad_token = "<PAD>"
    trace_ids = []
    traces = []
    for tid, df_ in log.groupby("case:concept:name"):
        trace = df_["concept:name"].values
        traces.append(trace)
        trace_ids.append(tid)
    

    all_trigrams = [trigram for trace in traces for trigram in extract_trigrams(trace, pad_token)]
    trigram_vocab = sorted(set(all_trigrams))
    encoded_traces = [encode_trace(trace, trigram_vocab, pad_token) for trace in traces]

    data = pd.DataFrame({
        "ix": list(range(len(traces))),
        "trace_ids": trace_ids,
        "traces": traces,
        "encoded_traces": encoded_traces
    })
    return data


