import pandas as pd
import numpy as np
from utils import retrieve_traces

from gensim.models.doc2vec import Word2Vec

CONFIG = {
    "vector_size": 16,
    "aggregation": "average", 
}

def retrieve_encoding(model, traces, aggregation="average"):
    vectors = []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(model.wv[token])
            except KeyError:
                pass
        if aggregation == "average":
            vectors.append(np.array(trace_vector).mean(axis=0))
        elif aggregation == "max":
            vectors.append(np.array(trace_vector).max(axis=0))
        else:
            raise Exception(
                "Please select a valid aggregation method: {average, max}"
            )
    return vectors


def encode(log, config=CONFIG):
    trace_ids, traces = retrieve_traces(log)
    model = Word2Vec(vector_size=config["vector_size"], window=3, min_count=1, sg=0, workers=-1)
    model.build_vocab(traces)
    model.train(traces, total_examples=len(traces), epochs=10)
    encoded_traces = retrieve_encoding(model, traces, config["aggregation"])
    data = pd.DataFrame({
        "ix": list(range(len(traces))),
        "trace_ids": trace_ids,
        "traces": traces,
        "encoded_traces": encoded_traces,
    }) 
    return data