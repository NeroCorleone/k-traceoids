import pandas as pd

from utils import retrieve_traces
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

CONFIG = {
    "vector_size": 16,
    "aggregation": "average", 
}

def encode(log, config=CONFIG):
    trace_ids, traces = retrieve_traces(log)
    tagged_traces = [TaggedDocument(words=act, tags=[str(i)]) for i, act in enumerate(traces)]
    model = Doc2Vec(vector_size=config["vector_size"], min_count=1, window=3, dm=1, workers=-1)
    model.build_vocab(tagged_traces)
    encoded_traces = [model.infer_vector(trace) for trace in traces]
    data = pd.DataFrame({
        "ix": list(range(len(traces))),
        "trace_ids": trace_ids,
        "traces": traces,
        "encoded_traces": encoded_traces,
    }) 
    return data