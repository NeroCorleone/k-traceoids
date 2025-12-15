import os

def get_encoding(competitor_name):
    if competitor_name == "one_hot":
        from one_hot import encode
    elif competitor_name == "threegram":
        from threegram import encode 
    elif competitor_name == "doc2vec":
        from doc2vec import encode
    elif competitor_name == "word2vec":
        from word2vec import encode
    else:
        raise ValueError(f"Unknown competitor: {competitor_name}")
    return encode


def store_results(results, result_dir):
    for c, k, clustering, execution_seconds in results:
        # c is competitor
        # Write result for each run of k
        k_dir = os.path.join(result_dir, f"{c}/k={k}")
        if not os.path.exists(k_dir):
            os.makedirs(k_dir)
        clustering.to_csv(os.path.join(k_dir, "clustering.csv"), index=False)
        with open(os.path.join(k_dir, "execution_time.txt"), "w") as f:
            f.write(f"{execution_seconds}\n")

def retrieve_traces(log):
    traces, ids = [], []
    for id in log["case:concept:name"].unique():
        events = list(log[log["case:concept:name"] == id]["concept:name"])
        traces.append(["".join(x) for x in events])
        ids.append(id)
    return ids, traces

