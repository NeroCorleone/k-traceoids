import os
import time
import datetime
from schedule import run
from utils import store_results

ks = range(2, 9, 1)

competitors = [
    "actitrac"
    "entroclus",
    "one_hot",
    "threegram",
    "doc2vec",
    "word2vec",
]

datasets = [
    "hospital_billing",
    "road_traffic",
    "sepsis",
]

for ds_name in datasets:
    ds_path = f"../datasets/{ds_name}.xes"
    run_timestamp = datetime.datetime.now()
    run_timestamp = run_timestamp.strftime("%Y%m%d-%H:%M:%S")

    result_dir = f"./results/{ds_name}/{run_timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    results = []
    for c in competitors:
        print(f"About to run competitor: {c} for dataset {ds_name}...")
        for k in ks:
            print(f"Running k={k} for competitor {c}...")
            start_time = time.time()
            result = run(k, ds_path, c)
            execution_seconds = time.time() - start_time
            results.append((c, k, result, execution_seconds)) 
            print(f"    Done k={k} for competitor {c} in {execution_seconds} seconds.")
        print(f"About to store results for competitor {c} and dataset {ds_name}.")
        store_results(results, result_dir)
        print(f"Stored results for competitor {c} and dataset {ds_name}.")
