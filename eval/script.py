import os
import k_traceoids as ktr
import logging

logging.basicConfig(
    filename="script.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a",
)

# Parameters
datasets = ["bpi2013-downsampled"]
pms = [
    "imf",
    # "hm",
    # "dfg",
]
ccs = [
    # "tbr",
    "al",
]

ks = range(2, 11, 1)
max_iterations = 100
num_workers = 20 

if __name__ == "__main__":
    for ds in datasets:
        result_dir = ktr.data.make_result_dir(ds)
        # Parallel execution for each hyperparamenter configuration with multiprocessing
        ktr.parallel.execute(ds, pms, ccs, ks, max_iterations, num_workers, result_dir)
