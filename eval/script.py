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
num_workers = 20 # 5 # int(mp.cpu_count() / 2)
result_dir = ktr.data.make_result_dir()

if __name__ == "__main__":
    import multiprocessing as mp
    for ds in datasets:
        log = ktr.data.prepare_log(os.path.abspath(f"./datasets/{ds}.xes"))
        # Parallel execution for each hyperparamenter configuration with multiprocessing
        ktr.parallel.execute(log, ds, pms, ccs, ks, max_iterations, num_workers, result_dir)
