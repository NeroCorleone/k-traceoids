import multiprocessing as mp
import os
import k_traceoids as ktr
import logging

logging.basicConfig(
    filename='script.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='a',
)

# Parameters
datasets = [
    "bpi2019-sample", "road_traffic", "sepsis",
]
pms = [
    'imf', "hm", "dfg",
]
ccs = [
    "tbr", "al",
]

ks = range(2, 11, 1)
max_iterations = 100 
num_workers = int(mp.cpu_count())
result_dir = ktr.data.make_result_dir()

for ds in datasets:
    log = ktr.data.read_log(
        os.path.abspath(f"./datasets/{ds}.xes")
    )
    trace_to_variant = ktr.data.get_variants(log)
    log = log.merge(trace_to_variant)
    ktr.parallel.execute(log, ds, pms, ccs, ks, max_iterations, num_workers, result_dir)
