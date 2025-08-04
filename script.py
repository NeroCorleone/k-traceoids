import multiprocessing as mp
import os
import tk_means as tkm
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
    # "bpi2019-sample", "road_traffic", "sepsis",
    "bpi2019-sample"
]
pms = [
    # 'imf', "hm", "dfg",
    "imf"
]
ccs = [
    # "tbr", 
    "al",
]

ks = range(5, 6 ,1) # range(2, 11, 1)
max_iterations = 2 #10 # 100 
num_workers = int(mp.cpu_count())
result_dir = tkm.data.make_result_dir()

for ds in datasets:
    log = tkm.data.read_log(
        os.path.abspath(f"./datasets/{ds}.xes")
    )
    trace_to_variant = tkm.data.get_variants(log)
    log = log.merge(trace_to_variant)
    tkm.parallel.execute(log, ds, pms, ccs, ks, max_iterations, num_workers, result_dir)
