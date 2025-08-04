import multiprocessing as mp
import os
import logging

import tk_means as tkm

_logger = logging.getLogger(__name__)

def execute(log, ds, pms, ccs, ks, max_iterations, num_workers, result_dir):
    _logger.info(f"Starting to execute for dataset {ds}...")
    tq = mp.Queue()
    rq = mp.Queue()
    
    _logger.info("About to setup task queue...")
    for pm in pms:
        for cc in ccs:
            for k in ks:
                params = [log, k, pm, cc, max_iterations, ds]
                tq.put(params)
    _logger.info("Task queue set up.")
    
    num_tasks = len(pms) * len(ccs) * len(ks)

    _logger.info("Starting to cluster...")
    results = cluster_parallel(tq, rq, num_workers, num_tasks)
    _logger.info("Clustering done.")

    _logger.info("Starting to write results...")
    write_results(results, result_dir)
    _logger.info("Writing results done.")

    

def worker(task_queue, result_queue):
    while True:
        params = task_queue.get()
        if params is None:
            break
        result = tkm.algorithm.tk_means(params)
        result_queue.put((params, result))



def cluster_parallel(tq, rq, num_workers, num_tasks):
    results = []
    workers = []

    for _ in range(num_workers):
        tq.put(None)

    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(tq, rq))
        p.start()
        workers.append(p)
    
    for _ in range(num_tasks):
        results.append(rq.get())

    for p in workers:
        p.join()

    return results


def write_results(results, result_dir):
    for params, result in results:
        log, k, pm, cc, max_iterations, ds = params
        if result is None:
            continue
        run_dir = os.path.join(
            result_dir,
            f"{ds}/pm_{pm}/cc_{cc}/k_{k}/max_iter_{max_iterations}",
        )
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        cluster_assignment, all_fitness, all_models, all_times = result 
        cluster_assignment.to_csv(os.path.join(run_dir, "ca.csv"))

        tkm.data.store_time(run_dir, all_times)

        for iteration in range(len(all_fitness)):
            models = all_models[iteration]
            fitness = all_fitness[iteration]
            tkm.data.store_intermediate_results(models, fitness, iteration, run_dir)

