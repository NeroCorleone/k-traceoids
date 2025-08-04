import multiprocessing as mp
import os
import logging

import tk_means as tkm

_logger = logging.getLogger(__name__)

def execute(log, ds, pms, ccs, ks, max_iterations, num_workers, result_dir):
    _logger.info(f"Starting to execute for dataset {ds}...")
    tq = mp.Queue()
    rq = mp.Queue()

    # Fill task queue
    for pm in pms:
        for cc in ccs:
            for k in ks:
                params = [log, k, pm, cc, max_iterations, ds]
                tq.put(params)
    
    num_tasks = len(pms) * len(ccs) * len(ks)

    _logger.info("Starting to cluster...")

    # Start writer process
    writer_proc = mp.Process(target=writer, args=(rq, result_dir, num_tasks))
    writer_proc.start()

    # Start worker processes
    cluster_parallel(tq, rq, num_workers)

    # Wait for writer to finish
    writer_proc.join()

    _logger.info("Clustering and writing done.")

   

def worker(task_queue, result_queue):
    while True:
        params = task_queue.get()
        if params is None:
            break
        result = tkm.algorithm.tk_means(params)
        result_queue.put((params, result))



def cluster_parallel(tq, rq, num_workers):
    workers = []

    # Signal end of queue for workers
    for _ in range(num_workers):
        tq.put(None)

    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(tq, rq))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()


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


def write_result(params, result, result_dir):
    log, k, pm, cc, max_iterations, ds = params
    print(f"Writing results for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}")
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

def writer(result_queue, result_dir, num_tasks):
    for _ in range(num_tasks):
        params, result = result_queue.get()
        if result is None:
            continue
        write_result(params, result, result_dir)
