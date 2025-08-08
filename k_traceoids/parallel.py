import multiprocessing as mp
import os
import logging

import k_traceoids as ktr

_logger = logging.getLogger(__name__)


def execute(log, ds, pms, ccs, ks, max_iterations, num_workers, result_dir):
    _logger.info(f"Starting to execute for dataset {ds}...")
    tq = mp.Queue()
    rq = mp.Queue()

    # Fill the task queue
    for pm in pms:
        for cc in ccs:
            for k in ks:
                params = [log, k, pm, cc, max_iterations, ds]
                tq.put(params)
    
    num_tasks = len(pms) * len(ccs) * len(ks)

    _logger.info("Starting to cluster...")

    # Start the writer processes
    writers = []
    for _ in range(num_workers):
        writer_proc = mp.Process(target=writer, args=(rq, result_dir, num_tasks))
        writer_proc.start()
        writers.append(writer_proc)

    # Add break condition for workers
    for _ in range(num_workers):
        tq.put(None)

    # Start the workers: clustering
    workers = []
    for _ in range(num_workers):
        worker_proc = mp.Process(target=worker, args=(tq, rq))
        worker_proc.start()
        workers.append(worker_proc)

    # Wait for workers to finish
    for p in workers:
        p.join()
    
    # Add break condition for writers
    for _ in range(num_workers):
        rq.put(None)

    # Wait for writers
    for p in writers:
        p.join()

    _logger.info("Clustering and writing done.")


def worker(task_queue, result_queue):
    while True:
        params = task_queue.get()
        if params is None:
            break
        result = ktr.algorithm.tk_means(params)
        result_queue.put((params, result))


def writer(result_queue, result_dir, num_tasks):
    for _ in range(num_tasks):
        res = result_queue.get()
        if res is None:
            break
        params, result = res
        if result is None:
            continue
        write_result(params, result, result_dir)


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

    ktr.data.store_time(run_dir, all_times)

    for iteration in range(len(all_fitness)):
        models = all_models[iteration]
        fitness = all_fitness[iteration]
        ktr.data.store_intermediate_results(models, fitness, iteration, run_dir)
