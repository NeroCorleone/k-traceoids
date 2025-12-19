import time
import traceback
import warnings
import numpy as np
import k_traceoids as ktr

# annoying pandas warnings from pm4py, silence them
warnings.filterwarnings("ignore", category=Warning, module="pm4py")
warnings.filterwarnings("ignore", category=Warning, module="pd")
warnings.filterwarnings("ignore", category=Warning, module="pandas")


# TODO Split this up into A) Core clustering logic and B) Experimental result, times, fitnesses, etc...
def cluster(params):
    k, pm, cc, max_iterations, ds = params
    print(
        f"Starting k-traceoids for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}",
    )
    log = ktr.data.prepare_log(f"./datasets/{ds}.xes")

    iteration = 0
    cluster_assignment = ktr.initialization.initialize_clusters(log, k)
    all_models = []
    all_fitness = []
    all_times = []
    while True:
        try:
            start_time = time.time()

            print(
                f"Discovering process models for {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}",
            )
            models = ktr.models.calculate_model(log, pm, cluster_assignment)
            print(
                f"Checking conformance for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}",
            )
            fitness = ktr.conformance.check_conformance(log, models, cc)
            ca_col = f"cluster_assignment_{iteration}"

            print(
                f"Reassining clusters for data set {ds} with k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}",
            )
            cluster_assignment[ca_col] = ktr.reassign.reassign_clusters(
                fitness,
                cluster_assignment,
                k,
            )
            all_models.append(models)
            all_fitness.append(fitness)

            end_time = time.time()
            execution_seconds = end_time - start_time
            all_times.append(execution_seconds)

            if check_convergence(cluster_assignment, iteration, max_iterations):
                print(
                    f"K-traceoids run complete with parameters k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}",
                )
                # df, list[df], list[models]
                return (cluster_assignment, all_fitness, all_models, all_times)
            iteration += 1
        except Exception:
            print(
                f"K-traceoids run failed with parameters k={k}, pm={pm}, cc={cc} and max_iter={max_iterations}",
            )
            print(traceback.format_exc())
            return None


def check_convergence(cluster_assignment, iteration, max_iterations):
    # Reached maximum number of iterations
    print(
        f"Checking conformance for iteration {iteration} of {max_iterations}....",
    )
    if iteration >= max_iterations:
        return True
    # Check if the last two columns are the same
    before_last_col, last_col = cluster_assignment.columns[-2:]
    cond = cluster_assignment[before_last_col].equals(
        cluster_assignment[last_col],
    )
    if cond:
        return True
    return False
