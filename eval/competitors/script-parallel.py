import os
import time
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from schedule import run
from utils import store_results


ks = range(2, 9, 1)

competitors = [
    "actitrac",
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_single_experiment(ds_name, run_timestamp, competitor, k):
    ds_path = f"../datasets/{ds_name}.xes"

    result_dir = (
        f"./results/{ds_name}/{run_timestamp}/"
        f"{competitor}/k={k}"
    )
    ensure_dir(result_dir)

    print(f"[START] {ds_name} | {competitor} | k={k}")
    start_time = time.time()

    result = run(k, ds_path, competitor)

    execution_seconds = time.time() - start_time
    print(
        f"[DONE]  {ds_name} | {competitor} | k={k} "
        f"in {execution_seconds:.2f}s"
    )

    store_results(
        [(competitor, k, result, execution_seconds)],
        result_dir,
    )

    return ds_name, competitor, k, execution_seconds


def main():
    jobs = []

    with ProcessPoolExecutor() as executor:
        for ds_name in datasets:
            run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")

            base_dir = f"./results/{ds_name}/{run_timestamp}"
            ensure_dir(base_dir)

            for competitor in competitors:
                for k in ks:
                    jobs.append(
                        executor.submit(
                            run_single_experiment,
                            ds_name,
                            run_timestamp,
                            competitor,
                            k,
                        )
                    )

        for future in as_completed(jobs):
            ds_name, competitor, k, seconds = future.result()
            print(
                f"[RESULT] {ds_name} | {competitor} | k={k} "
                f"took {seconds:.2f}s"
            )


if __name__ == "__main__":
    main()
