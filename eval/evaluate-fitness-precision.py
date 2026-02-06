from joblib import Parallel, delayed
import os
import pandas as pd
import pm4py

from pm4py.algo.discovery.inductive.variants.imf import IMFUVCL
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.util.compression.dtypes import UVCL
from pm4py.algo.evaluation.precision.algorithm import apply as precision_apply


COMP_RES_DIR = "./competitors/results"  
KTR_RES_DIR = "./results"

EVAL_RES_DIR = "./evaluation-results/" # This is where model and fitness will be stored in
if not os.path.exists(EVAL_RES_DIR):
    os.mkdir(EVAL_RES_DIR)

KS = range(2, 9, 1)
DATASETS = ["hospital_billing", "road_traffic", "sepsis"]
COMP = [
    "actitrac",
    "entroclus",
    "doc2vec",
    "one_hot",
    "threegram",
    "word2vec",
]

PARAMS_IMF = {"noise_threshold": 0.2}

def _discover_model(log):
    variants = pm4py.get_variants(log)
    uvcl = UVCL()

    for var, occ in variants.items():
        uvcl[var] = occ

    imfuvcl = IMFUVCL(PARAMS_IMF)
    tree = imfuvcl.apply(IMDataStructureUVCL(uvcl), parameters=PARAMS_IMF)
    model = pm4py.convert_to_petri_net(tree)
    return model


def _create_evaluation_dir(competitor, dataset, k, runtime):
    eval_dir = os.path.join(
        EVAL_RES_DIR,
        f"competitor={competitor}",
        f"dataset={dataset}",
        f"k={k}",
        f"runtime={runtime}",
    )
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def process_ktr(dataset, k, runtime, log):
    try:
        data_dir = os.path.join(KTR_RES_DIR, dataset)
        sub_dir = f"{runtime}/{dataset}/pm_imf/cc_al/k_{k}/max_iter_100"
        base_dir = os.path.join(data_dir, sub_dir)

        # fitness
        fitness_dir = os.path.join(base_dir, "fitness")
        last_iter = max(
            os.listdir(fitness_dir),
            key=lambda s: int(s.rsplit("_", 1)[1])
        )

        fitness = pd.read_csv(
            os.path.join(fitness_dir, last_iter, "fitness.csv")
        )

        eval_dir = _create_evaluation_dir("k-traceoids", dataset, k, runtime)
        fitness.to_csv(os.path.join(eval_dir, "fitness.csv"), index=False)

        # models
        model_dir = os.path.join(base_dir, "models", last_iter)
        model_precision = []
        for i, m in enumerate(os.listdir(model_dir)):
            if m.endswith(".pnml"):
                net, im, fm = pm4py.read_pnml(os.path.join(model_dir, m))
                pm4py.write_pnml(
                    net, im, fm,
                    os.path.join(eval_dir, f"model-{i}.pnml")
                )
                model_precision.append(
                    precision_apply(log, net, im, fm)
                )
        return ("k-traceoids", dataset, k, runtime, model_precision)


    except Exception as e:
        print(f"[KTR ERROR] {dataset=} {k=} {runtime=} → {e}")


def process_competitor(dataset, runtime, competitor, k, log):
    try:
        sub_dir = f"{COMP_RES_DIR}/{dataset}/{runtime}/{competitor}/k={k}"
        ca = pd.read_csv(os.path.join(sub_dir, "clustering.csv"))

        if "trace_id" not in ca.columns:
            ca = ca.rename(columns={"trace_ids": "trace_id"})
        if "label" not in ca.columns:
            ca = ca.rename(columns={"labels": "label"})

        eval_dir = _create_evaluation_dir(competitor, dataset, k, runtime)

        log_ca = log.merge(
            ca,
            left_on="case:concept:name",
            right_on="trace_id"
        )

        models = []
        for _, log_ in log_ca.groupby("label"):
            models.append(_discover_model(log_))

        fitness_model = {}
        model_precision = []
        for i, (net, im, fm) in enumerate(models):
            alignment = pm4py.conformance_diagnostics_alignments(log, net, im, fm)
            alignment = pd.DataFrame(alignment)
            fitness_model[i] = alignment["fitness"]

            pm4py.write_pnml(
                net, im, fm,
                os.path.join(eval_dir, f"model-{i}.pnml")
            )
            for net, im, fm in models:
                model_precision.append(
                    precision_apply(log, net, im, fm)
                )

        pd.DataFrame(fitness_model).to_csv(
            os.path.join(eval_dir, "fitness.csv"),
            index=False
        )
        return (competitor, dataset, k, runtime, model_precision)

    except Exception as e:
        print(f"[COMP ERROR] {dataset=} {runtime=} {competitor=} {k=} → {e}")


#### Main script

### 1. KTR

ktr_jobs = []

for ds in DATASETS:
    log = pm4py.read_xes(f"./datasets/{ds}.xes")
    log = pm4py.format_dataframe(log)
    log = log[['case:concept:name', 'concept:name', 'time:timestamp', '@@case_index']]
    data_dir = os.path.join(KTR_RES_DIR, ds)
    for k in KS:
        for runtime in os.listdir(data_dir):
            ktr_jobs.append((ds, k, runtime, log))

result = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_ktr)(ds, k, runtime, log)
    for ds, k, runtime, log in ktr_jobs
)

df = pd.DataFrame(result, columns=["competitor", "dataset", "k", "runtime", "model_precisions"])

### 2. COMPETITORS

comp_jobs = []

for ds in DATASETS:
    log = pm4py.read_xes(f"./datasets/{ds}.xes")
    log = pm4py.format_dataframe(log)
    log = log[['case:concept:name', 'concept:name', 'time:timestamp', '@@case_index']]

    data_dir = os.path.join(COMP_RES_DIR, ds)
    for runtime in os.listdir(data_dir):
        for c in COMP:
            for k in KS:  # TODO remove
                comp_jobs.append((ds, runtime, c, k, log))

result_competitor = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_competitor)(ds, rt, c, k, log)
    for ds, rt, c, k, log in comp_jobs
)

df_competitor = pd.DataFrame(result_competitor, columns=["competitor", "dataset", "k", "runtime", "model_precisions"])

### 3. STORE RESULTS

df_all = pd.concat([df, df_competitor])
df_all.to_csv(os.path.join(f"{EVAL_RES_DIR}", "precisions.csv"))