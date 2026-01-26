import pm4py
import numpy as np
import random
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.variants.imf import IMFUVCL
from pm4py.util.compression.dtypes import UVCL
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.obj import Operator
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from itertools import combinations
import pandas as pd
from datetime import datetime
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator

PARAMS_IMF = {"noise_threshold": 0.2}


def _measure_overlaps(to_compare):
    overlaps = [] 
    for s1, s2 in combinations(to_compare, 2):
        overlaps.append(len(s1.intersection(s2)) / len(s1.union(s2)))
    return overlaps


def _convert_to_string_set(traces):
    string_set = set()
    for t in traces:
        string_trace = "".join(s for s in t)
        string_set.add(string_trace)
    return string_set

def _generate_traces_from_model(m):
    net, im, fm = m
    simulated_log = pm4py.algo.simulation.playout.petri_net.algorithm.apply(net, im, fm)

    traces = []
    for log_entry in simulated_log:
        trace = [event["concept:name"] for event in log_entry._list]
        traces.append(trace)

    return traces


def collect_nodes(node):
    nodes = [node]
    for c in node.children:
        nodes.extend(collect_nodes(c))
    return nodes


def load_tree(log):
    variants = pm4py.get_variants(log)
    uvcl = UVCL()

    for var, occ in variants.items():
        uvcl[var] = occ

    imfuvcl = IMFUVCL(PARAMS_IMF)
    tree = imfuvcl.apply(IMDataStructureUVCL(uvcl), parameters=PARAMS_IMF)
    return tree

def get_all_nodes(tree):
    nodes = []

    def recurse(node):
        nodes.append(node)
        for c in node.children:
            recurse(c)

    recurse(tree)
    return nodes


# all nodes
def get_all_nodes(tree):
    nodes = []

    def recurse(node):
        nodes.append(node)
        for child in node.children:
            recurse(child)

    recurse(tree)
    return nodes


# just the leaves
def get_leaves(tree):
    leaves = []
    def recurse(node):
        if not node.children:
            if node.label is not None:
                # Empty label
                leaves.append(node)
        else:
            for child in node.children:
                recurse(child)
    recurse(tree)
    return leaves


def clone_tree(node):
    new_node = ProcessTree(
        operator=node.operator,
        label=node.label
    )
    for child in node.children:
        child_clone = clone_tree(child)
        child_clone.parent = new_node
        new_node.children.append(child_clone)
    return new_node

# M1: swap two random activities
def swap_two_random_activities(tree, seed):
    # Swap two random activity leaf nodes 
    if seed is not None:
        random.seed(seed)

    leaves = get_leaves(tree)

    if len(leaves) < 2:
        raise ValueError("Process tree must contain at least two activities to swap.")

    # leave selection
    node_a, node_b = random.sample(leaves, 2)
    print(f"Swapping activities: {node_a.label} <--> {node_b.label}")

    parent_a = node_a.parent
    parent_b = node_b.parent

    if parent_a is None or parent_b is None:
        raise ValueError("Cannot swap the root node.")

    idx_a = parent_a.children.index(node_a)
    idx_b = parent_b.children.index(node_b)

    # Swap: node_b --> parent_a, node_a --> parent_b
    parent_a.children[idx_a], parent_b.children[idx_b] = node_b, node_a
    node_a.parent, node_b.parent = parent_b, parent_a

    return tree


# M2: swap two random activity labels
def swap_two_random_activity_labels(tree, seed):
    # Similar as above, but just swap the labels
    if seed is not None:
        random.seed(seed)

    leaves = get_leaves(tree)

    if len(leaves) < 2:
        raise ValueError("Process tree must contain at least two activities to swap labels.")

    # Pick two distinct activity nodes
    node_a, node_b = random.sample(leaves, 2)
    print(f"Swapping activity labels: {node_a.label} <--> {node_b.label}")

    # Swap labels only
    node_a.label, node_b.label = node_b.label, node_a.label

    return tree

# M3: Insert one additional random activity in a sequence
def insert_random_activity_sequence(tree, activity_label, seed):
    # Insert new activity with defined label at a random position
    # For simplicity: add it into a sequence somwhere
    if seed is not None:
        random.seed(seed)

    leaves = get_leaves(tree)

    if not leaves:
        raise ValueError("Process tree contains no activities to insert into.")

    # Pick random activity node and create a new activity with given label
    target = random.choice(leaves)
    parent = target.parent
    new_activity = ProcessTree(label=activity_label)

    # needed to nest the activity somewhere, for simplicity: use sequence
    # and randomly decide order
    seq = ProcessTree(operator=Operator.SEQUENCE)
    if random.random() < 0.5:
        seq.children = [new_activity, target]
    else:
        seq.children = [target, new_activity]

    # Set parantes and attach seq to original parent
    new_activity.parent = seq
    target.parent = seq

    if parent is None:
        # if target was root --> then new sequence becomes root
        seq.parent = None
        return seq
    else:
        idx = parent.children.index(target)
        parent.children[idx] = seq
        seq.parent = parent
        return tree


# M4: Add a loop somewhere
def add_random_loop(tree, seed=None):
    if seed is not None:
        random.seed(seed)
    tree = clone_tree(tree)

    candidates = get_all_nodes(tree)

    candidates = [n for n in candidates if n.parent is not None]

    if not candidates:
        return tree

    target = random.choice(candidates)
    parent = target.parent

    # Create redo branch 
    redo = ProcessTree(label=None)

    # Create loop node
    loop_node = ProcessTree(operator=Operator.LOOP)
    loop_node.children = [target, redo]

    # Fix parents
    loop_node.parent = parent
    target.parent = loop_node
    redo.parent = loop_node

    # Replace target with loop node in parent
    idx = parent.children.index(target)
    parent.children[idx] = loop_node

    return tree

def change_random_seq_to_xor(tree):
    seq_nodes = [
        n for n in get_all_nodes(tree)
        if n.operator == Operator.SEQUENCE and len(n.children) >= 2
    ]

    if not seq_nodes:
        return tree  # nothing to change

    target = random.choice(seq_nodes)
    target.operator = Operator.XOR

    return tree


# M5: Change SEQ to Parallel
def change_random_seq_to_parallel(tree):
    seq_nodes = [
        n for n in get_all_nodes(tree)
        if n.operator == Operator.SEQUENCE and len(n.children) >= 2
    ]

    if not seq_nodes:
        return tree

    target = random.choice(seq_nodes)
    target.operator = Operator.PARALLEL

    return tree


def insert_optional_activity(tree, activity_name, seed=None):
    # Insert an optional activity with XOR at random position in the tree
    if seed is not None:
        random.seed(seed)

    all_nodes = collect_nodes(tree)

    # chose where to add the activity
    target = random.choice(all_nodes)
    parent = target.parent

    activity_node = ProcessTree(label=activity_name)
    tau_node = ProcessTree(label=None)

    xor_node = ProcessTree(operator=Operator.XOR)
    xor_node.children = [activity_node, tau_node]
    activity_node.parent = xor_node
    tau_node.parent = xor_node

    seq_node = ProcessTree(operator=Operator.SEQUENCE)
    seq_node.children = [target, xor_node]

    target.parent = seq_node
    xor_node.parent = seq_node

    if parent is None:
        return seq_node
    else:
        idx = parent.children.index(target)
        parent.children[idx] = seq_node
        seq_node.parent = parent
        return tree

def eventlog_to_xes_dataframe(log):
    rows = []

    for trace in log:
        case_id = trace.attributes.get("concept:name", "UNKNOWN_CASE")

        trace_attrs = {
            f"case:{k}": v
            for k, v in trace.attributes.items()
            if k != "concept:name"
        }

        for event in trace:
            row = {}

            row["case:concept:name"] = case_id
            row["concept:name"] = event.get("concept:name")

            ts = event.get("time:timestamp")
            if isinstance(ts, datetime):
                row["time:timestamp"] = ts
            else:
                row["time:timestamp"] = pd.to_datetime(ts)

            for attr, value in event.items():
                if attr not in {"concept:name", "time:timestamp"}:
                    row[attr] = value

            row.update(trace_attrs)
            rows.append(row)

    return pd.DataFrame(rows)


def simulate_log(base_log, n_traces, var_to_trace_ratio):
    variants = list(pm4py.get_variants(base_log).keys())
    # -----------------------
    # 1. determine target number of variants
    # -----------------------
    n_vars = int(n_traces * var_to_trace_ratio)
    n_vars = min(n_vars, len(variants))  # safety

    # -----------------------
    # 2. select variants
    # -----------------------
    idx_variants = np.random.choice(
        np.arange(len(variants)),
        size=n_vars,
        replace=False
    )
    selected_variants = [variants[i] for i in idx_variants]

    # -----------------------
    # 3. sample how many traces per variant
    # -----------------------
    sample = np.random.random(size=n_vars)
    sample_norm = sample / sample.sum()
    variant_sample = np.round(sample_norm * n_traces).astype(int)

    # fix rounding issues
    diff = n_traces - variant_sample.sum()
    variant_sample[0] += diff

    # -----------------------
    # 4. build event log as DataFrame
    # -----------------------
    rows = []
    case_id = 0

    for variant, freq in zip(selected_variants, variant_sample):
        for _ in range(freq):
            case_id += 1
            timestamp = pd.Timestamp("2024-01-01")

            for activity in variant:
                rows.append({
                    "case:concept:name": f"{case_id}",
                    "concept:name": activity,
                    "time:timestamp": timestamp
                })
                timestamp += pd.Timedelta(minutes=1)

    df = pd.DataFrame(rows)
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    return df

def create_synthetic_log(modified_trees, n_traces, var_to_trace_ratio, max_trace_length):
    parameters = {
        simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: n_traces,  # total nb traces
        simulator.Variants.BASIC_PLAYOUT.value.Parameters.MAX_TRACE_LENGTH: max_trace_length,  # max trace length
    }
    simulations = []

    for k, mt in enumerate(modified_trees):
        # Step 1: create a simulated log based on the model
        m = pm4py.convert_to_petri_net(mt)
        net, im, fm = m
        playout_log = pm4py.algo.simulation.playout.petri_net.algorithm.apply(net, im, fm, parameters=parameters)

        # Step 2: the playout is highly variable in the number of variants
        # --> sample this one up in a simulation
        simulated_log  = simulate_log(playout_log, n_traces, var_to_trace_ratio)
        simulated_log["label"] = k 
        # Update trace id to be unique
        simulated_log["case:concept:name"] = simulated_log["case:concept:name"] + "-" + simulated_log["label"].astype(str)
        simulations.append(simulated_log)

    synthetic_log = pd.concat(simulations)
    return synthetic_log

def create_modified_models(tree, k):
    seed = None
    modified_trees = []

    for i in range(k):
        tree_mod = clone_tree(tree)
        tree_mod = swap_two_random_activities(tree_mod, seed)
        tree_mod = insert_random_activity_sequence(activity_label="Added Activity 1", tree=tree_mod, seed=seed)
        tree_mod = insert_optional_activity(activity_name="Added Activity 2", tree=tree_mod, seed=seed)
        modified_trees.append(tree_mod)
    return modified_trees


def create_string_traces_to_compare(modified_trees):
    # Create a list of "stringified" traces for each modified model
    # All lists will then be compared pairwise for overlap

    to_compare = []
    for mt in modified_trees:
        # convert to model
        m = pm4py.convert_to_petri_net(mt)
        gt = []
        for i in range(10):
            gt.extend(_generate_traces_from_model(m))
        string_traces = _convert_to_string_set(gt)
        to_compare.append(string_traces)
    return to_compare