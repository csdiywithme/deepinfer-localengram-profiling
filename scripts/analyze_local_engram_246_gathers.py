"""Find LocalEngram gather activity in raw #246 EP4 FP8 Kineto replay traces."""

import argparse
from collections import Counter, defaultdict
from itertools import groupby
import json
from pathlib import Path
import re

from analyze_local_engram_246_trace import EXPECTED_STEPS, correlation, same_host_interval


FORWARD_SCOPE = re.compile(r"deepinfer:launch_forward_N[0-9]+_B[0-9]+\Z")
GRAPH_LAUNCH = re.compile(r"(?:cuda|cu)GraphLaunch(?:WithFlags)?(?:_ptsz|_ptds)?\Z")
HASH_KERNEL = re.compile(r"\bengram_hash_kernel_cuda(?:_kernel)?\b")
LAYER_ANCHOR = re.compile(r"\b(?:expand_to_mhc_fwd_cuda|mhc_pre_[A-Za-z0-9_]+_cuda)(?:_kernel)?\b")
INDEX_SELECT = re.compile(
    r"\b(?:indexSelect(?:Small|Large)Index|index_select(?:_cuda)?_kernel|"
    r"vectorized_gather_kernel|_scatter_gather_elementwise_kernel)\b"
)
EXPECTED_GATHERS = 2 * 2 * 4  # layers × (FP8 rows, packed scales) × peers
TOLERANCE_US = 0.1


def arg(event, name):
    return event.get("args", {}).get(name)


def end(event):
    return event["ts"] + event["dur"]


def lane(event):
    return tuple(arg(event, key) for key in ("device", "context", "stream"))


def node(event):
    return {
        "name": event["name"], "start_us": event["ts"], "duration_us": event["dur"],
        "node_id": arg(event, "graph node id"), "grid": arg(event, "grid"), "block": arg(event, "block"),
    }


def layer_split(gathers):
    """Infer the two source-order groups only when their launch signatures distinguish them."""
    def signature(event):
        return (event["name"], arg(event, "grid"), arg(event, "block"))

    groups = [(sig, list(events)) for sig, events in groupby(gathers, signature)]
    if [len(events) for _, events in groups] != [4, 4, 4, 4]:
        return {"status": "no_distinct_repeated_data_scale_signature"}
    first, second, third, fourth = [sig for sig, _ in groups]
    if first != third or second != fourth or first == second:
        return {"status": "no_distinct_repeated_data_scale_signature"}

    layers = []
    for layer_id, start in ((1, 0), (14, 2)):
        data, scales = groups[start][1], groups[start + 1][1]
        layers.append({
            "layer": layer_id,
            "data_node_ids": [arg(e, "graph node id") for e in data],
            "scale_node_ids": [arg(e, "graph node id") for e in scales],
            "first_gather_start_us": data[0]["ts"], "last_gather_end_us": end(scales[-1]),
            "gather_kernel_duration_sum_us": sum(e["dur"] for e in data + scales),
        })
    return {"status": "inferred_from_repeated_signatures_and_pinned_source_order", "layers": layers}


def inspect_step(step, host_scopes, runtimes, kernels_by_correlation):
    row = {"step": step["name"], "host_start_us": step["ts"], "index_select_nodes_observed": None}

    def unresolved(reason):
        return {**row, "status": reason}

    forward = [e for e in host_scopes if FORWARD_SCOPE.fullmatch(e["name"]) and same_host_interval(step, e)]
    if len(forward) != 1:
        return unresolved("unique_main_forward_scope_not_observed")
    calls = sorted((e for e in runtimes if same_host_interval(forward[0], e)), key=lambda e: (e["ts"], -e["dur"]))
    if not calls or any(not same_host_interval(calls[0], e) for e in calls[1:]):
        return unresolved("unique_main_graph_launch_not_observed")

    correlations = {correlation(e) for e in calls}
    kernels = sorted((e for key in correlations for e in kernels_by_correlation.get(key, ())), key=lambda e: e["ts"])
    graph_ids = {arg(e, "graph id") for e in kernels}
    if len(graph_ids) != 1 or next(iter(graph_ids)) in (None, 0, "0"):
        return unresolved("main_graph_id_missing_or_ambiguous")
    row.update(main_graph_id=next(iter(graph_ids)), main_graph_api_correlations=sorted(correlations),
               main_forward_scope=forward[0]["name"])

    hashes = [e for e in kernels if HASH_KERNEL.search(e["name"])]
    anchors = [e for e in kernels if LAYER_ANCHOR.search(e["name"])]
    if len(hashes) != 1 or not anchors:
        return unresolved("hash_or_early_layer_compute_anchor_missing_or_ambiguous")
    hashed, first_layer = hashes[0], anchors[0]
    if hashed["ts"] < calls[0]["ts"] - TOLERANCE_US or end(hashed) >= first_layer["ts"]:
        return unresolved("hash_to_layer_bracket_order_not_established")
    row.update(hash_kernel=node(hashed), early_layer_compute_anchor=node(first_layer))

    between = [e for e in kernels if e["ts"] < first_layer["ts"] and end(e) > end(hashed) and e is not hashed]
    gathers = [e for e in between if INDEX_SELECT.search(e["name"])]
    if not gathers:
        return unresolved("no_recognized_index_select_events_in_bracket")
    ordered = [hashed, *gathers, first_layer]
    kernel_lane = lane(hashed)
    if None in kernel_lane or any(lane(e) != kernel_lane for e in ordered[1:]):
        return unresolved("hash_gathers_and_layer_anchor_do_not_share_known_lane")
    if any(end(a) > b["ts"] + TOLERANCE_US for a, b in zip(ordered, ordered[1:])):
        return unresolved("hash_gather_layer_order_or_nonoverlap_not_established")
    ids = [arg(e, "graph node id") for e in ordered]
    if any(x in (None, 0, "0") for x in ids) or len(set(ids)) != len(ids):
        return unresolved("distinct_graph_node_ids_not_observed")

    row.update(
        status="expected_16_nodes_observed" if len(gathers) == EXPECTED_GATHERS else "observed_count_differs_from_16",
        lane=dict(zip(("device", "context", "stream"), kernel_lane)),
        index_select_nodes_observed=len(gathers),
        index_select_kernel_duration_sum_us=sum(e["dur"] for e in gathers),
        first_to_last_gather_span_us=end(gathers[-1]) - gathers[0]["ts"],
        hash_end_to_early_layer_anchor_start_us=first_layer["ts"] - end(hashed),
        gathers=[node(e) for e in gathers],
        layer_split=layer_split(gathers),
    )
    return row


def inspect_trace(trace):
    host_scopes, runtimes = [], []
    kernels_by_correlation = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or not all(k in event for k in ("name", "ts", "dur")):
            continue
        category = event.get("cat")
        key = correlation(event)
        if category == "user_annotation":
            host_scopes.append(event)
        elif key is not None:
            if category in ("cuda_runtime", "cuda_driver") and GRAPH_LAUNCH.fullmatch(event["name"]):
                runtimes.append(event)
            elif category == "kernel":
                kernels_by_correlation[key].append(event)
    steps = sorted((e for e in host_scopes if e["name"] in EXPECTED_STEPS), key=lambda e: e["ts"])
    rows = [inspect_step(step, host_scopes, runtimes, kernels_by_correlation) for step in steps]
    return {
        "exact_steady_step_labels": Counter(e["name"] for e in steps) == Counter(EXPECTED_STEPS),
        "status_counts": dict(Counter(row["status"] for row in rows)), "steps": rows,
    }


LIMITS = (
    "Expected count applies to the pinned #246 two-layer FP8 EP4 path. This is name-based identification "
    "of GPU kernel nodes in an observed main-graph bracket. A missing marker, graph ID, node ID, "
    "or unrecognized kernel symbol leaves evidence incomplete. The early mHC kernel anchors layer compute, "
    "not the exact first instruction of layer 0. A layer split is an inference from launch signatures "
    "and source order. Kernel durations omit unidentified routing, merge and layout operations. "
    "These observations do not measure peer bytes, hardware bandwidth, critical-path slack or speedup."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A raw Kineto JSON file, a trace directory, or its parent case directory")
    path = parser.parse_args().path
    files = [path]
    if path.is_dir():
        folder = path / "trace" if (path / "trace").is_dir() else path
        files = sorted(folder.glob("rank-*.kineto.json"))
        if not files:
            parser.error(f"no rank-*.kineto.json files in {folder}")
    print(json.dumps({"limits": LIMITS, "traces": [
        {"path": str(file), **inspect_trace(json.loads(file.read_text()))} for file in files
    ]}, indent=2))


if __name__ == "__main__":
    main()
