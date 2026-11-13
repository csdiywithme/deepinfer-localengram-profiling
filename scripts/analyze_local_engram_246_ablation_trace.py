"""Attribute Engram read and model-kernel intervals in the pinned EP4 decode ablation."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import median

from analyze_local_engram_246_gathers import (
    EXPECTED_GATHERS,
    EXPECTED_STEPS,
    FORWARD_SCOPE,
    GRAPH_LAUNCH,
    HASH_KERNEL,
    INDEX_SELECT,
    TOLERANCE_US,
    arg,
    correlation,
    end,
    lane,
    layer_split,
    node,
    same_host_interval,
)


VARIANTS = ("base", "fused", "prefetch", "fused_prefetch")
LAYER0 = re.compile(r"\bexpand_to_mhc_fwd_cuda(?:_kernel)?\b")
HC_POST = re.compile(r"\bmhc_post_fwd_cuda(?:_kernel)?\b")
GATE = re.compile(r"\bengram_gate_fwd_kernel_cuda(?:_kernel)?\b")
PROJECTION = re.compile(r"\bsm100_fp8_fp4_gemm_1d1d_impl<[^>]*\b25600u,\s*6144u,")
OWNER = re.compile(r"\b_owner_gather\b")
# A reporting band for this experiment, not a calibration of CUPTI accuracy.
BOUNDARY_AMBIGUITY_US = 0.25


def signature(event):
    return event["name"], tuple(arg(event, "grid") or ()), tuple(arg(event, "block") or ())


def intervals(events):
    """Sorted interval union, without counting simultaneous activities twice."""
    result = []
    for start, stop in sorted((event["ts"], end(event)) for event in events):
        if not result or start > result[-1][1]:
            result.append((start, stop))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], stop))
    return result


def overlap_us(left, right, threshold=BOUNDARY_AMBIGUITY_US):
    a = b = 0
    total = 0.0
    while a < len(left) and b < len(right):
        low, high = left[a]
        start, stop = right[b]
        overlap = min(high, stop) - max(low, start)
        if overlap > threshold:
            total += overlap
        if high < stop:
            a += 1
        else:
            b += 1
    return total


def inspect_replay(step, scopes, launches, by_correlation, variant):
    row = {"step": step["name"]}

    def unresolved(reason):
        row["status"] = reason
        return row, None

    forwards = [e for e in scopes if FORWARD_SCOPE.fullmatch(e["name"]) and same_host_interval(step, e)]
    if len(forwards) != 1:
        return unresolved("unique_main_forward_scope_not_observed")
    forward = forwards[0]
    calls = sorted((e for e in launches if same_host_interval(forward, e)), key=lambda e: (e["ts"], -e["dur"]))
    if not calls or any(not same_host_interval(calls[0], e) for e in calls[1:]):
        return unresolved("unique_main_graph_launch_not_observed")
    correlations = {correlation(e) for e in calls}
    kernels = [e for key in correlations for e in by_correlation.get(key, ())]
    graph_ids = {arg(e, "graph id") for e in kernels}
    if len(graph_ids) != 1 or next(iter(graph_ids)) in (None, 0, "0"):
        return unresolved("main_graph_id_missing_or_ambiguous")
    row.update(main_graph_id=next(iter(graph_ids)), forward_scope=forward["name"], kernel_count=len(kernels),
               launch_correlations=sorted(correlations))
    ids = [arg(e, "graph node id") for e in kernels]
    if any(type(i) is not int or i <= 0 for i in ids) or len(set(ids)) != len(ids):
        return unresolved("unique_numeric_graph_node_ids_not_observed")
    devices = {lane(e)[:2] for e in kernels}
    if len(devices) != 1 or None in next(iter(devices)):
        return unresolved("gpu_or_context_not_unique")
    row["main_graph_gpu_kernel_envelope_us"] = max(map(end, kernels)) - min(e["ts"] for e in kernels)
    # Numeric ordering is used as a checked source-order heuristic, not as a
    # record of execution or a portable CUDA guarantee.
    kernels.sort(key=lambda e: arg(e, "graph node id"))
    by_id = {arg(e, "graph node id"): e for e in kernels}
    hashes = [e for e in kernels if HASH_KERNEL.search(e["name"])]
    layer0 = [e for e in kernels if LAYER0.search(e["name"])]
    gates = [i for i, e in enumerate(kernels) if GATE.search(e["name"])]
    projections = [i for i, e in enumerate(kernels) if PROJECTION.search(e["name"])]
    if len(hashes) != 1 or len(layer0) != 1:
        return unresolved("hash_or_layer0_anchor_missing_or_ambiguous")
    if len(gates) != 2 or projections != [i - 1 for i in gates]:
        return unresolved("two_engram_projection_gate_pairs_not_observed")
    hashed, early = hashes[0], layer0[0]
    projection1, projection14 = (kernels[i] for i in projections)
    if not (arg(hashed, "graph node id") < arg(early, "graph node id") < arg(projection1, "graph node id")
            < arg(projection14, "graph node id")):
        return unresolved("main_graph_source_brackets_inconsistent")
    row["hash_end_to_layer0_early_mhc_start_us"] = early["ts"] - end(hashed)
    anchors = {"hash": hashed, "layer0_early_mhc": early,
               "layer1_engram_projection": projection1, "layer14_engram_projection": projection14}
    for layer, gate_pos, proj_pos in zip((1, 14), gates, projections):
        anchors[f"layer{layer}_engram_gate"] = kernels[gate_pos]
        if HC_POST.search(kernels[proj_pos - 1]["name"]):
            anchors[f"layer{layer}_pre_engram_hc_post"] = kernels[proj_pos - 1]
    row["anchors"] = {key: {**node(e), "reported_stream": arg(e, "stream")} for key, e in anchors.items()}

    stock = [e for e in kernels if INDEX_SELECT.search(e["name"])]
    fused = [e for e in kernels if OWNER.search(e["name"])]
    row.update(stock_gather_nodes=len(stock), fused_owner_nodes=len(fused))
    if variant in ("fused", "fused_prefetch"):
        if len(fused) != 2 or stock:
            return unresolved("expected_two_fused_reads_and_no_stock_gathers")
        reads = {1: fused[:1], 14: fused[1:]}
        row["read_assignment"] = "two_unique_owner_nodes_in_inferred_source_order"
    else:
        if len(stock) != EXPECTED_GATHERS or fused:
            return unresolved("expected_16_stock_gathers_and_no_fused_reads")
        split = layer_split(stock)
        if "layers" not in split:
            return unresolved(split["status"])
        reads = {layer["layer"]: [by_id[i] for i in layer["data_node_ids"] + layer["scale_node_ids"]]
                 for layer in split["layers"]}
        row["read_assignment"] = "repeated_stock_data_and_scale_signatures_in_inferred_source_order"

    all_reads = reads[1] + reads[14]
    if not all(arg(hashed, "graph node id") < arg(e, "graph node id") < arg(early, "graph node id")
               for e in all_reads):
        return unresolved("read_nodes_outside_hash_to_layer0_source_bracket")
    read_intervals = {layer: intervals(events) for layer, events in reads.items()}
    deltas = {
        "first_gpu_kernel_start_minus_graph_api_start_us": min(e["ts"] for e in kernels) - calls[0]["ts"],
        "first_read_start_minus_hash_end_us": min(e["ts"] for e in all_reads) - end(hashed),
        "last_read_end_minus_layer0_start_us": max(map(end, all_reads)) - early["ts"],
        "layer1_last_read_end_minus_projection_end_us": read_intervals[1][-1][1] - end(projection1),
        "layer14_last_read_end_minus_projection_end_us": read_intervals[14][-1][1] - end(projection14),
    }
    boundary_reversal = max(
        0.0, -deltas["first_gpu_kernel_start_minus_graph_api_start_us"], -deltas["first_read_start_minus_hash_end_us"],
        deltas["layer1_last_read_end_minus_projection_end_us"], deltas["layer14_last_read_end_minus_projection_end_us"],
        deltas["last_read_end_minus_layer0_start_us"] if variant in ("base", "fused") else 0.0,
    )
    if boundary_reversal == 0:
        timing_status = "ordered_activity_boundaries"
    elif boundary_reversal <= BOUNDARY_AMBIGUITY_US:
        timing_status = "qualified_small_boundary_reversal"
    else:
        timing_status = "boundary_reversal_exceeds_reporting_band"
    row["timing"] = {
        "status": timing_status, "boundary_deltas_us": deltas, "max_boundary_reversal_us": boundary_reversal,
        "strict_0p1us_order_checks_passed": boundary_reversal <= TOLERANCE_US,
    }
    row["layer_reads"] = {}
    for layer, events in reads.items():
        active = read_intervals[layer]
        start, stop = active[0][0], active[-1][1]
        row["layer_reads"][str(layer)] = {
            "first_start_us": start, "last_end_us": stop, "span_us": stop - start,
            "kernel_active_union_us": sum(b - a for a, b in active),
            "kernels": [node(e) for e in events],
            "reported_streams": sorted({arg(e, "stream") for e in events if arg(e, "stream") is not None}),
            "start_end_relative_to_anchor_us": {key: [start - e["ts"], stop - e["ts"]]
                                                 for key, e in anchors.items()},
        }

    # The main sequence must match stock by name, grid and block, so unlabeled
    # routing/merge nodes from the side branch cannot count as model work.
    main = kernels[kernels.index(early):projections[1]]
    concurrency = {"overlap_with_layer1_reads_us": None, "inside_layer0_to_layer14_projection_envelope_us": None,
                   "overlap_with_verified_main_kernels_us": None, "raw_overlap_with_verified_main_kernels_us": None,
                   "raw_overlap_with_layer1_reads_us": None}
    if timing_status != "boundary_reversal_exceeds_reporting_band":
        concurrency.update(
            overlap_with_layer1_reads_us=overlap_us(read_intervals[14], read_intervals[1]),
            raw_overlap_with_layer1_reads_us=overlap_us(read_intervals[14], read_intervals[1], threshold=0),
            inside_layer0_to_layer14_projection_envelope_us=overlap_us(
                read_intervals[14], [(early["ts"], projection14["ts"])]
            ),
        )
    row["layer14_concurrency"] = concurrency
    row["status"] = "attributed"
    identity = (row["main_graph_id"], tuple((arg(e, "graph node id"), signature(e)) for e in kernels))
    return row, (identity, tuple(map(signature, main)), intervals(main), read_intervals[14])


def inspect_trace(trace, variant, base_signature=None):
    scopes, launches = [], []
    kernels_by_correlation = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or not all(key in event for key in ("name", "ts", "dur")):
            continue
        category, corr = event.get("cat"), correlation(event)
        if category == "user_annotation":
            scopes.append(event)
        elif corr is not None:
            if category in ("cuda_runtime", "cuda_driver") and GRAPH_LAUNCH.fullmatch(event["name"]):
                launches.append(event)
            elif category == "kernel" and event["dur"] > 0:
                kernels_by_correlation[corr].append(event)

    steps = sorted((e for e in scopes if e["name"] in EXPECTED_STEPS), key=lambda e: e["ts"])
    exact = Counter(e["name"] for e in steps) == Counter(EXPECTED_STEPS)
    windows = [inspect_replay(e, scopes, launches, kernels_by_correlation, variant) for e in steps]
    valid = [details for _, details in windows if details is not None]
    stable = exact and len(valid) == len(steps) and all(details[0] == valid[0][0] for details in valid)
    if variant == "base" and stable:
        base_signature = valid[0][1]

    for row, details in windows:
        if details is None:
            continue
        _, main_signature, main, read14 = details
        row["main_sequence_matches_stock"] = stable and base_signature is not None and main_signature == base_signature
        if not stable:
            reason = "four_stable_replays_not_observed"
        elif base_signature is None:
            reason = "matching_rank_stock_reference_unavailable"
        elif main_signature != base_signature:
            reason = "main_kernel_signatures_differ_from_stock_reference"
        elif row["timing"]["status"] == "boundary_reversal_exceeds_reporting_band":
            reason = "matched_model_sequence_but_timestamps_not_qualified"
        else:
            reason = "matched_stock_main_kernel_signatures"
            row["layer14_concurrency"]["overlap_with_verified_main_kernels_us"] = overlap_us(read14, main)
            row["layer14_concurrency"]["raw_overlap_with_verified_main_kernels_us"] = overlap_us(read14, main, threshold=0)
        row["main_model_overlap_status"] = reason

    rows = [row for row, _ in windows]
    return {
        "exact_steady_step_labels": exact, "stable_four_replays": stable,
        "status_counts": dict(Counter(row["status"] for row in rows)), "steps": rows,
    }, base_signature


def summary(traces):
    windows = [row for trace in traces for row in trace.get("steps", ())]
    rows = [row for row in windows if row["status"] == "attributed"]
    timed = [row for row in rows if row["timing"]["status"] != "boundary_reversal_exceeds_reporting_band"]

    def stats(values):
        data = [value for value in values if value is not None]
        return {"n": len(data), "median_us": median(data), "min_us": min(data), "max_us": max(data)} if data else {"n": 0}

    def offsets(layer, anchor):
        pairs = [row["layer_reads"][str(layer)]["start_end_relative_to_anchor_us"][anchor] for row in timed]
        return {"start": stats(start for start, _ in pairs), "end": stats(stop for _, stop in pairs)}

    return {
        "attributed_windows": len(rows),
        "stable_ranks": sum(trace.get("stable_four_replays", False) for trace in traces),
        "stock_gather_node_counts": dict(Counter(row["stock_gather_nodes"] for row in windows if "stock_gather_nodes" in row)),
        "fused_owner_node_counts": dict(Counter(row["fused_owner_nodes"] for row in windows if "fused_owner_nodes" in row)),
        "timing_status_counts": dict(Counter(row["timing"]["status"] for row in rows)),
        "strict_0p1us_order_windows": sum(row["timing"]["strict_0p1us_order_checks_passed"] for row in rows),
        "max_boundary_reversal": stats(row["timing"]["max_boundary_reversal_us"] for row in rows),
        "main_graph_gpu_kernel_envelope": stats(row.get("main_graph_gpu_kernel_envelope_us") for row in windows),
        "hash_end_to_layer0_early_mhc_start": stats(row.get("hash_end_to_layer0_early_mhc_start_us") for row in windows),
        "layer_read_kernel_active_union": {str(layer): stats(row["layer_reads"][str(layer)]["kernel_active_union_us"] for row in rows)
                                           for layer in (1, 14)},
        "layer_read_span": {str(layer): stats(row["layer_reads"][str(layer)]["span_us"] for row in rows) for layer in (1, 14)},
        "layer_reads_relative_to_layer0_mhc": {str(layer): offsets(layer, "layer0_early_mhc") for layer in (1, 14)},
        "layer_reads_relative_to_own_projection": {str(layer): offsets(layer, f"layer{layer}_engram_projection")
                                                  for layer in (1, 14)},
        "layer14_overlap_with_layer1_reads": stats(row["layer14_concurrency"]["overlap_with_layer1_reads_us"] for row in rows),
        "layer14_overlap_with_verified_main_kernels": stats(
            row["layer14_concurrency"]["overlap_with_verified_main_kernels_us"] for row in rows
        ),
        "layer14_raw_overlap_with_verified_main_kernels": stats(
            row["layer14_concurrency"]["raw_overlap_with_verified_main_kernels_us"] for row in rows
        ),
        "layer14_raw_overlap_with_layer1_reads": stats(
            row["layer14_concurrency"]["raw_overlap_with_layer1_reads_us"] for row in rows
        ),
    }


LIMITS = (
    "IDs correlate graph activities and support a pinned-source order heuristic; numerical node order is not an execution order. "
    "The layer 0 mHC anchor, HC-post, projection and gate are distinct GPU activities. Node counts and replay sequence "
    "do not depend on strict timestamp inequalities. Positive boundary reversals up to 0.25 us are retained and "
    "flagged, with raw deltas and the stricter 0.1 us check reported. The 0.25 us reporting band is not a calibration "
    "of CUPTI jitter and does not establish the cause of reversals. Qualified overlap drops intersections at or below "
    "0.25 us; raw overlap is shown separately. Both exclude unidentified Engram routing and layout work. "
    "A matched name/grid/block model prefix is required for main-kernel overlap. Kernel intervals may include PDL stalls; "
    "they do not expose actual memory transactions, SM activity, event wait duration, or saved critical-path time. "
    "The profiled main graph envelope runs from the first observed kernel start to the last observed kernel end "
    "and includes intervening gaps. There is no cross-rank timestamp subtraction. Use separately validated clean timings for step effects."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Ablation result root, one trace directory, or one raw Kineto JSON.")
    parser.add_argument("--variant", choices=VARIANTS, default="base", help="Variant label for a standalone trace input.")
    options = parser.parse_args()
    path = options.path

    if path.is_file() or list(path.glob("rank-*.kineto.json")):
        files = [path] if path.is_file() else sorted(path.glob("rank-*.kineto.json"))
        traces = [{"path": str(file), **inspect_trace(json.loads(file.read_text()), options.variant)[0]} for file in files]
        output = {options.variant: {"traces": traces, "summary": summary(traces)}}
    else:
        if not path.is_dir():
            parser.error(f"No trace input at {path}")
        reference = {}
        output = {}
        for variant in VARIANTS:
            traces = []
            for rank in range(4):
                file = path / variant / "profile-8192-8" / "trace" / f"rank-{rank}.kineto.json"
                if file.is_file():
                    report, base_signature = inspect_trace(json.loads(file.read_text()), variant, reference.get(rank))
                    traces.append({"rank": rank, "path": str(file), **report})
                    if variant == "base" and base_signature is not None:
                        reference[rank] = base_signature
                else:
                    traces.append({"rank": rank, "path": str(file), "missing": True})
            output[variant] = {"traces": traces, "summary": summary(traces)}
    print(json.dumps({"limits": LIMITS, "variants": output}, indent=2))


if __name__ == "__main__":
    main()
