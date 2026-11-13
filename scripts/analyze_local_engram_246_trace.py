"""Attribute #246 ZOS CPU scopes to device transfers without summing overlapping time."""

import argparse
import json
from pathlib import Path
from statistics import median


SCOPES = {
    "pack": "zos/cpp_pack_and_unpack",
    "offsets": "zos/h2d/block_table_offsets",
    "flat": "zos/h2d/block_tables_flat",
}
KERNELS = ("_prepare_v4_decode_metadata_kernel", "_copy_block_tables_kernel")
EXPECTED_STEPS = {f"zos/steady_step/{i}" for i in range(1, 5)}


def same_host_interval(parent, child):
    return (parent.get("pid") == child.get("pid") and parent.get("tid") == child.get("tid")
            and parent["ts"] <= child["ts"]
            and child["ts"] + child.get("dur", 0) <= parent["ts"] + parent.get("dur", 0) + 0.01)


def correlation(event):
    return event.get("args", {}).get("correlation")


def trace_rows(events):
    complete = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    # Kineto also emits projected GPU annotations with the same names.
    host_scopes = [e for e in complete if e.get("cat") == "user_annotation"]
    steps = [e for e in host_scopes if e.get("name", "").startswith("zos/steady_step/")]
    runtimes = [e for e in complete if e.get("cat") in ("cuda_runtime", "cuda_driver") and correlation(e) is not None]
    gpu_copies = [e for e in complete if "gpu_memcpy" in e.get("cat", "") and "htod" in e.get("name", "").lower()]
    gpu_kernels = [e for e in complete if e.get("cat") == "kernel" and any(k in e.get("name", "") for k in KERNELS)]
    rows = []
    for step in sorted(steps, key=lambda x: x["ts"]):
        row = {"step": step["name"], "host_step_start_us": step["ts"], "host_step_us": step["dur"],
               "scopes": {}, "copies": {}, "attributed_kernels": []}
        step_launches = {correlation(e) for e in runtimes if same_host_interval(step, e)}
        for key, name in SCOPES.items():
            scopes = [e for e in host_scopes if e.get("name") == name and same_host_interval(step, e)]
            row["scopes"][key] = [{"start_us": e["ts"], "duration_us": e["dur"]} for e in scopes]
            if key == "pack":
                continue
            launch_ids = {correlation(e) for e in runtimes if "memcpy" in e.get("name", "").lower()
                          and any(same_host_interval(scope, e) for scope in scopes)}
            row["copies"][key] = [gpu_row(e) for e in gpu_copies if correlation(e) in launch_ids]
        row["attributed_kernels"] = [gpu_row(e) for e in gpu_kernels if correlation(e) in step_launches]
        rows.append(row)
    return rows, [gpu_row(e) for e in gpu_kernels]


def gpu_row(event):
    args = event.get("args", {})
    return {"name": event["name"], "start_us": event["ts"], "duration_us": event["dur"],
            "device": args.get("device"), "context": args.get("context"), "stream": args.get("stream"),
            "bytes": args.get("bytes"),
            "correlation": correlation(event)}


def complete_step(step):
    if any(len(scopes) != 1 for scopes in step["scopes"].values()):
        return False
    if any(len(copies) != 1 for copies in step["copies"].values()):
        return False
    copies = [step["copies"][key][0] for key in ("offsets", "flat")]
    return all((copy["bytes"] or 0) > 0 for copy in copies) and len({(copy["context"], copy["correlation"]) for copy in copies}) == 2


def analyze(run):
    ranks = []
    for rank in range(4):
        clean = json.loads((run / f"clean/rank-{rank}.json").read_text())
        traced = json.loads((run / f"trace/rank-{rank}.json").read_text())
        if not traced.get("reference_match") or clean["mode"] != "clean" or traced["mode"] != "trace":
            raise RuntimeError(f"Rank {rank} does not have matched tokens and workload")
        raw = json.loads((run / f"trace/rank-{rank}.kineto.json").read_text())
        rows, kernels = trace_rows(raw["traceEvents"])
        complete = [s for s in rows if complete_step(s)]
        ranks.append({"rank": rank, "scope_copy_pairs": len(complete), "trace_steps": len(rows),
                      "expected_step_labels": {s["step"] for s in rows} == EXPECTED_STEPS,
                      "engine_call_p50_us": median(clean["engine_timings"]["engine_call_ns"]) / 1000,
                      "step_interval_p50_us": median(clean["engine_timings"]["step_start_interval_ns"]) / 1000,
                      "pack_trace_p50_us": median(s["scopes"]["pack"][0]["duration_us"] for s in complete) if complete else None,
                      "gpu_h2d_p50_us": {k: median(s["copies"][k][0]["duration_us"] for s in complete) if complete else None
                                         for k in ("offsets", "flat")},
                      "rows": rows, "all_metadata_kernels": kernels})
    return {"matched_all_ranks": True,
            "complete_copy_attribution_all_ranks": all(r["scope_copy_pairs"] == r["trace_steps"] == len(EXPECTED_STEPS)
                                                       and r["expected_step_labels"] for r in ranks),
            "interpretation": "CPU scopes and GPU activities may overlap. Their durations do not establish a critical-path wait or speedup.",
            "ranks": ranks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Local artifact directory containing clean/ and trace/")
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))
