"""Summarize clean ranks and isolated fetch checks from the #246 Engram ablation."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


RANKS = tuple(range(4))
VARIANTS = ("base", "fused", "prefetch", "fused_prefetch", "base_repeat")
FACTORS = VARIANTS[:4]
TIMING_FIELDS = {"engine_call": "engine_call_ns", "step_start_interval": "step_start_interval_ns"}
MATCH_FIELDS = (
    "input_hashes", "outputs", "window_context_before", "window_context_after",
    "window_emitted_tokens", "prefill_cached_tokens_by_step",
)
CONTRASTS = (
    ("base", "fused"), ("base", "prefetch"), ("base", "fused_prefetch"),
    ("fused", "fused_prefetch"), ("prefetch", "fused_prefetch"), ("base", "base_repeat"),
)


def read_object(path):
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as error:
        return None, str(error)
    if not isinstance(value, dict):
        return None, "expected a JSON object"
    return value, None


def valid_samples(values, count=None):
    return (isinstance(values, list) and bool(values) and (count is None or len(values) == count)
            and all(type(value) in (int, float) and 0 < value < 2**63 for value in values))


def stats(samples_us):
    ordered = sorted(samples_us)
    return {"samples": len(ordered), "p50_us": median(ordered), "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def load_clean(root, variant, requests):
    ranks, errors = {}, {}
    for rank in RANKS:
        path = root / variant / f"profile-8192-{requests}" / "clean" / f"rank-{rank}.json"
        data, error = read_object(path)
        if error is None:
            try:
                workload = data["workload"]
                if (data["rank"] != rank or data["mode"] != "clean" or workload["context"] != 8192
                        or workload["requests_per_rank"] != requests or workload.get("variant", variant) != variant):
                    raise ValueError("rank, mode, or workload does not match the path")
                if not isinstance(data["runtime"], dict) or any(key not in data for key in MATCH_FIELDS):
                    raise ValueError("missing runtime or parity metadata")
                outputs = data["outputs"]
                ordinals = {str(index) for index in range(requests)}
                if set(outputs) != ordinals or set(data["input_hashes"]) != ordinals:
                    raise ValueError("missing or unexpected request ordinals")
                for output in outputs.values():
                    if (output["finished"] is not True or len(output["token_ids"]) != workload["output_tokens"]
                            or not 0 <= output["accepted"] <= output["drafted"] or output["drafted"] <= 0):
                        raise ValueError("incomplete output or invalid draft counts")
                timings = data["engine_timings"]
                steps = workload.get("clean_steps", len(timings["engine_call_ns"]))
                if (type(steps) is not int or steps < 2 or not valid_samples(timings["engine_call_ns"], steps)
                        or not valid_samples(timings["step_start_interval_ns"], steps - 1)):
                    raise ValueError("invalid engine timing samples")
                if "base_match" in data and type(data["base_match"]) is not bool:
                    raise ValueError("base_match must be a boolean")
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                error = str(exc)
        if error is None:
            ranks[rank] = data
        else:
            errors[rank] = error
    return ranks, errors


def same_workload(workload):
    return {key: value for key, value in workload.items() if key != "variant"}


def rank_max_timings(ranks):
    timings = {}
    for metric, field in TIMING_FIELDS.items():
        series = [ranks[rank]["engine_timings"][field] for rank in RANKS]
        if len({len(samples) for samples in series}) != 1:
            return None
        timings[metric] = stats([max(step) / 1000 for step in zip(*series)])
    return timings


def summarize_clean(ranks, errors, baseline, variant):
    result = {"status": "incomplete", "valid_ranks": sorted(ranks), "rank_errors": errors, "timings": None}
    mismatches = {}
    for rank in sorted(ranks.keys() & baseline.keys()):
        data, ref = ranks[rank], baseline[rank]
        keys = [key for key in MATCH_FIELDS if data[key] != ref[key]]
        if same_workload(data["workload"]) != same_workload(ref["workload"]):
            keys.append("workload")
        if data["runtime"] != ref["runtime"]:
            keys.append("runtime")
        if data.get("base_match") is False:
            keys.append("base_match")
        if keys:
            mismatches[rank] = keys
    result["paired_parity"] = {"compared_ranks": sorted(ranks.keys() & baseline.keys()), "mismatches": mismatches}
    if len(ranks) != len(RANKS):
        return result
    first = ranks[0]
    if any(row["workload"] != first["workload"] or row["runtime"] != first["runtime"] for row in ranks.values()):
        result["status"] = "inconsistent_rank_metadata"
        return result
    result["workload"], result["runtime"] = first["workload"], first["runtime"]
    result["timings"] = rank_max_timings(ranks)
    if result["timings"] is None:
        result["status"] = "inconsistent_step_counts"
    elif len(baseline) != len(RANKS):
        result["status"] = "baseline_incomplete"
    elif mismatches:
        result["status"] = "parity_failed"
    elif variant != "base" and not all(row.get("base_match") is True for row in ranks.values()):
        result["status"] = "worker_parity_missing"
    else:
        result["status"] = "matched"
    return result


def difference(reference, variant):
    return {"reference_us": reference, "variant_us": variant, "reduction_us": reference - variant,
            "reduction_pct": 100 * (reference - variant) / reference, "speedup": reference / variant}


def compare(variants, reference, candidate):
    left, right = variants[reference], variants[candidate]
    if left["status"] != "matched" or right["status"] != "matched":
        return {"status": "unavailable", "reference_status": left["status"], "candidate_status": right["status"]}
    if any(left["timings"][key]["samples"] != right["timings"][key]["samples"] for key in TIMING_FIELDS):
        return {"status": "step_counts_differ"}
    return {"status": "matched", "timings": {
        metric: {q: difference(left["timings"][metric][q], right["timings"][metric][q]) for q in ("p50_us", "p95_us")}
        for metric in TIMING_FIELDS
    }}


def factorial_effects(variants):
    if any(variants[variant]["status"] != "matched" for variant in FACTORS):
        return {"status": "unavailable"}
    timings = {}
    for metric in TIMING_FIELDS:
        if len({variants[variant]["timings"][metric]["samples"] for variant in FACTORS}) != 1:
            return {"status": "step_counts_differ"}
        timings[metric] = {}
        for q in ("p50_us", "p95_us"):
            base, fused, prefetch, both = [variants[variant]["timings"][metric][q] for variant in FACTORS]
            timings[metric][q] = {
                "fused_mean_reduction_us": ((base - fused) + (prefetch - both)) / 2,
                "prefetch_mean_reduction_us": ((base - prefetch) + (fused - both)) / 2,
                "interaction_reduction_us": (fused - both) - (base - prefetch),
            }
    return {"status": "matched", "timings": timings}


def summarize_components(root):
    records, errors = {}, {}
    for rank in RANKS:
        data, error = read_object(root / f"rank-{rank}-component.json")
        if error is None:
            records[rank] = data
        else:
            errors[rank] = error
    expected = [f"q{q}-aligned{aligned}" for q in (4, 5, 32) for aligned in (0, 1)]
    byte_parity = {}
    for variant in FACTORS[1:]:
        ranks = {}
        for rank, row in records.items():
            checks = row.get("byte_parity")
            checks = checks if isinstance(checks, dict) else {}
            ranks[rank] = {key: checks[key].get(variant) if isinstance(checks.get(key), dict) else None for key in expected}
        if any(value is False for checks in ranks.values() for value in checks.values()):
            status = "failed"
        elif len(ranks) != len(RANKS) or any(value is not True for checks in ranks.values() for value in checks.values()):
            status = "incomplete"
        else:
            status = "passed"
        byte_parity[variant] = {"status": status, "checks_by_rank": ranks}

    isolated = {}
    for q in ("4", "32"):
        by_rank = {}
        for rank, row in records.items():
            timings = row.get("isolated_graph_us")
            values = timings.get(q) if isinstance(timings, dict) else None
            if isinstance(values, dict) and all(valid_samples(values.get(key), 12) for key in ("base", "fused")):
                by_rank[rank] = {key: stats(values[key]) for key in ("base", "fused")}
        status = "incomplete" if len(by_rank) != len(RANKS) else byte_parity["fused"]["status"]
        entry = {"status": status, "statistics_by_rank": by_rank}
        if status == "passed":
            entry["worst_rank_comparison"] = {
                stat: difference(max(rank["base"][stat] for rank in by_rank.values()),
                                 max(rank["fused"][stat] for rank in by_rank.values()))
                for stat in ("p50_us", "p95_us")
            }
        isolated[q] = entry
    return {"rank_errors": errors, "byte_parity": byte_parity, "isolated_graph": isolated}


def analyze(root):
    root = Path(root)
    cases = {}
    for requests in (1, 8):
        loaded = {variant: load_clean(root, variant, requests) for variant in VARIANTS}
        variants = {variant: summarize_clean(ranks, errors, loaded["base"][0], variant)
                    for variant, (ranks, errors) in loaded.items()}
        cases[f"profile-8192-{requests}"] = {
            "variants": variants,
            "contrasts": {f"{candidate}_vs_{reference}": compare(variants, reference, candidate)
                          for reference, candidate in CONTRASTS},
            "factorial_effects": factorial_effects(variants),
        }
    components = summarize_components(root)
    complete = (all(contrast["status"] == "matched" for case in cases.values() for contrast in case["contrasts"].values())
                and all(item["status"] == "passed" for item in components["byte_parity"].values())
                and all(item["status"] == "passed" for item in components["isolated_graph"].values()))
    return {
        "complete_and_matched": complete, "cases": cases, "components": components,
        "measurement": (
            "Clean latency uses the maximum across four ranks at each step ordinal. The median and empirical nearest-rank "
            "p95 summarize those samples. The base_repeat_vs_base contrast measures drift in the fixed run order. "
            "Positive reductions favor the variant. A positive factorial interaction means the combined reduction "
            "exceeds the sum of the individual reductions at that percentile. Isolated samples are averages of 32 graph "
            "replays; their worst-rank percentiles summarize independent rank series. These are descriptive comparisons."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Artifact root containing the variant directories and rank component JSONs")
    args = parser.parse_args()
    print(json.dumps(analyze(args.path), indent=2))
