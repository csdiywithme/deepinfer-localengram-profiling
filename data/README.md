# Measurement data

`ablation/` contains the timing and parity fields from one four-B300 run. The records cover five configurations (`base`, `fused`, `prefetch`, `fused_prefetch`, and `base_repeat`), two 8,192-token workloads (one or eight requests per rank), and four ranks per case. The component records contain separate fetch timing and byte-parity checks. The model used generated benchmark prompts, and the token IDs are included so the analyzer can recheck parity.

`clean-analysis.json` contains the calculated latency percentiles, paired workload checks, and component results. To regenerate it from the repository root, run:

```sh
python3 scripts/analyze_local_engram_246_ablation.py data/ablation > /tmp/localengram-clean-analysis.json
cmp /tmp/localengram-clean-analysis.json data/clean-analysis.json
```

`trace-samples/` contains rank 0 Kineto captures from the eight-request case for `base` and `fused`. The captures retain the recorded events and omit host and storage metadata. Decompress them for viewing in Perfetto or inspecting with the trace analyzer:

```sh
trace_dir=$(mktemp -d)
for variant in base fused; do
  trace_file="$trace_dir/$variant-rank-0.kineto.json"
  gzip -dc "data/trace-samples/$variant-rank-0.kineto.json.gz" > "$trace_file"
  python3 scripts/analyze_local_engram_246_ablation_trace.py \
    "$trace_file" --variant "$variant" > "$trace_dir/$variant-analysis.json"
done
```

The rank 0 samples each contain four steady decode replays. The stock main forward graph has 16 `index_select` nodes associated with the Engram gather in each replay. The fused graph has two `_owner_gather` nodes. Use the clean data for step latency and the profiled captures for kernel attribution.
