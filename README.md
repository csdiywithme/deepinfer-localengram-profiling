# DeepInfer LocalEngram profiling

Four-B300 experiments on DeepSeek V4.1 Flash with full weights. This repository contains the measurement harness, two experimental fetch adapters, per-rank timing records, and representative Kineto traces.

## Why this experiment

The original DeepInfer Engram route did not initialize GDAKI in the Modal configuration I tested. The failure reached the DevX event-channel path. [Community PR #246](https://github.com/deepseek-ai/DeepInfer/pull/246), submitted by [wukehu](https://github.com/wukehu) and credited to the Intellifusion Cluster Inference Team, provides an opt-in LocalEngram route that stores table shards in GPU symmetric memory and reads them over NVLink. It ran on four B300s with GIN and InfiniBand disabled. I used the stock PR as the baseline to test two ways to change the fetch path.

- **Fused gather** uses a Triton kernel to read each requested row from its owner GPU and write the output data and FP8 scale layout together.
- **Layer prefetch** fetches the first Engram layer on the caller stream and schedules later layers on a side stream. The factorial comparison also enables both adapters together.

## Results

P50 decode step start-to-start interval, in milliseconds. Positive reductions favor the variant. The baseline repeat ran last.

| Requests per rank, 8,192 tokens each | Stock #246 | Fused | Prefetch | Fused + prefetch | Stock repeat | Fused reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.320 | 9.106 | 9.258 | 9.134 | 9.288 | **2.30%** |
| 8 | 10.364 | 10.173 | 10.347 | 10.204 | 10.363 | **1.84%** |

The variants used the same four GPUs (EP4/DP4/CP1), fixed DSpark `nextn=3`, deterministic generated prompts, and 12 warmup steps. The adapters are installed while recapturing the decode CUDA Graphs. Eager prefill uses the stock fetch in every variant. Each case measured 32 clean steps, giving 31 start-to-start intervals. For each step the analysis takes the maximum across the four ranks. Input tokens, generated tokens, and draft counters matched the stock run on all ranks. The stock repeat was 32.2 µs and 1.1 µs faster at P50 for the two cases. These results describe one fixed-order allocation with synthetic prompts. Service throughput, multi-node behavior, and model quality on real workloads were outside the measurement.

The eight-request rank 0 captures show **16 stock Engram `index_select` nodes per main-model graph replay** and **two fused owner-gather nodes per replay** in the corresponding capture. The profiler records kernel activity separately from clean timing. Prefetch by itself had a smaller measured effect, and adding it to fusion gave back about 28–30 µs at P50. The saved timing does not isolate a cause for that interaction.

[Per-rank records and trace samples](data/README.md) · [Calculated results](data/clean-analysis.json)

## Verify the published evidence

Run from the repository root with Python 3.10 or newer. The analyzer uses only the standard library.

```sh
out=$(mktemp)
python3 scripts/analyze_local_engram_246_ablation.py data/ablation > "$out"
cmp "$out" data/clean-analysis.json
```

See the [data guide](data/README.md) for the standalone trace analyzer and Perfetto input. Both samples retain the recorded kernel events and omit host and storage metadata.

## Re-run on GPUs

The historical run used DeepInfer PR head [`0a9059c`](https://github.com/deepseek-ai/DeepInfer/tree/0a9059c516f9b458f2b271d207a49ac80388763e), the [pinned DeepSeek V4.1 Flash snapshot](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), Python 3.12, PyTorch `2.13.0+cu132`, CUDA 13.2, and the `nvidia-nccl-cu13==2.32.3` runtime wheel. The image builds the pinned FlashMLA with its post-link register-spill check skipped because that check rejected four sparse-decode symbols in this toolchain. The benchmark uses that build for all variants.

### Modal

Use a Modal workspace with a four-B300 allocation and capacity for the pinned model snapshot (about 510 GB). The weight bootstrap downloads that snapshot to a workspace volume. The smoke and ablation use separate paid GPU jobs. This release was checked offline against the historical records. The repackaged Modal workflow has not been rerun on GPUs.

```sh
python -m pip install 'modal==1.5.5'
modal run scripts/modal_local_engram_246_weights.py --full
modal run scripts/modal_local_engram_246.py --mode smoke
modal run scripts/modal_local_engram_246.py
```

Set `RUN_ID` to the printed run ID, then download the artifacts. Set `LOCALENGRAM_WEIGHTS_VOLUME` or `LOCALENGRAM_RESULTS_VOLUME` to reuse named volumes in your workspace.

```sh
mkdir -p runs
modal volume get "${LOCALENGRAM_RESULTS_VOLUME:-localengram-246-results}" "/$RUN_ID" "runs/$RUN_ID"
python3 scripts/analyze_local_engram_246_ablation.py "runs/$RUN_ID/ablation"
```

### Existing Linux GPU environment

Use the pinned DeepInfer checkout and vendor extensions already built, Python 3.12 or newer, the pinned weights, exactly four visible B300s on one NVLink node, and a new or empty output directory.

```sh
PYTHONPATH="$DEEPINFER_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
DEEPINFER_MODAL_GPU_MODEL=B300 \
python scripts/local_engram_246_variants.py \
  --source-root "$DEEPINFER_ROOT" --model-dir "$WEIGHTS" --output-dir "$RUN_DIR"
```

Analyze the resulting directory with `scripts/analyze_local_engram_246_ablation.py` and `scripts/analyze_local_engram_246_ablation_trace.py`.

## Attribution

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) and [PR #246](https://github.com/deepseek-ai/DeepInfer/pull/246) supply the engine and stock LocalEngram implementation. This repository publishes the reproduction harness, prototype adapters, and measurements for the pinned PR. The engine, vendor libraries, and model weights are fetched from their original sources.
