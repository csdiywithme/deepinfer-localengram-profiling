"""Measure steady decode on the pinned #246 EP4 engine, or annotate a short Nsight capture.

Run clean and trace in separate, bounded process groups with matching workload arguments.
Trace with: nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop
  -o <report> python -u <this script> --mode trace --reference-dir <clean output> ...
Without nsys, run --mode trace --trace-backend torch to collect raw Kineto traces.
Use --sweep to load each rank's engine once, run three clean cases, then trace the same cases.
Stage logs use the prefix LOCAL_ENGRAM_PROFILE. Rank JSON files retain raw tokens and times.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from local_engram_246_model_smoke import (
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODEL_REV,
    NEXTN,
    RANKS,
    SMOKE_ENV,
    SOURCE_REV,
    check_model_snapshot,
    check_rank_device,
)


PREFILL_CHUNK = 4096
OUTPUT_TOKENS = 256
WARMUP_STEPS = 12
CLEAN_STEPS = 32
TRACE_STEPS = 5  # First step absorbs capture and initial annotation setup costs.
PROMPT = "In an orderly sequence, positive integers start at one and increase by one each time."
SWEEP_CASES = ((8192, 1), (131072, 1), (8192, 8))
SWEEP_MAX_MODEL_LEN = 132 * 1024
SWEEP_MAX_NUM_SEQS = 8
SWEEP_HISTORY = (
    "Requests are fully drained and never explicitly prefix-forked. A live engine retains its allocator, "
    "JIT and CUDA graph history across cases. Clean precedes all profiling. Treat per-case timing as "
    "fixed-order warm-engine data, with order and shape effects potentially mixed."
)


def log(stage, rank=None, **fields):
    event = {"stage": stage, "rank": rank, "pid": os.getpid(), **fields}
    os.write(1, ("LOCAL_ENGRAM_PROFILE " + json.dumps(event) + "\n").encode())


@contextmanager
def mark_zos_metadata(range_factory):
    """Mark the pinned wrapper's C++ pack and the two distinct H2D submissions."""
    from deepinfer.model_executor.input_metadata.prepare_tensor_input_zos import prepare_decode_inputs_zos

    code = prepare_decode_inputs_zos.__code__
    source, first = inspect.getsourcelines(prepare_decode_inputs_zos)
    boundaries = {
        ") = prepare_decode_zos_inputs(": "zos/cpp_pack_and_unpack",
        "cu_seqlens_q = cu_seqlens_q_cpu.cuda(non_blocking=True)": None,
        "block_tables_offsets = block_tables_offsets_cpu.cuda(non_blocking=True)": "zos/h2d/block_table_offsets",
        "block_tables_in = block_tables_in_cpu.cuda(non_blocking=True)": "zos/h2d/block_tables_flat",
        "fresh_input_ids = None if fresh_input_ids_cpu is None else fresh_input_ids_cpu.cuda(non_blocking=True)": None,
    }
    line_marks = {first + i: boundaries[line.strip()] for i, line in enumerate(source) if line.strip() in boundaries}
    if len(line_marks) != len(boundaries):
        raise RuntimeError("Unexpected source layout for #246 ZOS metadata annotation")
    counts = dict.fromkeys((label for label in boundaries.values() if label is not None), 0)
    monitor = sys.monitoring
    tool_id = 3
    active = None
    active_label = None

    def on_line(_code, line):
        nonlocal active, active_label
        if line not in line_marks:
            return monitor.DISABLE
        label = line_marks[line]
        # CPython may revisit the opening line of a multiline call just before calling it.
        if label == active_label:
            return None
        if active is not None:
            active.__exit__(None, None, None)
            active = None
        active_label = label
        if label is not None:
            active = range_factory(label)
            active.__enter__()
            counts[label] += 1

    monitor.use_tool_id(tool_id, "DeepInfer #246 ZOS metadata NVTX")
    try:
        monitor.register_callback(tool_id, monitor.events.LINE, on_line)
        monitor.set_local_events(tool_id, code, monitor.events.LINE)
        yield counts
    finally:
        if active is not None:
            active.__exit__(None, None, None)
        monitor.set_local_events(tool_id, code, monitor.events.NO_EVENTS)
        monitor.register_callback(tool_id, monitor.events.LINE, None)
        monitor.free_tool_id(tool_id)


@contextmanager
def trace_session(options, rank):
    import torch
    import torch.distributed as dist

    dist.barrier()
    if options.trace_backend == "nsys":
        from torch.cuda import nvtx, profiler

        if rank == 0:
            profiler.start()
        dist.barrier()
        yield nvtx.range
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            profiler.stop()
    else:
        from torch.profiler import ProfilerActivity, profile, record_function

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as capture:
            dist.barrier()
            yield record_function
            torch.cuda.synchronize()
        capture.export_chrome_trace(str(options.output_dir / f"rank-{rank}.kineto.json"))
    dist.barrier()


def run_case(llm, rank, options, workload, runtime, *, reuse_engine):
    import numpy as np
    import torch
    import torch.distributed as dist
    from deepinfer._C.engine import RequestOutput
    from deepinfer.sampling_params import SamplingParams

    engine = llm.llm_engine
    case_fields = {"mode": options.mode, "context": options.context, "requests_per_rank": options.requests}
    free_blocks = getattr(engine.scheduler, "get_num_free_gpu_blocks", None)
    free_before = None
    if reuse_engine:
        if engine.get_num_unfinished_requests() or engine.driver_worker.prev_step_state is not None:
            raise RuntimeError("Engine is not drained before the next case")
        free_before = free_blocks() if callable(free_blocks) else None
    log("case_start", rank, **case_fields)
    requests = {}
    input_hashes = {}
    request_ordinals = {}
    sequences = []
    for i in range(options.requests):
        prompt = f"Example {i}. {PROMPT}"
        tokens = np.resize(np.asarray(engine.tokenizer.encode(prompt, add_special_tokens=False), dtype="<i4"),
                           options.context)
        tokens[0] = engine.tokenizer.bos_token_id
        ordinal = str(i)
        input_hashes[ordinal] = hashlib.sha256(tokens.tobytes()).hexdigest()
        sampling = SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True, seq_seed=i)
        seq = llm._add_request(None, sampling, tokens)
        if seq is None or seq.request_id in request_ordinals:
            raise RuntimeError(f"Engine refused request or reused an ID for ordinal {ordinal}")
        if reuse_engine and seq.num_cached_tokens != 0:
            raise RuntimeError(f"New request {seq.request_id} already has cached tokens")
        request_ordinals[seq.request_id] = ordinal
        if reuse_engine:
            sequences.append(seq)
        requests[ordinal] = {"token_ids": [], "drafted": 0, "accepted": 0, "finished": False}
    if reuse_engine:
        # Match the upstream benchmark's admission phase for each fresh batch.
        engine.scheduler.decode_step = 0

    def consume(raw):
        for item in raw:
            token = RequestOutput.clone(item)
            out = requests[request_ordinals[token.request_id]]
            if out["finished"] or token.prompt_len != options.context or not math.isfinite(token.token_logprobs):
                raise RuntimeError(f"Invalid output for request {token.request_id}")
            out["token_ids"].append(token.token_id)
            includes_target = int(token.draft_tok > 0)
            out["drafted"] += token.draft_tok - includes_target
            out["accepted"] += token.accept_tok - includes_target
            out["finished"] = token.finished

    def full_batch():
        return len(engine.scheduler.running) == options.requests and not engine.scheduler.waiting

    # Advance every rank together until all requests have left prefill. This is outside the timing window.
    ready = torch.zeros((), dtype=torch.int32)
    prefill_limit = math.ceil(options.context * options.requests / PREFILL_CHUNK) + 256
    prefill_progress = []
    for prefill_step in range(1, prefill_limit + 1):
        consume(engine.step())
        if reuse_engine:
            prefill_progress.append([seq.num_cached_tokens for seq in sequences])
        ready.fill_(full_batch())
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if ready.item():
            break
    else:
        raise RuntimeError("The full decode batch did not become active")
    sequences.clear()

    def context_lengths():
        return [seq.get_len() for seq in engine.scheduler.running]

    for _ in range(WARMUP_STEPS):
        consume(engine.step())
    if not full_batch():
        raise RuntimeError("The full batch was not retained through decode warmup")
    log("steady_ready", rank, prefill_steps=prefill_step, context_lengths=context_lengths(), **case_fields)

    before = context_lengths()
    emitted_before = sum(len(out["token_ids"]) for out in requests.values())
    timings = None
    marks = None
    if options.mode == "clean":
        starts, returns = [], []
        for _ in range(CLEAN_STEPS):
            starts.append(time.perf_counter_ns())
            raw = engine.step()
            returns.append(time.perf_counter_ns())
            consume(raw)
            if not full_batch():
                raise RuntimeError("Decode batch size changed during timing")
        timings = {
            "engine_call_ns": [end - start for start, end in zip(starts, returns)],
            "step_start_interval_ns": [b - a for a, b in zip(starts, starts[1:])],
        }
    else:
        with trace_session(options, rank) as mark, mark_zos_metadata(mark) as marks:
            for step in range(TRACE_STEPS):
                name = "zos/capture_guard" if step == 0 else f"zos/steady_step/{step}"
                with mark(name):
                    raw = engine.step()
                consume(raw)
                if not full_batch():
                    raise RuntimeError("Decode batch size changed during trace")
        if len(set(marks.values())) != 1 or min(marks.values()) < TRACE_STEPS:
            raise RuntimeError(f"The steady window missed required ZOS metadata calls: {marks}")
    after = context_lengths()
    emitted = sum(len(out["token_ids"]) for out in requests.values()) - emitted_before

    for _ in range(2 * OUTPUT_TOKENS + 128):
        if not engine.has_unfinished_requests():
            break
        consume(engine.step())
    else:
        raise RuntimeError("Decode did not finish within the drain step limit")
    consume(engine.flush_inflight_reqs())
    for request_id, out in requests.items():
        if not (out["finished"] and len(out["token_ids"]) == OUTPUT_TOKENS and
                0 <= out["accepted"] <= out["drafted"] and out["drafted"] > 0):
            raise RuntimeError(f"Incomplete output or invalid MTP counts for request {request_id}")
    free_after = None
    if reuse_engine:
        torch.cuda.synchronize()
        engine.deferred_release.release()
        if engine.get_num_unfinished_requests() or engine.driver_worker.prev_step_state is not None:
            raise RuntimeError("Engine is not drained after the case")
        free_after = free_blocks() if callable(free_blocks) else None
        if free_before != free_after:
            raise RuntimeError(f"Free GPU block count changed after drain: {free_before} -> {free_after}")

    result = {
        "mode": options.mode, "rank": rank, "pid": os.getpid(), "workload": workload, "runtime": runtime,
        "input_hashes": input_hashes, "outputs": requests, "engine_timings": timings,
        "trace_backend": options.trace_backend, "metadata_marker_counts": marks,
        "window_context_before": before, "window_context_after": after, "window_emitted_tokens": emitted,
        "reference_match": None,
    }
    if reuse_engine:
        result.update(prefill_cached_tokens_by_step=prefill_progress, free_gpu_blocks_before=free_before,
                      free_gpu_blocks_after=free_after, engine_history=SWEEP_HISTORY,
                      request_ids_by_ordinal={ordinal: request_id for request_id, ordinal in request_ordinals.items()})
    if options.reference_dir is not None:
        reference = json.loads((options.reference_dir / f"rank-{rank}.json").read_text())
        keys = ("workload", "runtime", "input_hashes", "outputs", "window_context_before")
        if reuse_engine:
            keys += ("prefill_cached_tokens_by_step", "free_gpu_blocks_before", "free_gpu_blocks_after")
        result["reference_match"] = reference["mode"] == "clean" and all(result[k] == reference[k] for k in keys)
    with (options.output_dir / f"rank-{rank}.json").open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    log("window_done", rank, context_before=before, context_after=after, tokens=emitted, metadata_marker_counts=marks,
        reference_match=result["reference_match"], **case_fields)
    return result["reference_match"]


def run_rank(dp_size, rank, *, model_dir, cases, reuse_engine):
    from _config import create_llm, setup_common_env

    options, workload = cases[0]
    log("rank_start", rank)
    try:
        setup_common_env(model_name=model_dir, sampler_force_topk=4096, set_override_experts=False, enable_eplb=False)
        llm = create_llm(
            model_dir, rank, ep_size=RANKS, cp_size=1, data_parallel_size=dp_size,
            max_num_seqs=workload.get("engine_max_num_seqs", options.requests), max_num_batched_tokens=PREFILL_CHUNK,
            max_model_len=workload["max_model_len"],
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION, lookahead_tokens=1, profile_init=False,
        )

        import torch
        import torch.distributed as dist
        from deepinfer import envs
        from deepinfer.model_executor.parallel_utils.engram_local import LocalEngramBuffer
        from deepinfer.model_executor.parallel_utils.userbuffer import userbuffer

        engine = llm.llm_engine
        engram = userbuffer.engram
        if not isinstance(engram, LocalEngramBuffer) or (engram.rank_idx, engram.num_ranks) != (rank, RANKS):
            raise RuntimeError("Expected the local EP4 Engram shard")
        if not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING or envs.DEEPINFER_USE_NEXTN_DECODE != NEXTN:
            raise RuntimeError("Expected ZOS and nextn=3")
        graph_runner = engine.driver_worker.model_runner.cuda_graph_runner
        if envs.DEEPINFER_ENABLE_MTP_VARLEN or graph_runner is None or not graph_runner.graphs:
            raise RuntimeError("Expected fixed-size verification and captured CUDA graphs")
        gpu = check_rank_device(torch)
        if (dist.get_rank(), dist.get_world_size()) != (rank, RANKS):
            raise RuntimeError(f"Unexpected rank or device mapping: {gpu}, rank {dist.get_rank()}")
        runtime = {
            "gpu": gpu, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version()), "cache_block_size": engine.cache_config.block_size,
            "graph_batch_sizes": list(graph_runner.batch_sizes), "engram_shard_bytes": engram.num_bytes,
        }
        log("engine_ready", rank, visible=os.environ["CUDA_VISIBLE_DEVICES"], **runtime)

        bad_case = None
        for options, workload in cases:
            matched = run_case(llm, rank, options, workload, runtime, reuse_engine=reuse_engine)
            if reuse_engine:
                # Stop together if any rank fails the matched-reference check.
                matched_rank = torch.tensor(int(matched is not False), dtype=torch.int32)
                dist.all_reduce(matched_rank, op=dist.ReduceOp.MIN)
                matched = bool(matched_rank.item())
            if matched is False:
                bad_case = f"{options.context} tokens, {options.requests} requests/rank"
                break
        llm.close()
        if bad_case:
            raise RuntimeError(f"Clean/trace reference mismatch for {bad_case}")
    except Exception as error:
        log("rank_error", rank, error_type=type(error).__name__, error=str(error)[:1000])
        raise
    log("rank_pass", rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path("/opt/DeepInfer"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sweep", action="store_true", help="Reuse one engine for three matched clean/Kineto cases")
    parser.add_argument("--reference-dir", type=Path, help="Required for trace. Contains clean rank JSON files.")
    parser.add_argument("--mode", choices=("clean", "trace"), help="Single-case mode, defaults to clean")
    parser.add_argument("--trace-backend", choices=("nsys", "torch"), help="Trace only, defaults to nsys")
    parser.add_argument("--context", type=int, choices=(8192, 131072), help="Single-case context, defaults to 8192")
    parser.add_argument("--requests", type=int, choices=(1, 8), help="Single-case requests per GPU rank, defaults to 1")
    options = parser.parse_args()
    if options.sweep:
        if any(value is not None for value in (options.mode, options.context, options.requests, options.reference_dir)):
            parser.error("--sweep fixes the modes, cases and references")
        if options.trace_backend not in (None, "torch"):
            parser.error("--sweep supports only --trace-backend torch")
        options.mode, options.trace_backend = "sweep", "torch"
    else:
        options.mode = options.mode or "clean"
        options.context = options.context or 8192
        options.requests = options.requests or 1
        if options.mode == "trace":
            if options.reference_dir is None:
                parser.error("trace requires --reference-dir from a separate clean run")
            options.trace_backend = options.trace_backend or "nsys"
        elif options.reference_dir is not None or options.trace_backend is not None:
            parser.error("--reference-dir and --trace-backend are only for trace")
    if options.mode in ("trace", "sweep") and not hasattr(sys, "monitoring"):
        raise RuntimeError("Trace annotation requires Python 3.12+")
    source = options.source_root.resolve(strict=True)
    model = options.model_dir.resolve(strict=True)
    options.output_dir = options.output_dir.resolve()
    rev = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    if rev != SOURCE_REV:
        raise RuntimeError(f"Expected #246 head {SOURCE_REV}, got {rev}")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "--ignore-submodules=untracked", "HEAD", "--"],
                   check=True, timeout=30)
    common = {
        "source_rev": rev, "model_rev": MODEL_REV, "model_index_sha256": check_model_snapshot(model),
        "prefill_chunk": PREFILL_CHUNK, "output_tokens": OUTPUT_TOKENS, "warmup_steps": WARMUP_STEPS,
        "ep": RANKS, "dp": RANKS, "cp": 1, "nextn": NEXTN, "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }
    cases = []
    if options.sweep:
        for mode in ("clean", "trace"):
            for context, requests in SWEEP_CASES:
                case_dir = options.output_dir / f"profile-{context}-{requests}"
                case = argparse.Namespace(mode=mode, context=context, requests=requests, output_dir=case_dir / mode,
                                          reference_dir=case_dir / "clean" if mode == "trace" else None,
                                          trace_backend="torch" if mode == "trace" else None)
                workload = dict(common, context=context, requests_per_rank=requests, max_model_len=SWEEP_MAX_MODEL_LEN,
                                engine_max_num_seqs=SWEEP_MAX_NUM_SEQS, warm_engine_sweep=True)
                cases.append((case, workload))
    else:
        workload = dict(common, context=options.context, requests_per_rank=options.requests, max_model_len=MAX_MODEL_LEN)
        if options.reference_dir is not None:
            options.reference_dir = options.reference_dir.resolve(strict=True)
            for rank in range(RANKS):
                reference = json.loads((options.reference_dir / f"rank-{rank}.json").read_text())
                if reference["mode"] != "clean" or reference["workload"] != workload:
                    raise RuntimeError(f"Rank {rank} reference belongs to a different workload")
        cases.append((options, workload))
    for case, _ in cases:
        if any(case.output_dir.glob("rank-*.json")):
            raise RuntimeError(f"Choose a fresh output directory: {case.output_dir}")
    for case, _ in cases:
        case.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(SMOKE_ENV, DEEPINFER_LOGGING_INTERVAL_STEP="0")
    os.environ.update(env)
    sys.path.insert(0, str(source / "examples"))

    import torch
    from _launch import launch_dp

    if torch.cuda.device_count() != RANKS:
        raise RuntimeError("Expected exactly four visible CUDA GPUs")
    launch_fields = {"workloads": [workload for _, workload in cases[:len(SWEEP_CASES)]], "engine_history": SWEEP_HISTORY}
    if not options.sweep:
        launch_fields = {"workload": cases[0][1]}
    log("launch", mode=options.mode, trace_backend=options.trace_backend, model_dir=str(model), env=env, **launch_fields)
    launch_dp(partial(run_rank, model_dir=str(model), cases=cases, reuse_engine=options.sweep), RANKS)
    log("pass", ranks=RANKS, mode=options.mode)


if __name__ == "__main__":
    main()
