"""Paired Engram decode ablation on one resident #246 model per rank."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from local_engram_246_model_profile import (
    CLEAN_STEPS, OUTPUT_TOKENS, PREFILL_CHUNK, SWEEP_MAX_MODEL_LEN,
    SWEEP_MAX_NUM_SEQS, WARMUP_STEPS, log, run_case,
)
from local_engram_246_model_smoke import (
    GPU_MEMORY_UTILIZATION, MODEL_REV, NEXTN, RANKS, SMOKE_ENV,
    SOURCE_REV, check_model_snapshot, check_rank_device,
)


VARIANTS = ("base", "fused", "prefetch", "fused_prefetch")
CONTEXT = 8192
REQUEST_COUNTS = (1, 8)


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def fetch_implementations(buffer):
    from local_engram_246_fused import build_fused_fetch, build_fused_layer_fetch
    from local_engram_246_prefetch import build_layered_fetch

    return {"base": buffer.fetch, "fused": build_fused_fetch(buffer), "prefetch": build_layered_fetch(buffer),
            "fused_prefetch": build_layered_fetch(buffer, build_fused_layer_fetch(buffer))}


def component_checks(buffer, fetches, rank, output, *, preflight=False):
    """Compare actual peer-shard bytes and graph time using identical synthetic IDs."""
    import torch
    import torch.distributed as dist

    def indices_for(tokens, edge_cases):
        generator = torch.Generator().manual_seed(2718 + rank + tokens)
        layers = []
        for rows in buffer.num_entries_per_layer:
            ids = torch.randint(0, RANKS * rows, (tokens, buffer.num_entries_per_token),
                                generator=generator, dtype=torch.int32)
            if edge_cases:
                flat = ids.flatten()
                flat[0] = -1
                flat[1] = RANKS * rows
                for peer in range(RANKS):
                    flat[2 + 2 * peer] = peer * rows
                    flat[3 + 2 * peer] = (peer + 1) * rows - 1
            layers.append(ids)
        return torch.stack(layers).cuda()

    def fetch_all(fetch, ids, aligned=True):
        return [hook() for hook in fetch(ids, use_tma_aligned_col_major_sf=aligned)]

    def exact(expected, actual):
        for left, right in zip(expected, actual, strict=True):
            left = left if isinstance(left, tuple) else (left,)
            right = right if isinstance(right, tuple) else (right,)
            for ref, val in zip(left, right, strict=True):
                if (ref.shape != val.shape or ref.dtype != val.dtype or ref.stride() != val.stride()
                        or not torch.equal(ref.view(torch.uint8) if ref.is_contiguous() else ref,
                                           val.view(torch.uint8) if val.is_contiguous() else val)):
                    return False
        return True

    report = {"dtype": str(buffer.dtype), "num_entries_per_layer": buffer.num_entries_per_layer,
              "entries_per_token": buffer.num_entries_per_token, "hidden": buffer.hidden,
              "byte_parity": {}, "isolated_graph_us": {}}
    for tokens in (4, 5, 32):
        ids = indices_for(tokens, edge_cases=True)
        for aligned in (True, False):
            expected = fetch_all(fetches["base"], ids, aligned)
            key = f"q{tokens}-aligned{int(aligned)}"
            report["byte_parity"][key] = {}
            for variant in VARIANTS[1:]:
                actual = fetch_all(fetches[variant], ids, aligned)
                if not exact(expected, actual):
                    raise RuntimeError(f"Engram byte/layout mismatch: {variant} {key}")
                if preflight and aligned:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                        captured = fetch_all(fetches[variant], ids, aligned)
                    graph.replay()
                    if not exact(expected, captured):
                        raise RuntimeError(f"Engram preflight graph mismatch: {variant} {key}")
                report["byte_parity"][key][variant] = True

    if preflight:
        write_json(output / "preflight.json", report)
        log("preflight_pass", parity=True)
        return

    for tokens in (4, 32):
        ids = indices_for(tokens, edge_cases=False)
        expected = fetch_all(fetches["base"], ids)
        report["isolated_graph_us"][str(tokens)] = {}
        for variant in ("base", "fused"):
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                captured = fetch_all(fetches[variant], ids)
            graph.replay()
            if not exact(expected, captured):
                raise RuntimeError(f"Engram graph replay mismatch: {variant} q{tokens}")
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(12):
                start.record()
                for _ in range(32):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / 32)
            report["isolated_graph_us"][str(tokens)][variant] = samples
            del graph, captured
    write_json(output / f"rank-{rank}-component.json", report)
    log("component_done", rank, parity=True)


def run_rank(dp_size, rank, *, model_dir, output, common):
    from _config import create_llm, setup_common_env

    log("rank_start", rank)
    try:
        setup_common_env(model_name=model_dir, sampler_force_topk=4096,
                         set_override_experts=False, enable_eplb=False)
        llm = create_llm(
            model_dir, rank, ep_size=RANKS, cp_size=1, data_parallel_size=dp_size,
            max_num_seqs=SWEEP_MAX_NUM_SEQS, max_num_batched_tokens=PREFILL_CHUNK,
            max_model_len=SWEEP_MAX_MODEL_LEN, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            lookahead_tokens=1, profile_init=False,
        )

        import torch
        import torch.distributed as dist
        from deepinfer import envs
        from deepinfer.model_executor.parallel_utils.engram_local import LocalEngramBuffer
        from deepinfer.model_executor.parallel_utils.userbuffer import userbuffer

        engine = llm.llm_engine
        worker = engine.driver_worker
        runner = worker.model_runner
        main_graphs, draft_graphs = runner.cuda_graph_runner, runner.dflash_cuda_graph_runner
        buffer = userbuffer.engram
        if not isinstance(buffer, LocalEngramBuffer) or (buffer.rank_idx, buffer.num_ranks) != (rank, RANKS):
            raise RuntimeError("Expected the pinned local EP4 Engram shard")
        if (not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING or envs.DEEPINFER_USE_NEXTN_DECODE != NEXTN
                or envs.DEEPINFER_ENABLE_MTP_VARLEN or not main_graphs.graphs or not draft_graphs.graphs
                or (dist.get_rank(), dist.get_world_size()) != (rank, RANKS)):
            raise RuntimeError("Expected four-rank ZOS, fixed nextn and main/DFlash graphs")
        runtime = {
            "gpu": check_rank_device(torch), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version()), "cache_block_size": engine.cache_config.block_size,
            "graph_batch_sizes": list(main_graphs.batch_sizes),
            "draft_graph_batch_sizes": list(draft_graphs.batch_sizes), "engram_shard_bytes": buffer.num_bytes,
        }
        log("engine_ready", rank, **runtime)

        fetches = fetch_implementations(buffer)
        with torch.inference_mode():
            component_checks(buffer, fetches, rank, output)
        active = "base"

        def activate(variant):
            nonlocal active
            if active == variant:
                return
            if engine.get_num_unfinished_requests() or worker.prev_step_state is not None:
                raise RuntimeError("Cannot recapture while a request is live")
            torch.cuda.synchronize()
            engine.deferred_release.release()
            dist.barrier()
            start = time.monotonic()
            log("recapture_begin", rank, variant=variant)
            buffer.fetch = fetches[variant]
            try:
                # The draft graphs reference the largest main graph's static metadata.
                draft_graphs.destroy()
                runner.capture_model(worker.gpu_cache)
            finally:
                # The variants affect CUDA-graph decode. Eager prefill remains stock.
                del buffer.fetch
            runner.capture_dflash_model(worker.gpu_cache)
            torch.cuda.synchronize()
            dist.barrier()
            active = variant
            log("recapture_done", rank, variant=variant, seconds=time.monotonic() - start)

        def measure(label, requests, mode):
            path = output / label / f"profile-{CONTEXT}-{requests}"
            options = argparse.Namespace(
                mode=mode, context=CONTEXT, requests=requests, output_dir=path / mode,
                reference_dir=path / "clean" if mode == "trace" else None,
                trace_backend="torch" if mode == "trace" else None,
            )
            options.output_dir.mkdir(parents=True, exist_ok=True)
            workload = dict(common, context=CONTEXT, requests_per_rank=requests, variant=label,
                            max_model_len=SWEEP_MAX_MODEL_LEN, engine_max_num_seqs=SWEEP_MAX_NUM_SEQS)
            log("ablation_case", rank, label=label, mode=mode, requests=requests)
            matched = run_case(llm, rank, options, workload, runtime, reuse_engine=True)
            if mode == "clean":
                result_path = options.output_dir / f"rank-{rank}.json"
                result = json.loads(result_path.read_text())
                baseline_path = output / "base" / f"profile-{CONTEXT}-{requests}" / "clean" / result_path.name
                baseline = json.loads(baseline_path.read_text())
                parity_keys = ("input_hashes", "outputs", "window_context_before", "window_context_after",
                               "window_emitted_tokens", "prefill_cached_tokens_by_step")
                matched = all(result[key] == baseline[key] for key in parity_keys)
                result["base_match"] = matched
                write_json(result_path, result)
            agreement = torch.tensor(int(bool(matched)), dtype=torch.int32)
            dist.all_reduce(agreement, op=dist.ReduceOp.MIN)
            if not agreement.item():
                raise RuntimeError(f"{label} {requests=} {mode=} failed token/MTP/workload parity")

        for label in (*VARIANTS, "base_repeat"):
            activate("base" if label == "base_repeat" else label)
            for requests in REQUEST_COUNTS:
                measure(label, requests, "clean")

        # Trace after all clean measurements, while reusing the same loaded model.
        for variant in VARIANTS:
            activate(variant)
            measure(variant, 8, "trace")
        llm.close()
    except Exception as error:
        log("rank_error", rank, error_type=type(error).__name__, error=str(error)[:1000])
        raise
    log("rank_pass", rank)


def check_graphs_before_load(output):
    import torch
    from deepinfer.model_executor.parallel_utils.engram_local import LocalEngramBuffer, LocalEngramLayout

    # Simulate four address ranges on one device to check Triton codegen and
    # capture before loading weights. Real NVLink parity is checked on all ranks.
    buffer = object.__new__(LocalEngramBuffer)
    buffer.device = torch.device("cuda", 0)
    buffer.rank_idx, buffer.num_ranks = 0, RANKS
    buffer.num_entries_per_layer = [257, 311]
    buffer.num_layers = len(buffer.num_entries_per_layer)
    buffer.hidden, buffer.num_max_tokens, buffer.num_entries_per_token = 768, 32, 8
    buffer.num_sf_packs, buffer.use_fp8 = 6, True
    buffer.dtype, buffer.sf_dtype = torch.float8_e4m3fn, torch.int32
    buffer.layout = layout = LocalEngramLayout.build(buffer.num_entries_per_layer, 768, 1, 6)
    buffer._peers = [torch.randint(0, 256, (layout.num_bytes,), dtype=torch.uint8, device=buffer.device)
                     for _ in range(RANKS)]
    buffer._keepalive = None
    buffer._peer_tables = [[buffer._region(peer, offset, n, layout.row_bytes).view(torch.int64)
                            for peer in buffer._peers]
                           for offset, n in zip(layout.table_offsets, buffer.num_entries_per_layer)]
    buffer._peer_sfs = [[buffer._region(peer, offset, n, layout.sf_row_bytes).view(torch.int64)
                        for peer in buffer._peers]
                       for offset, n in zip(layout.sf_offsets, buffer.num_entries_per_layer)]
    buffer._logged_first_fetch = True
    with torch.inference_mode():
        component_checks(buffer, fetch_implementations(buffer), 0, output, preflight=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/opt/DeepInfer"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.preflight:
        check_graphs_before_load(options.output_dir)
        return
    source = options.source_root.resolve(strict=True)
    model = options.model_dir.resolve(strict=True)
    output = options.output_dir.resolve()
    rev = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    if rev != SOURCE_REV:
        raise RuntimeError(f"Expected #246 head {SOURCE_REV}, got {rev}")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "--ignore-submodules=untracked", "HEAD", "--"],
                   check=True, timeout=30)
    if not hasattr(sys, "monitoring"):
        raise RuntimeError("Trace annotation requires Python 3.12+")
    common = {
        "source_rev": rev, "model_rev": MODEL_REV, "model_index_sha256": check_model_snapshot(model),
        "prefill_chunk": PREFILL_CHUNK, "output_tokens": OUTPUT_TOKENS, "warmup_steps": WARMUP_STEPS,
        "clean_steps": CLEAN_STEPS, "ep": RANKS, "dp": RANKS, "cp": 1, "nextn": NEXTN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }
    env = dict(SMOKE_ENV, DEEPINFER_LOGGING_INTERVAL_STEP="0")
    os.environ.update(env)
    sys.path.insert(0, str(source / "examples"))

    import torch
    from _launch import launch_dp

    if torch.cuda.device_count() != RANKS:
        raise RuntimeError("Expected exactly four visible CUDA GPUs")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"Expected an empty output directory: {output}")
    subprocess.run([sys.executable, __file__, "--preflight", "--model-dir", str(model),
                    "--output-dir", str(output)], check=True, timeout=90)
    log("launch", mode="engram_variants", variants=VARIANTS, model_dir=str(model), workload=common, env=env)
    launch_dp(partial(run_rank, model_dir=str(model), output=output, common=common), RANKS)
    log("pass", ranks=RANKS, mode="engram_variants")


if __name__ == "__main__":
    main()
