"""Run a bounded text workload on the fixed DeepInfer #246 checkout and local 1017 weights.

Run under a process-group timeout: DeepInfer's rank launch and shutdown are collective.
This runner emits structured smoke evidence, not benchmark timings.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path


SOURCE_REV = "0a9059c516f9b458f2b271d207a49ac80388763e"
MODEL_REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
MODEL_CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
RANKS = 4
NEXTN = 3
# 1017's candidate index needs a wider logits buffer even for short actual prompts.
# Keep the same capacity for the 8K and 128K profiling workloads.
MAX_MODEL_LEN = 132 * 1024
MAX_BATCHED_TOKENS = 256
MAX_OUTPUT_TOKENS = 16
GPU_MEMORY_UTILIZATION = 0.90
PROMPT = "The sequence of positive integers begins 1, 2, 3, 4,"
SMOKE_ENV = {
    "RANK": "0",
    "WORLD_SIZE": "1",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "23333",
    "DEVICE_OFFSET": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DEEPINFER_ACCELERATOR": "nvidia",
    "DEEPINFER_ENGRAM_LOCAL": "1",
    "EP_DISABLE_GIN": "1",
    "NCCL_IB_DISABLE": "1",
    "DEEPINFER_ZERO_OVERHEAD_SCHEDULING": "1",
    "DEEPINFER_USE_NEXTN_DECODE": str(NEXTN),
    "DEEPINFER_ENABLE_MTP_VARLEN": "0",
    "DEEPINFER_ENFORCE_EAGER": "0",
    "DEEPINFER_DFLASH_ENFORCE_EAGER": "0",
    "DEEPINFER_PROFILE_HTTP_ENABLE": "0",
    "DEEPINFER_BATCH_INVARIANT": "1",
    "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",
    "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
    "DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE": "0",
    "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": "off",
    "DEEPINFER_PREFILL_STORE_EXTRA_STATE": "0",
}


def log(stage, rank=None, **details):
    event = {"stage": stage, "rank": rank, "pid": os.getpid(), **details}
    os.write(1, ("LOCAL_ENGRAM_MODEL " + json.dumps(event) + "\n").encode())


def check_rank_device(torch):
    expected = os.environ.get("DEEPINFER_MODAL_GPU_MODEL", "B200")
    capability = {"B200": (10, 0), "B300": (10, 3)}[expected]
    gpu = torch.cuda.get_device_name()
    if torch.cuda.device_count() != 1 or expected not in gpu or torch.cuda.get_device_capability() != capability:
        raise RuntimeError(f"Expected one {expected} per rank, got {gpu} {torch.cuda.get_device_capability()}")
    return gpu


def check_model_snapshot(model_dir):
    config = (model_dir / "config.json").read_bytes()
    if hashlib.sha256(config).hexdigest() != MODEL_CONFIG_SHA256:
        raise RuntimeError(f"config.json does not match the pinned 1017 revision {MODEL_REV}")
    index = (model_dir / "model.safetensors.index.json").read_bytes()
    shards = set(json.loads(index)["weight_map"].values())
    expected = {f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)}
    if shards != expected:
        raise RuntimeError("The weight index does not describe the expected 48-shard 1017 checkpoint")
    missing = [
        name for name in sorted(shards)
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing or empty checkpoint shards: {missing}")
    return hashlib.sha256(index).hexdigest()


def generate_with_path_counts(llm, sampling_params):
    from deepinfer.model_executor.input_metadata.prepare_tensor_input_zos import prepare_decode_inputs_zos, prepare_decode_v4_zos
    from deepinfer.model_executor.parallel_utils.engram_local import LocalEngramBuffer
    from deepinfer.worker.cuda_graph_runner import CudaGraphRunner

    functions = (prepare_decode_inputs_zos, prepare_decode_v4_zos, LocalEngramBuffer.fetch, CudaGraphRunner.replay)
    names = {function.__code__: function.__qualname__ for function in functions}
    calls = dict.fromkeys(names.values(), 0)

    def on_start(code, _instruction_offset):
        calls[names[code]] += 1

    monitor = sys.monitoring
    tool_id = 3
    # Only these four Python entry points are observed, and only during real generation.
    monitor.use_tool_id(tool_id, "DeepInfer #246 model smoke")
    try:
        monitor.register_callback(tool_id, monitor.events.PY_START, on_start)
        for code in names:
            monitor.set_local_events(tool_id, code, monitor.events.PY_START)
        outputs = llm.generate(PROMPT, sampling_params=sampling_params, use_tqdm=False)
    finally:
        for code in names:
            monitor.set_local_events(tool_id, code, monitor.events.NO_EVENTS)
        monitor.register_callback(tool_id, monitor.events.PY_START, None)
        monitor.free_tool_id(tool_id)
    return outputs, calls


def run_rank(dp_size, dp_rank, *, model_dir):
    from _config import create_llm, setup_common_env

    log("rank_start", dp_rank)
    try:
        setup_common_env(model_name=model_dir, sampler_force_topk=4096, set_override_experts=False, enable_eplb=False)
        llm = create_llm(
            model_dir, dp_rank, ep_size=RANKS, cp_size=1, data_parallel_size=dp_size,
            max_num_seqs=1, max_num_batched_tokens=MAX_BATCHED_TOKENS, max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION, lookahead_tokens=1, profile_init=False,
        )

        import torch
        import torch.distributed as dist
        from deepinfer import envs
        from deepinfer.model_executor.parallel_utils.engram_local import LocalEngramBuffer
        from deepinfer.model_executor.parallel_utils.userbuffer import userbuffer
        from deepinfer.sampling_params import SamplingParams

        engram = userbuffer.engram
        if not isinstance(engram, LocalEngramBuffer) or engram.num_ranks != RANKS or engram.rank_idx != dp_rank:
            raise RuntimeError("Unexpected Engram buffer or Engram rank placement")
        if not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING or envs.DEEPINFER_USE_NEXTN_DECODE != NEXTN:
            raise RuntimeError("Expected ZOS with nextn=3")
        # create_llm must set CUDA_VISIBLE_DEVICES before the first CUDA query in the rank.
        gpu = check_rank_device(torch)
        if (dist.get_rank(), dist.get_world_size()) != (dp_rank, RANKS):
            raise RuntimeError(f"Unexpected rank or device mapping: {gpu}, rank {dist.get_rank()}")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        log("engine_ready", dp_rank, gpu=gpu, visible=os.environ["CUDA_VISIBLE_DEVICES"],
            torch=torch.__version__, free_bytes=free_bytes, total_bytes=total_bytes,
            engram_type=type(engram).__name__, engram_shard_bytes=engram.num_bytes)

        sampling = SamplingParams(temperature=0, max_tokens=MAX_OUTPUT_TOKENS, ignore_eos=True, seq_seed=0)
        outputs, calls = generate_with_path_counts(llm, sampling)
        log("generation_paths", dp_rank, calls=calls)
        if not all(calls.values()):
            raise RuntimeError(f"Generation did not exercise all required paths: {calls}")
        if len(outputs) != 1 or len(outputs[0]) != MAX_OUTPUT_TOKENS or not outputs[0][-1].finished:
            raise RuntimeError("Expected one finished fixed-length output")
        output = outputs[0]
        if not all(math.isfinite(token.token_logprobs) for token in output):
            raise RuntimeError("Non-finite token logprob")
        target_steps = sum(token.draft_tok > 0 for token in output)
        drafted = sum(token.draft_tok for token in output) - target_steps
        accepted = sum(token.accept_tok for token in output) - target_steps
        if drafted <= 0 or not 0 <= accepted <= drafted:
            raise RuntimeError(f"Unexpected MTP counts: {drafted=}, {accepted=}")
        token_ids = [token.token_id for token in output]
        log("generation_ok", dp_rank, token_ids=token_ids, draft_tokens=drafted, accepted_draft_tokens=accepted)
        log("close_begin", dp_rank)
        llm.close()
    except Exception as error:
        log("rank_error", dp_rank, error_type=type(error).__name__, error=str(error)[:1000])
        raise
    log("rank_pass", dp_rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/opt/DeepInfer"))
    parser.add_argument("--model-dir", type=Path, required=True, help="Local snapshot of the fixed 1017 checkpoint")
    args = parser.parse_args()
    if not hasattr(sys, "monitoring"):
        raise RuntimeError("Python 3.12+ is required for generation-path verification")
    source = args.source_root.resolve(strict=True)
    model = args.model_dir.resolve(strict=True)
    rev = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    if rev != SOURCE_REV:
        raise RuntimeError(f"Expected #246 head {SOURCE_REV}, got {rev}")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "--ignore-submodules=untracked", "HEAD", "--"],
                   check=True, timeout=30)
    index_sha256 = check_model_snapshot(model)
    os.environ.update(SMOKE_ENV)
    sys.path.insert(0, str(source / "examples"))

    import torch
    from _launch import launch_dp

    if torch.cuda.device_count() != RANKS:
        raise RuntimeError("Expected exactly four visible CUDA GPUs")
    log("launch", source_rev=rev, model_rev=MODEL_REV, model_index_sha256=index_sha256, model_dir=str(model),
        ep=RANKS, dp=RANKS, cp=1, nextn=NEXTN, max_model_len=MAX_MODEL_LEN, max_batched_tokens=MAX_BATCHED_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, prompt=PROMPT, env=SMOKE_ENV)
    launch_dp(partial(run_rank, model_dir=str(model)), RANKS)
    log("pass", ranks=RANKS)


if __name__ == "__main__":
    main()
