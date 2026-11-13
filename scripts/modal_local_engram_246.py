"""Run the pinned full-model smoke or ablation on one four-B300 Modal allocation."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import modal

from local_engram_246_model_smoke import MODEL_REV, RANKS


HERE = Path(__file__).resolve().parent
REMOTE = "/opt/local-engram-246"
WEIGHTS_VOLUME = os.getenv("LOCALENGRAM_WEIGHTS_VOLUME", "localengram-246-weights")
RESULTS_VOLUME = os.getenv("LOCALENGRAM_RESULTS_VOLUME", "localengram-246-results")
MODEL_PATH = Path("/weights") / MODEL_REV
READY = {"repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "revision": MODEL_REV,
         "files": 83, "verified_bytes": 510_311_045_295}
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True, version=2)

if modal.is_local():
    from modal_local_engram_246_model_image import image

    image = image.add_local_python_source("local_engram_246_model_smoke")
    for name in ("model_smoke", "model_profile", "variants", "fused", "prefetch"):
        filename = f"local_engram_246_{name}.py"
        image = image.add_local_file(HERE / filename, f"{REMOTE}/{filename}")
else:
    image = None

app = modal.App("deepinfer-localengram-profiling")


def require_ready(volume):
    try:
        ready = json.loads(b"".join(volume.read_file(f"/{MODEL_REV}/.ready.json")))
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError("Stage the pinned model with modal_local_engram_246_weights.py --full first") from error
    if ready != READY:
        raise RuntimeError("The weights volume does not contain the verified model snapshot")


def smoke_passed(log):
    events = []
    decoder = json.JSONDecoder()
    for line in log.splitlines():
        for fragment in line.split("LOCAL_ENGRAM_MODEL ")[1:]:
            try:
                event, _ = decoder.raw_decode(fragment.lstrip())
            except json.JSONDecodeError:
                continue
            events.append(event)
    ranks = {event["rank"] for event in events if event["stage"] == "rank_pass"}
    generations = {event["rank"]: tuple(event["token_ids"]) for event in events
                   if event["stage"] == "generation_ok"}
    return (ranks == set(range(RANKS)) and set(generations) == ranks and len(set(generations.values())) == 1
            and any(event["stage"] == "pass" for event in events))


@app.function(image=image, gpu="B300:4", cpu=8, memory=262144, timeout=2700,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/weights": weights.with_mount_options(read_only=True), "/artifacts": results})
def run(run_id, mode):
    if json.loads((MODEL_PATH / ".ready.json").read_text()) != READY:
        raise RuntimeError("The weights volume does not contain the verified model snapshot")
    output = Path("/artifacts") / run_id
    output.mkdir(exist_ok=False)
    entry = "model_smoke" if mode == "smoke" else "variants"
    command = [sys.executable, "-u", f"{REMOTE}/local_engram_246_{entry}.py", "--source-root", "/opt/DeepInfer",
               "--model-dir", str(MODEL_PATH)]
    if mode == "ablation":
        command += ["--output-dir", str(output / "ablation")]

    log_path = output / "worker.log"
    start = time.monotonic()
    timed_out = False
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd="/opt/DeepInfer", stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=dict(os.environ, DEEPINFER_MODAL_GPU_MODEL="B300"))
        try:
            proc.wait(timeout=1230 if mode == "smoke" else 2480)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

    passed = proc.returncode == 0 and not timed_out
    if mode == "smoke":
        passed = passed and smoke_passed(log_path.read_text())
    status = {"run_id": run_id, "mode": mode, "passed": passed, "returncode": proc.returncode,
              "timed_out": timed_out, "elapsed_s": time.monotonic() - start}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    results.commit()
    return status


@app.local_entrypoint()
def main(mode: str = "ablation"):
    if mode not in {"smoke", "ablation"}:
        raise ValueError("Select smoke or ablation")
    require_ready(weights)
    run_id = uuid4().hex
    print(f"Results will be stored at {RESULTS_VOLUME}:/{run_id}", flush=True)
    status = run.remote(run_id, mode)
    print(json.dumps(status, indent=2), flush=True)
    if not status["passed"]:
        raise RuntimeError(f"{mode} failed. Inspect {RESULTS_VOLUME}:/{run_id}/worker.log")
