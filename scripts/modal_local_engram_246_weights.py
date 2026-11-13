"""Bounded CPU-only staging of a pinned public 1017 checkpoint, directly into one volume."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import modal

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
VOLUME_NAME = os.getenv("LOCALENGRAM_WEIGHTS_VOLUME", "localengram-246-weights")
MOUNT = "/weights"
SNAPSHOT = f"{MOUNT}/{REV}"

app = modal.App("deepinfer-pr246-weights-cpu")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.2").env({
    "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DOWNLOAD_TIMEOUT": "40", "HF_HUB_ETAG_TIMEOUT": "20", "HF_HOME": "/tmp/hf-metadata",
    "LOCALENGRAM_WEIGHTS_VOLUME": VOLUME_NAME,
})


def checked_manifest(manifest):
    if manifest["id"] != REPO or manifest["sha"] != REV or manifest["private"] or manifest["gated"]:
        raise ValueError("Unexpected repo, revision, or access conditions")
    files = manifest["siblings"]
    names = [row["rfilename"] for row in files]
    if len(set(names)) != 83 or len(names) != 83 or sum(row["size"] for row in files) != 510_311_045_295:
        raise ValueError("Unexpected snapshot manifest")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or ".cache" in p.parts:
            raise ValueError(f"Unsafe checkpoint path: {name}")
    weights = [row for row in files if row["rfilename"].endswith(".safetensors")]
    if len(weights) != 48 or any(len(x.get("lfs", {}).get("sha256", "")) != 64 for x in weights):
        raise ValueError("Expected 48 verifiable weights")
    return files, weights


def fetch_manifest():
    from huggingface_hub import HfApi

    info = HfApi().model_info(REPO, revision=REV, timeout=40, files_metadata=True, token=False)
    siblings = []
    for file in info.siblings:
        entry = {"rfilename": file.rfilename, "size": file.size, "blobId": file.blob_id}
        if file.lfs is not None:
            entry["lfs"] = {"sha256": file.lfs.sha256}
        siblings.append(entry)
    return {"id": info.id, "sha": info.sha, "private": info.private, "gated": info.gated, "siblings": siblings}


def download_large(name, size, target):
    """The HF SDK HTTP path rejects files over its single-file size limit."""
    partial = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{quote(name, safe='/')}"
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            return partial
        if offset > size:
            raise RuntimeError(f"Oversized partial: {name}")
        try:
            request = Request(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"})
            with urlopen(request, timeout=40) as response:
                value = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
                if response.status != 206 or not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
                    raise RuntimeError(f"Server did not honor the pinned byte range for {name}")
                with partial.open("ab") as output:
                    next_report = time.monotonic() + 20
                    while chunk := response.read(8 * 2**20):
                        if output.tell() + len(chunk) > size:
                            raise RuntimeError(f"Oversized response: {name}")
                        output.write(chunk)
                        if time.monotonic() >= next_report:
                            print(f"STAGE progress {name}: {output.tell()}/{size} bytes", flush=True)
                            next_report = time.monotonic() + 20
            if partial.stat().st_size == size:
                return partial
        except (OSError, TimeoutError):
            print(f"STAGE retry {name}, attempt={attempt + 1}", flush=True)
    raise RuntimeError(f"Pinned range download did not finish: {name}")


@app.function(image=image, cpu=2, memory=4096, timeout=14_400, retries=0, max_containers=1,
              scaledown_window=2, volumes={MOUNT: vol})
def stage(sample_only):
    from huggingface_hub import hf_hub_download

    files, weights = checked_manifest(fetch_manifest())
    snapshot = Path(SNAPSHOT)
    snapshot.mkdir(parents=True, exist_ok=True)
    receipts = snapshot.parent / f".{REV}-receipts"
    receipts.mkdir(exist_ok=True)
    start = time.monotonic()

    def stage_one(row):
        name, size = row["rfilename"], row["size"]
        receipt_path = receipts / (hashlib.sha256(name.encode()).hexdigest() + ".json")
        target = snapshot / name
        if receipt_path.is_file() and target.is_file() and not target.is_symlink() and target.stat().st_size == size:
            receipt = json.loads(receipt_path.read_text())
            if receipt["revision"] == REV and receipt["file"] == name and receipt["digest"] == expected_digest(row):
                return {**receipt, "already_verified": True}
        t0 = time.monotonic()
        print(f"STAGE begin {name} ({size} bytes)", flush=True)
        if size > 20 * 2**30:
            fetched = target if target.is_file() else download_large(name, size, target)
        else:
            fetched = Path(hf_hub_download(REPO, name, revision=REV, token=False, local_dir=snapshot))
        if not fetched.is_file() or fetched.is_symlink() or fetched.stat().st_size != size:
            raise RuntimeError(f"Unexpected downloaded path or size: {name}")
        seconds_download = time.monotonic() - t0
        with fetched.open("rb") as f:
            if row.get("lfs"):
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            else:
                sha1 = hashlib.sha1(f"blob {size}\0".encode())
                while data := f.read(1024 * 1024):
                    sha1.update(data)
                digest = sha1.hexdigest()
        if digest != expected_digest(row):
            if fetched in (target, target.with_name(target.name + ".part")):
                fetched.unlink()
                receipt_path.unlink(missing_ok=True)
            raise RuntimeError(f"Hugging Face digest mismatch: {name}")
        if fetched != target:
            if fetched != target.with_name(target.name + ".part"):
                raise RuntimeError(f"Unexpected source path: {name}")
            fetched.replace(target)
        receipt = {"file": name, "revision": REV, "bytes": size, "digest": digest,
                   "download_s": seconds_download, "verified_s": time.monotonic() - t0}
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt) + "\n")
        temporary.replace(receipt_path)
        print(f"STAGE verified {name} in {receipt['verified_s']:.1f}s", flush=True)
        return receipt

    if sample_only:
        # This is the smallest actual weight shard and is reusable in a complete run.
        selected = [min(weights, key=lambda row: row["size"])]
        results = [stage_one(selected[0])]
        vol.commit()
    else:
        selected = files
        ordinary = [row for row in files if row["size"] <= 20 * 2**30]
        oversized = [row for row in files if row["size"] > 20 * 2**30]
        results = []
        # Bound uncommitted bytes; don't run both ~95 GiB shards concurrently.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start_at in range(0, len(ordinary), 4):
                results.extend(pool.map(stage_one, ordinary[start_at:start_at + 4]))
                vol.commit()
                print(f"STAGE committed {len(results)}/{len(files)} files; {sum(x['bytes'] for x in results)} bytes", flush=True)
        for row in oversized:
            try:
                results.append(stage_one(row))
            finally:
                # Preserve a closed, partial file when the transport fails.
                vol.commit()
            print(f"STAGE committed {len(results)}/{len(files)} files; {sum(x['bytes'] for x in results)} bytes", flush=True)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"].values()) == {row["rfilename"] for row in weights}
        readiness = {"repo": REPO, "revision": REV, "files": len(results), "verified_bytes": sum(x["bytes"] for x in results)}
        ready = snapshot / ".ready.json"
        ready.write_text(json.dumps(readiness, indent=2) + "\n")
        vol.commit()

    return {"repo": REPO, "revision": REV, "volume": VOLUME_NAME, "snapshot": SNAPSHOT,
            "sample_only": sample_only, "files": results, "count": len(selected), "elapsed_s": time.monotonic() - start}


def expected_digest(row):
    return row["lfs"]["sha256"] if row.get("lfs") else row["blobId"]


@app.local_entrypoint()
def main(full: bool = False):
    result = stage.remote(not full)
    print(json.dumps({key: result[key] for key in ("revision", "volume", "snapshot", "count", "elapsed_s")}),
          flush=True)
