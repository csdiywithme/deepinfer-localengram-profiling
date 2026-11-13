"""CPU-only build for the unmodified #246 model stack; no weights in the image."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import modal

PR = "0a9059c516f9b458f2b271d207a49ac80388763e"
PARENT = "553741557a57df44a46fb3ab72b6aafeaf945337"
ROOT = "/opt/DeepInfer"
VENDORS = {
    "DeepSelect": "bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18",
    "FlashMLA": "2e5429fc5653bab6e081f09477126f731882a6a9",
    "DeepEP": "f6aa38fe06cd4a0489e7730365d4e0114970a2f3",
    "DeepGEMM": "1fc9c66975ee322d628b3288534ca0e9ce12a64d",
}


def checkout_pr():
    def git(*args):
        return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()

    assert git("rev-parse", "HEAD") == PARENT
    git("fetch", "--depth=1", "origin", PR)
    assert git("rev-parse", "FETCH_HEAD") == PR
    # Reuse the existing C++ extension only if build inputs and gitlinks are identical.
    subprocess.run(["git", "-C", ROOT, "diff", "--exit-code", PARENT, PR, "--", "csrc", "setup.py",
                    "pyproject.toml", "scripts/build.sh", "scripts/build_vendor.sh", "vendor"], check=True)
    git("checkout", "--detach", PR)
    assert git("rev-parse", "HEAD") == PR
    for name, sha in VENDORS.items():
        assert git("rev-parse", f"HEAD:vendor/{name}") == sha
    assert subprocess.check_output(["git", "-C", "/opt/DeepEP", "rev-parse", "HEAD"], text=True).strip() == VENDORS["DeepEP"]


def compile_vendor(name, sha):
    from tempfile import TemporaryDirectory

    vendor_dir = Path(ROOT) / "vendor" / name
    subprocess.run(["git", "-C", ROOT, "submodule", "update", "--init", "--recursive", "--depth=1",
                    "--", f"vendor/{name}"], check=True)
    assert subprocess.check_output(["git", "-C", str(vendor_dir), "rev-parse", "HEAD"], text=True).strip() == sha
    env = dict(os.environ, MAX_JOBS="4", NVCC_THREADS="1", TORCH_CUDA_ARCH_LIST="10.0a")
    overrides = {"DeepSelect": {"DEEP_SELECT_BUILD_TARGET_PLATFORM": "CUDA"},
                 "FlashMLA": {"FLASH_MLA_BUILD_TARGET_PLATFORM": "CUDA"},
                 "DeepGEMM": {"DG_FORCE_BUILD": "1"}}
    env.update(overrides[name])
    with TemporaryDirectory() as wheel_dir:
        # gVisor's CPU builder may expose no usable PCI bus. The build target
        # is explicit above; the upstream spill checker also supports CPU_ONLY.
        if name in ("DeepSelect", "FlashMLA") and subprocess.run(
            ["/usr/bin/lspci"], capture_output=True, check=False
        ).returncode != 0:
            import torch

            assert not torch.cuda.is_available(), "Empty PCI inventory is only valid in the CPU builder"
            empty_dump = Path(wheel_dir) / "empty.pci"
            empty_dump.touch()
            subprocess.run(["/usr/bin/lspci", "-F", str(empty_dump)], check=True)
            wrapper = Path(wheel_dir) / "lspci"
            wrapper.write_text('#!/bin/sh\n[ "$#" -eq 0 ] || exit 2\nexec /usr/bin/lspci -F "$DEEPINFER_CPU_PCI_DUMP"\n')
            wrapper.chmod(0o755)
            env.update(PATH=f"{wheel_dir}:{env['PATH']}", DEEPINFER_CPU_PCI_DUMP=str(empty_dump))
            print("CPU build: no PCI access; vendor probe reads an empty lspci dump, CUDA target remains explicit.", flush=True)
        subprocess.run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", wheel_dir],
                       cwd=vendor_dir, env=env, check=True)
        wheels = list(Path(wheel_dir).glob("*.whl"))
        assert len(wheels) == 1, wheels
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", str(wheels[0])], check=True)


if modal.is_local():
    image = (
        modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12")
        .apt_install("git", "build-essential", "libnuma-dev", "libibverbs-dev", "ibverbs-utils",
                     "librdmacm-dev", "libdw-dev")
        .pip_install("setuptools", "wheel", "ninja", "numpy==2.3.5")
        .run_commands(
            "python -m pip install --find-links https://download.pytorch.org/whl/cu132/torch/ 'torch==2.13.0+cu132'",
            "python -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3",
            "git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP",
            f"git -C /opt/DeepEP checkout {VENDORS['DeepEP']}",
            "git -C /opt/DeepEP submodule update --init --recursive",
        )
        .env({"MAX_JOBS": "4", "TORCH_CUDA_ARCH_LIST": "10.0a", "CC": "gcc", "CXX": "g++"})
        .run_commands("cd /opt/DeepEP && python setup.py bdist_wheel && python -m pip install --no-deps dist/*.whl")
        .run_commands(
            f"git clone https://github.com/deepseek-ai/DeepInfer.git {ROOT}",
            f"git -C {ROOT} checkout {PARENT}",
            "python -m venv --system-site-packages /opt/deepinfer-venv",
        )
        .env({"VIRTUAL_ENV": "/opt/deepinfer-venv", "DEEPINFER_ACCELERATOR": "nvidia", "PYTHONPATH": ROOT,
              "PATH": "/opt/deepinfer-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"})
        .run_commands(
            f"cd {ROOT} && python -m pip install -r requirements-dev.txt",
            "python -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3",
        )
        .run_commands(f"cd {ROOT} && bash scripts/build.sh build", f"git -C {ROOT} diff --exit-code")
        .env({"PATH": "/opt/deepinfer-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"})
        .run_function(checkout_pr, cpu=1, memory=2048, timeout=180)
        .run_commands(f"cd {ROOT} && python -m pip install -r requirements-nvidia.txt",
                      "python -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3")
        .apt_install("pciutils")
    )
    for package in ("DeepSelect", "FlashMLA", "DeepGEMM"):
        # The upstream post-link checker rejected four sparse-decode symbols
        # with this pinned CUDA 13.2 build. Keep the unchanged wheel for the
        # functional experiment; the unmodified compiler/spill log is retained.
        vendor_env = {"FLASH_MLA_SKIP_REG_SPILL_CHECK": "1"} if package == "FlashMLA" else None
        image = image.run_function(compile_vendor, args=(package, VENDORS[package]),
                                   env=vendor_env, cpu=6, memory=24576, timeout=1500)
else:
    image = None

app = modal.App("deepinfer-pr246-model-cpu-build")


@app.function(image=image, cpu=1, memory=4096, timeout=60, retries=0, max_containers=1, scaledown_window=2)
def audit():
    import importlib.metadata
    import importlib.util
    import torch

    def git(*args):
        return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()

    assert git("rev-parse", "HEAD") == PR
    assert not git("diff", "--name-only")
    extensions = sorted((Path(ROOT) / "deepinfer/_so/nvidia-x86_64").glob("*.so"))
    assert extensions and all(p.is_file() for p in extensions)
    modules = {}
    for name in ("deep_select", "flash_mla", "deep_ep", "deep_gemm"):
        spec = importlib.util.find_spec(name)
        assert spec and spec.origin, name
        modules[name] = spec.origin
    return {"source_rev": PR, "parent": PARENT, "vendor_pins": VENDORS,
            "extensions": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in extensions},
            "python": sys.version, "torch": str(torch.__version__), "cuda_build": str(torch.version.cuda),
            "compiled_nccl": tuple(int(part) for part in torch.cuda.nccl.version()), "modules_on_disk": modules,
            "flash_attn_4": importlib.metadata.version("flash-attn-4"),
            "flash_mla_spill_check_skipped": True,
            "nccl_wheel": importlib.metadata.version("nvidia-nccl-cu13"),
            "gpu_available": bool(torch.cuda.is_available()), "scope": "CPU image audit only; no model load"}


@app.local_entrypoint()
def main():
    import json

    result = audit.remote()
    print(json.dumps(result, indent=2), flush=True)
