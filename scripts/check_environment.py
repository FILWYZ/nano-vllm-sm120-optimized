"""Print a machine-readable readiness report for a local CUDA/WSL install."""
import json, os, platform, shutil, subprocess, sys
from pathlib import Path

def command_output(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def main():
    report = {"platform": platform.platform(), "python": sys.version.split()[0],
              "cuda_home": os.environ.get("CUDA_HOME") or "/usr/local/cuda-12.8", "nvcc": shutil.which("nvcc"),
              "nvcc_version": None, "nvidia_smi": shutil.which("nvidia-smi"),
              "gpu": None, "torch": None, "cuda_available": False, "flashinfer": None,
              "git_commit": command_output(["git", "rev-parse", "HEAD"])}
    nvcc = report["nvcc"]
    if not nvcc and report["cuda_home"]:
        candidate = Path(report["cuda_home"]) / "bin/nvcc"
        if candidate.exists():
            nvcc = str(candidate)
            report["nvcc"] = nvcc
    if nvcc:
        report["nvcc_version"] = command_output([nvcc, "--version"])
    if report["nvidia_smi"]:
        report["gpu"] = command_output([
            "nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.used,driver_version",
            "--format=csv,noheader"])
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available()}
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            free, total = torch.cuda.mem_get_info()
            report["torch"].update({"device": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "free_gib": round(free / 2**30, 3), "total_gib": round(total / 2**30, 3)})
    except ImportError as exc:
        report["torch"] = {"error": str(exc)}
    try:
        import flashinfer
        report["flashinfer"] = getattr(flashinfer, "__version__", "installed")
    except ImportError:
        report["flashinfer"] = "not installed (SDPA fallback remains available)"
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
