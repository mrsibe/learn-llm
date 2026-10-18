"""训练和 Benchmark 共用的实验环境记录，不读取凭据或环境变量。"""

import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch


def environment_info(device):
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    info = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ("torch", "numpy", "datasets", "tokenizers")
        },
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": device,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": gpu.name,
            "memory_bytes": gpu.total_memory,
            "capability": [gpu.major, gpu.minor],
        }
    return info
