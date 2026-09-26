"""Process-level setup: UTF-8 stdio, CUDA requirement, optional-dependency report.

Called once from the CLI entry point rather than at import time, so feature
worker processes don't repeat the CUDA check or the startup messages.
"""
import sys


def configure_stdio():
    # Jupyter's OutStream has no reconfigure(); notebook output is already UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def check_cuda():
    """Fail fast unless CuPy can see a CUDA device (XGBoost/SBERT run on cuda)."""
    try:
        import cupy as cp
        gpu_ok = cp.cuda.is_available()
    except Exception as exc:
        raise RuntimeError(
            "CUDA is required. Install dependencies from requirements.txt, "
            "including cupy-cuda12x."
        ) from exc
    if not gpu_ok:
        raise RuntimeError(
            "CUDA is required but no GPU device is available. "
            "Check the NVIDIA driver and CUDA installation."
        )
    print(f"[GPU] CUDA device: {_cuda_device_name(cp)}")


def _cuda_device_name(cp) -> str:
    """Device name across CuPy versions (CuPy 14 dropped ``Device.name``)."""
    device = cp.cuda.Device()
    name = getattr(device, "name", None)
    if not name:
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props.get("name", "CUDA device") if isinstance(props, dict) else "CUDA device"
    return name.decode() if isinstance(name, bytes) else str(name)


def report_optional_dependencies():
    from .blocking import HNSW_AVAILABLE, LSH_AVAILABLE
    from .features import PHONETIC_AVAILABLE

    if HNSW_AVAILABLE:
        print("[HNSW] sentence-transformers + hnswlib available")
    else:
        print("[HNSW] Not available (pip install sentence-transformers hnswlib) — skipping dense retrieval")
    if LSH_AVAILABLE:
        print("[LSH] datasketch available")
    else:
        print("[LSH] Not available (pip install datasketch) — skipping MinHash/LSH")
    if PHONETIC_AVAILABLE:
        print("[Phonetic] jellyfish available")
    else:
        print("[Phonetic] Not available (pip install jellyfish) — phonetic features disabled")
