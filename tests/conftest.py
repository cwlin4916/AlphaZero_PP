import os
import sys
from pathlib import Path

# Add archive/ to the import path so legacy nsai_experiments tests still work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "archive"))


def pytest_sessionstart(session):
    # Disable NumPy multithreading (must be set before importing NumPy).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    # Deterministic CUDA (must be set before importing PyTorch).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
