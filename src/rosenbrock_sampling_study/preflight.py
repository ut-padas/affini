"""Read-only preflight checks for a Vista idev session."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys

from mpi4py import MPI

from src import IS_CUPY_BACKEND
from src.surrogate import AFINNISurrogate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)

    print("Rosenbrock study preflight")
    print(f"  host: {platform.node()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  MPI ranks: {MPI.COMM_WORLD.Get_size()}")
    print(f"  CuPy backend: {IS_CUPY_BACKEND}")
    print(f"  AFINNI import: {AFINNISurrogate.__name__}")

    if MPI.COMM_WORLD.Get_size() != 1:
        print("ERROR: this study driver currently expects exactly one MPI rank.")
        return 2

    if IS_CUPY_BACKEND:
        import cupy as cp

        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        probe = cp.arange(1024, dtype=cp.float64)
        probe_sum = float(cp.sum(probe).get())
        print(f"  GPU: {name}")
        print(f"  GPU memory free/total: {free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB")
        print(f"  GPU arithmetic probe: {probe_sum:.1f}")
    elif args.require_gpu:
        print("ERROR: CuPy did not initialize, so the GH200 would not be used.")
        print("Check that the uv environment includes the `gpu` dependency group.")
        return 3

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"  nvidia-smi: {result.stdout.strip()}")
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
