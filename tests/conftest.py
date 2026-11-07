"""Shared pytest configuration."""


def pytest_report_header(config):
    """Show the detected backends at the top of every test run, so a passing
    run also documents *what* it exercised (CPU vs GPU, serial vs MPI)."""
    try:
        from src import backend

        info = backend.backend_info()
    except Exception as e:  # e.g. AFINNI_MPI=on without MPI: let the tests report it
        return [f"afinni backend: detection failed - {type(e).__name__}: {e}"]

    gpu = f"CuPy ({info.get('gpu_name', info['gpu'])})" if info["is_cupy_backend"] else "NumPy"
    return [
        f"afinni backend: array={gpu} "
        f"| MPI={info['mpi']} | {info['platform']}, Python {info['python']}",
        f"afinni overrides: AFINNI_DEVICE={info['device_mode']} AFINNI_MPI={info['mpi_mode']}",
    ]
