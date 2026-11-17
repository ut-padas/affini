"""Shared pytest configuration."""


def pytest_report_header(config):
    """Show the detected backends at the top of every test run, so a passing
    run also documents *what* it exercised (CPU vs GPU, serial vs MPI)."""
    try:
        from src.infra import backend

        info = backend.backend_info()
    except Exception as e:  # e.g. SURROGATE_MPI=on without MPI: let the tests report it
        return [f"surrogate backend: detection failed - {type(e).__name__}: {e}"]

    gpu = f"CuPy ({info.get('gpu_name', info['gpu'])})" if info["is_cupy_backend"] else "NumPy"
    return [
        f"surrogate backend: array={gpu} "
        f"| MPI={info['mpi']} | {info['platform']}, Python {info['python']}",
        f"surrogate overrides: SURROGATE_DEVICE={info['device_mode']} SURROGATE_MPI={info['mpi_mode']}",
    ]
