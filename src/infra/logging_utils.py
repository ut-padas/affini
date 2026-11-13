"""Centralized logging configuration for the knnsur project."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from tqdm import tqdm

from src import _get_mpi_rank, _get_mpi_size

logger = logging.getLogger(__name__)


def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str = ".logs",
    console_output: bool = True,
    file_output: bool = True,
    suppress_external: bool = True,
    per_rank_files: bool = True,
) -> None:
    """Configure logging for the entire application.

    This function should be called once at the application entry point. It sets up
    logging to output to both console and timestamped log files in the specified
    directory. All loggers created with logging.getLogger(__name__) will
    automatically use this configuration.

    MPI behaviour:
        - File output defaults to rank 0 only (single clean log).
        - Set ``per_rank_files=True`` to write one log file per MPI rank
          (e.g. ``rank_3_20260217_190000.log``) which is useful for debugging
          worker-side issues.

    Args:
        console_level: Logging level for console output.
        file_level: Logging level for file output. Default: DEBUG (captures everything).
        log_dir: Directory where log files will be stored.
        console_output: Whether to log to console.
        file_output: Whether to log to file.
        suppress_external: Whether to suppress logs from external libraries. Default: True.
        per_rank_files: If True, every MPI rank writes its own timestamped log file
            (e.g. ``rank_3_20260217_190000.log``). Useful for debugging non-rank-0
            workers. If False (default), only rank 0 writes a log file.

    Example:
        >>> from src.logging_config import setup_logging
        >>> setup_logging(level=logging.DEBUG)
        >>>
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")

    Notes:
        - Log files are named with timestamp: YYYYMMDD_HHMMSS.log
        - The .logs directory is created automatically if it doesn't exist
        - Calling this function multiple times will reset the logging configuration
        - All modules using logging.getLogger(__name__) will share this config
    """
    # Default format: timestamp - logger name - level - message
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    mpi_rank = _get_mpi_rank()
    mpi_size = _get_mpi_size()
    _is_rank0 = mpi_rank == 0

    # Get the root logger (parent of all loggers)
    root_logger = logging.getLogger()

    # Clear any existing handlers (in case setup_logging is called multiple times)
    root_logger.handlers.clear()

    # Set root logger to the minimum level needed (so handlers can filter independently)
    root_logger.setLevel(min(console_level, file_level))

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File Handler:
    #   - Default (per_rank_files=False): single file written by rank 0 only.
    #   - per_rank_files=True: each rank writes its own file, named rank_N_<timestamp>.log.
    _write_file = file_output and (_is_rank0 or per_rank_files)

    if _write_file:
        # Create .logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if per_rank_files and mpi_size >= 2:
            log_filename = log_path / f"rank_{mpi_rank:02}_{timestamp}.log"
        else:
            log_filename = log_path / f"{timestamp}.log"

        file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        if _is_rank0:
            root_logger.info(f"Logging to file: {log_filename.absolute()}")
        else:
            root_logger.info(
                f"Rank {mpi_rank:02}: logging to file: {log_filename.absolute()}"
            )

    # Suppress external library logs
    if suppress_external:
        external_loggers = [
            "absl",
            "cupy",
            "faiss",
            "jax",
            "jaxlib",
            "matplotlib",
            "numba",
            "numpy",
            "pandas",
            "PIL",
            "scipy",
            "sklearn",
            "tensorflow",
            "torch",
        ]
        for lib_name in external_loggers:
            logging.getLogger(lib_name).setLevel(logging.WARNING)


def log_mem_stats(tag: str = ""):
    """Log current GPU and system RAM memory usage."""
    # Get system RAM usage
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / 1e9
    ram_total_gb = ram.total / 1e9
    try:
        import pynvml

        # Get GPU VRAM usage
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used_gb = int(info.used) / 1e9
        vram_total_gb = int(info.total) / 1e9
        pynvml.nvmlShutdown()

        msg = f"[MEM] {tag:<25} | VRAM: {vram_used_gb:6.2f} / {vram_total_gb:6.2f} GB | RAM: {ram_used_gb:6.2f} / {ram_total_gb:6.2f} GB"
    except Exception:
        msg = f"[MEM] {tag:<25} | RAM: {ram_used_gb:6.2f} / {ram_total_gb:6.2f} GB"

    logger.debug(msg)


class EvaluationCapture:
    """Captures function evaluations and saves them to NPZ files."""

    def __init__(
        self, func, *func_args, save_dir: Path, prefix: str = "eval", **func_kwargs
    ):
        """
        Initialize the evaluation capture wrapper.

        Args:
            func: The function to wrap
            save_dir: Directory where NPZ files will be saved
            prefix: Prefix for the NPZ filenames
            *func_args, **func_kwargs: Arguments passed to the function
        """
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.save_dir = Path(save_dir)
        self.prefix = prefix

        # Create directory if needed
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, X):
        """Call function and save X, result to timestamped NPZ file."""
        import numpy as np

        from src import xp

        # Call the original function
        result = self.func(X, *self.func_args, **self.func_kwargs)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]  # Include milliseconds
        filename = f"{self.prefix}-{timestamp}.npz"
        filepath = self.save_dir / filename

        # Convert to numpy arrays for saving
        X_np = X.get() if hasattr(X, "get") else np.asarray(X)
        result_np = result.get() if hasattr(result, "get") else np.asarray(result)

        # Save just X and result
        np.savez_compressed(filepath, X=X_np, result=result_np)
        logger.debug(f"Saved evaluation: {filename}")

        # Convert back to original array type
        result = xp.asarray(result_np)
        return result


def capture_evaluations(
    func, *func_args, save_dir: Path, prefix: str = "eval", **func_kwargs
):
    """
    Create a function wrapper that captures evaluations to NPZ files.

    Args:
        func: Function to wrap
        save_dir: Directory to save NPZ files
        prefix: Filename prefix
        *func_args, **func_kwargs: Arguments for the function

    Returns:
        Wrapped function that saves X, result to timestamped NPZ files

    Example:
        >>> f_logged = capture_evaluations(rosenbrock, a=100.0,
        ...                               save_dir=Path("./evals"), prefix="rosenbrock")
        >>> result = f_logged(X)  # Saves X, result to rosenbrock-HH-MM-SS-fff.npz
    """
    return EvaluationCapture(
        func, *func_args, save_dir=save_dir, prefix=prefix, **func_kwargs
    )


def consolidate_evaluations(
    save_dir: Path,
    prefix: str,
    output_path: Path,
    remove_duplicates: bool = True,
    delete_individual_files_after: bool = False,
) -> Optional[Path]:
    """
    Consolidate multiple NPZ evaluation files into a single file.

    Args:
        save_dir: Directory containing NPZ files
        prefix: Filename prefix to identify relevant NPZ files
        output_path: Path to save the consolidated NPZ file
        remove_duplicates: Whether to remove duplicate evaluations
        delete_individual_files_after: Whether to delete individual NPZ files after consolidation

    Returns:
        Optional[Path]: Path to the consolidated NPZ file or None if no files found

    Example:
        >>> # After capturing many evaluations...
        >>> consolidated = consolidate_evaluations(Path("./evals"), "rosenbrock")
        >>> data = np.load(consolidated)
        >>> print(f"Total points: {data['X'].shape[0]}")
    """
    import numpy as np

    save_dir = Path(save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"Directory not found: {save_dir}")

    # Find all NPZ files with the given prefix
    npz_files = list(save_dir.glob(f"{prefix}-*.npz"))
    if not npz_files:
        logger.warning(f"No NPZ files found with prefix '{prefix}' in {save_dir}")
        return None

    logger.info(f"Consolidating {len(npz_files)} files with prefix '{prefix}'")
    all_X = []
    all_results = []

    # Determine expected result dimensionality from first file
    first_data = np.load(npz_files[0])
    expected_result_ndim = first_data["result"].ndim
    logger.debug(f"Expected result dimensionality: {expected_result_ndim}D")

    for npz_file in tqdm(npz_files, desc=f"Loading {prefix} files"):
        data = np.load(npz_file)
        all_X.append(data["X"])
        all_results.append(data["result"])

        assert data["X"].ndim == 2, f"X array in {npz_file} is not 2D"
        assert data["result"].ndim == expected_result_ndim, (
            f"result array in {npz_file} has {data['result'].ndim}D but expected {expected_result_ndim}D"
        )

    # Concatenate all data
    X_concat = np.vstack(all_X)
    results_concat = np.vstack(all_results)

    if remove_duplicates:
        if expected_result_ndim == 2:
            # For 2D results (function evaluations): can use hstack directly
            combined = np.hstack((X_concat, results_concat))
            unique_combined = np.unique(combined, axis=0)
            X_concat = unique_combined[:, : X_concat.shape[1]]
            results_concat = unique_combined[:, X_concat.shape[1] :]
        elif expected_result_ndim == 3:
            # For 3D results (gradients): flatten results for duplicate detection
            results_flat = results_concat.reshape(results_concat.shape[0], -1)
            combined = np.hstack((X_concat, results_flat))
            unique_combined = np.unique(combined, axis=0)
            X_concat = unique_combined[:, : X_concat.shape[1]]
            results_flat = unique_combined[:, X_concat.shape[1] :]
            # Reshape results back to 3D
            results_concat = results_flat.reshape(
                -1, results_concat.shape[1], results_concat.shape[2]
            )
        else:
            raise ValueError(
                f"Unsupported result dimensionality: {expected_result_ndim}D"
            )

    # Determine output filename
    np.savez_compressed(output_path, X=X_concat, result=results_concat)
    logger.info(f"Consolidated file saved: {output_path}")

    if delete_individual_files_after:
        for npz_file in npz_files:
            npz_file.unlink()
        logger.info(
            f"Deleted individual NPZ files with prefix '{prefix}' in {save_dir}"
        )

    return output_path
