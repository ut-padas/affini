"""Centralized logging configuration for the knnsur project."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import pynvml

logger = logging.getLogger(__name__)


def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str = ".logs",
    console_output: bool = True,
    file_output: bool = True,
    suppress_external: bool = True,
) -> None:
    """Configure logging for the entire application.

    This function should be called once at the application entry point. It sets up
    logging to output to both console and timestamped log files in the specified
    directory. All loggers created with logging.getLogger(__name__) will
    automatically use this configuration.

    Args:
        console_level: Logging level for console output.
        file_level: Logging level for file output. Default: DEBUG (captures everything).
        log_dir: Directory where log files will be stored.
        console_output: Whether to log to console.
        file_output: Whether to log to file.
        suppress_external: Whether to suppress logs from external libraries. Default: True.

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

    # Get the root logger (parent of all loggers)
    root_logger = logging.getLogger()

    # Clear any existing handlers (in case setup_logging is called multiple times)
    root_logger.handlers.clear()

    # Set root logger to the minimum level needed (so handlers can filter independently)
    root_logger.setLevel(min(console_level, file_level))

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler (StreamHandler)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File Handler (with timestamp)
    if file_output:
        # Create .logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = log_path / f"{timestamp}.log"

        file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log the location of the log file
        root_logger.info(f"Logging to file: {log_filename.absolute()}")

    # Suppress external library logs
    if suppress_external:
        external_loggers = [
            "matplotlib",
            "PIL",
            "numba",
            "cupy",
            "numpy",
            "scipy",
            "sklearn",
            "torch",
            "tensorflow",
            "pandas",
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

        logger.info(
            f"Capturing evaluations for {func.__name__} -> {self.save_dir}/{prefix}-*.npz"
        )

    def __call__(self, X):
        """Call function and save X, result to timestamped NPZ file."""
        import numpy as np

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
    output_name: Optional[str] = None,
    remove_duplicates: bool = True,
) -> Path:
    """
    Consolidate multiple NPZ evaluation files into a single file.

    Args:
        save_dir: Directory containing the NPZ files
        prefix: Prefix to filter files (e.g., "rosenbrock")
        output_name: Name for consolidated file (default: "{prefix}_consolidated.npz")
        remove_duplicates: Whether to remove duplicate (X, result) pairs

    Returns:
        Path to the consolidated NPZ file

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
        raise FileNotFoundError(
            f"No NPZ files found with prefix '{prefix}' in {save_dir}"
        )

    logger.info(f"Consolidating {len(npz_files)} files with prefix '{prefix}'")
    all_X = []
    all_results = []

    for npz_file in npz_files:
        data = np.load(npz_file)
        all_X.append(data["X"])
        all_results.append(data["result"])

    # Concatenate all data
    X_concat = np.vstack(all_X)
    results_concat = np.vstack(all_results)

    if remove_duplicates:
        # Remove duplicate rows based on (X, result) pairs
        combined = np.hstack((X_concat, results_concat))
        unique_combined = np.unique(combined, axis=0)
        X_concat = unique_combined[:, : X_concat.shape[1]]
        results_concat = unique_combined[:, X_concat.shape[1] :]

    # Determine output filename
    if output_name is None:
        output_name = f"{prefix}_consolidated.npz"
    output_path = save_dir / output_name
    np.savez_compressed(output_path, X=X_concat, result=results_concat)
    logger.info(f"Consolidated file saved: {output_path}")
    return output_path
