"""Centralized logging configuration for the knnsur project."""

import logging
import sys
from datetime import datetime
from pathlib import Path

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
    try:
        # Get system RAM usage
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / 1e9
        ram_total_gb = ram.total / 1e9

        # Get GPU VRAM usage
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used_gb = int(info.used) / 1e9
        vram_total_gb = int(info.total) / 1e9
        pynvml.nvmlShutdown()

        msg = f"[MEM] {tag:<25} | VRAM: {vram_used_gb:6.2f} / {vram_total_gb:6.2f} GB | RAM: {ram_used_gb:6.2f} / {ram_total_gb:6.2f} GB"
        logger.debug(msg)
    except Exception:
        pass
