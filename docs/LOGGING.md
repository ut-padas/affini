# Logging Setup Guide

## Quick Start

**Entry point (experiments_gpu.py, etc.):**
```python
import logging
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()  # Call ONCE at startup
    logger.info("Experiment started")
    
if __name__ == "__main__":
    main()
```

**Library modules (src/\*.py):**
```python
import logging
logger = logging.getLogger(__name__)  # Just get, don't configure

def my_function():
    logger.info("Function called")
    logger.debug("Debug info")
```

## Key Features

- **Console**: Logs at specified level (default: INFO)
- **File**: Captures EVERYTHING (default: DEBUG) in `.logs/YYYYMMDD_HHMMSS.log`
- **Unified**: All modules log together automatically

## Common Usage

```python
from src.logging_config import setup_logging

# Default: Console=INFO, File=DEBUG
setup_logging()

# Console shows only warnings, file gets everything
setup_logging(console_level=logging.WARNING)

# Both at DEBUG (very verbose)
setup_logging(console_level=logging.DEBUG)

# Console only (no file)
setup_logging(file_output=False)
```

## Log Levels

```python
logger.critical("...")  # 50 - System failure
logger.error("...")     # 40 - Errors
logger.warning("...")   # 30 - Warnings
logger.info("...")      # 20 - General info (console default)
logger.debug("...")     # 10 - Debug details (file default)
```

