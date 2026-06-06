"""
Project-wide logging configuration.

Usage:
    from utils.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import os
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
)

_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=_FMT,
    handlers=[
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
