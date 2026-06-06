"""
Small, generic helpers used across modules (file IO, persistence).
"""

import json
import os
from typing import List

from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_class_names(class_names: List[str], path: str) -> None:
    """Persist the ordered class-name list as JSON (used by inference)."""
    try:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
        logger.info("Saved class names to %s", path)
    except Exception as e:
        raise CustomException(e) from e


def load_class_names(path: str) -> List[str]:
    """Load the ordered class-name list from JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e) from e
