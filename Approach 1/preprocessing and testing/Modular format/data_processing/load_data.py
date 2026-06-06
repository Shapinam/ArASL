"""
Discover image classes (one folder per class) and collect file paths.
"""

import os
from typing import Dict, List

from utils.config import IMAGE_EXTENSIONS, MIN_IMAGES_PER_CLASS
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def discover_classes(input_dir: str) -> Dict[str, List[str]]:
    """
    Walk `input_dir` and return {class_name: [image_path, ...]}.
    Only leaf folders containing more than MIN_IMAGES_PER_CLASS images count.
    """
    try:
        valid_classes: Dict[str, List[str]] = {}
        for dirname, dirnames, filenames in os.walk(input_dir):
            if len(dirnames) != 0:
                continue
            images = [
                f for f in filenames
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]
            if len(images) <= MIN_IMAGES_PER_CLASS:
                continue
            class_name = os.path.basename(dirname)
            valid_classes[class_name] = [
                os.path.join(dirname, f) for f in images
            ]
        logger.info(
            "Discovered %d classes under %s", len(valid_classes), input_dir
        )
        return valid_classes
    except Exception as e:
        raise CustomException(e) from e
