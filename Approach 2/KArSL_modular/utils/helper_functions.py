"""Generic helpers: directory creation, downloading model files, file ops."""

import os
import shutil
import subprocess
import urllib.request

from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download(url: str, path: str) -> None:
    """Download `url` to `path` if not already present."""
    if os.path.exists(path):
        return
    try:
        ensure_dir(os.path.dirname(path) or ".")
        logger.info("Downloading %s -> %s", url, path)
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise CustomException(e) from e


def move_files(file_paths, dst_dir: str) -> int:
    """Move every path in `file_paths` into `dst_dir`. Returns count moved."""
    try:
        ensure_dir(dst_dir)
        count = 0
        for f in sorted(file_paths):
            shutil.move(f, os.path.join(dst_dir, os.path.basename(f)))
            count += 1
        logger.info("Moved %d files to %s", count, dst_dir)
        return count
    except Exception as e:
        raise CustomException(e) from e


def run_7z(archive_path: str, pattern: str, out_dir: str) -> None:
    """Extract a pattern from a 7z archive using the `7z` CLI."""
    try:
        ensure_dir(out_dir)
        result = subprocess.run(
            ["7z", "x", archive_path, pattern, f"-o{out_dir}", "-y"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("7z warning (%s): %s", archive_path, result.stderr.strip())
    except Exception as e:
        raise CustomException(e) from e
