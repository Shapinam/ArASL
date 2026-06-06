"""
One-off fix-ups for videos that landed in the wrong class folder in the raw
KArSL-502 archives. Mirrors the manual cells from the notebook.

Run once after extraction:
    python -m scripts.fix_misplaced_videos
"""

import glob
import os

from utils.config import BASE_DIR
from utils.helper_functions import move_files
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# (source_glob_patterns, destination_dir) tuples
FIXES = [
    # signer 01, test: videos 16_34 and 17_11 belong in class 0006, not 0005
    (
        [
            f"{BASE_DIR}/01/test/extracted/0005/*_16_34*_c.mp4",
            f"{BASE_DIR}/01/test/extracted/0005/*_17_11*_c.mp4",
        ],
        f"{BASE_DIR}/01/test/extracted/0006/",
    ),
    # signer 01, train: videos 16_34 and 16_35 belong in class 0006, not 0005
    (
        [
            f"{BASE_DIR}/01/train/extracted/0005/*_16_34*_c.mp4",
            f"{BASE_DIR}/01/train/extracted/0005/*_16_35*_c.mp4",
        ],
        f"{BASE_DIR}/01/train/extracted/0006/",
    ),
]


def main():
    for patterns, dst in FIXES:
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        if not files:
            logger.info("Nothing to move for destination %s", dst)
            continue
        logger.info("Moving %d files -> %s", len(files), dst)
        os.makedirs(dst, exist_ok=True)
        move_files(files, dst)


if __name__ == "__main__":
    main()
