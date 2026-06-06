"""
One-off helper that downloads the KArSL-502 archives from Google Drive.
Run once before the rest of the pipeline:

    python -m scripts.download_dataset
"""

import os
import subprocess

from utils.config import DRIVE_BASE, GDRIVE_FOLDER_URL
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def main():
    os.makedirs(DRIVE_BASE, exist_ok=True)
    logger.info("Downloading dataset from %s into %s", GDRIVE_FOLDER_URL, DRIVE_BASE)
    # gdown handles Google-Drive folders; needs to be installed (`pip install gdown`)
    subprocess.run(
        ["gdown", "--folder", GDRIVE_FOLDER_URL, "-O", DRIVE_BASE],
        check=True,
    )


if __name__ == "__main__":
    main()
