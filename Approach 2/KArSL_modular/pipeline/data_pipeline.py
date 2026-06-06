"""
Full data pipeline:
    7z extract -> per-video keypoint .npy -> experiment splits on disk.

Stages can be turned on/off — useful when re-running just the final step.
"""

from components.data_ingestion import (
    ArchiveExtractor,
    ExperimentSplitter,
    VideoProcessor,
)
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def run_data_pipeline(
    extract_archives: bool = True,
    process_videos:   bool = True,
    build_splits:     bool = True,
) -> None:
    try:
        if extract_archives:
            logger.info("Stage 1/3 — extracting 7z archives")
            ArchiveExtractor().run()
        if process_videos:
            logger.info("Stage 2/3 — converting videos to keypoint .npy")
            VideoProcessor().run()
        if build_splits:
            logger.info("Stage 3/3 — building leave-one-signer-out splits")
            ExperimentSplitter().run()
        logger.info("Data pipeline complete")
    except Exception as e:
        raise CustomException(e) from e


if __name__ == "__main__":
    run_data_pipeline()
