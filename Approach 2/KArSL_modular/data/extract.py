"""
data/extract.py
===============
Handles downloading the KArSL dataset from Google Drive and
extracting the selected sign classes from 7z archives.
"""

import os
import subprocess

from config.config import BASE_DIR, SIGNERS, KEEP_FIRST, KEEP_SECOND


def download_dataset(drive_url: str, out_dir: str) -> None:
    """Download the KArSL dataset folder from Google Drive using gdown."""
    os.system(f'gdown --folder "{drive_url}" -O {out_dir}')


def _extract_classes(archive_path: str, class_list: list, out_dir: str, archive_prefix: str) -> None:
    """
    Extract a specific set of class folders from a 7z archive.

    Args:
        archive_path:    Full path to the .7z file.
        class_list:      List of zero-padded class ID strings (e.g. ["0001", "0032"]).
        out_dir:         Destination directory for extracted files.
        archive_prefix:  Path prefix inside the archive (e.g. "0001-0070\\").
    """
    os.makedirs(out_dir, exist_ok=True)
    for cls in class_list:
        print(f"  Extracting class {cls}...")
        result = subprocess.run(
            ["7z", "x", archive_path, f"{archive_prefix}{cls}\\*", f"-o{out_dir}", "-y"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    WARNING: {result.stderr.strip()}")


def extract_dataset(
    base_dir: str = BASE_DIR,
    signers: list = SIGNERS,
    keep_first: list = None,
    keep_second: list = None,
) -> None:
    """
    Extract selected classes for every signer / split combination.

    Args:
        base_dir:     Root of the KArSL-502 directory.
        signers:      List of signer IDs to process.
        keep_first:   Class IDs to pull from the first archive (0001-0070.7z or 0001-0071.7z).
        keep_second:  Class IDs to pull from the second archive (0071-0170.7z).
    """
    if keep_first is None:
        keep_first = [f"{i:04d}" for i in KEEP_FIRST]
    if keep_second is None:
        keep_second = [f"{i:04d}" for i in KEEP_SECOND]

    for signer in signers:
        for split in ["train", "test"]:
            archive_dir = f"{base_dir}/{signer}/{split}"
            out_dir     = f"{archive_dir}/extracted"

            # First archive (naming varies between signers)
            arch1 = f"{archive_dir}/0001-0070.7z"
            if not os.path.exists(arch1):
                arch1 = f"{archive_dir}/0001-0071.7z"

            arch2 = f"{archive_dir}/0071-0170.7z"

            if os.path.exists(arch1):
                prefix1 = os.path.basename(arch1).replace(".7z", "") + "\\"
                print(f"\n[{signer}/{split}] First archive ({os.path.basename(arch1)})...")
                _extract_classes(arch1, keep_first, out_dir, prefix1)
            else:
                print(f"\n[{signer}/{split}] ⚠️  First archive not found, skipping...")

            if os.path.exists(arch2):
                print(f"\n[{signer}/{split}] Second archive...")
                _extract_classes(arch2, keep_second, out_dir, "0071-0170\\")
            else:
                print(f"\n[{signer}/{split}] ⚠️  Second archive not found, skipping...")

        print(f"\n✅ Signer {signer} done!")

    print("\n🎉 All signers extracted!")


def inspect_dataset(base_dir: str = BASE_DIR, signers: list = SIGNERS) -> None:
    """Print a summary of extracted videos per signer / split / class."""
    for signer in signers:
        print(f"\n{'='*40}\n  Signer {signer}\n{'='*40}")
        for split in ["train", "test"]:
            extracted_path = os.path.join(base_dir, signer, split, "extracted")
            if not os.path.exists(extracted_path):
                print(f"  {split}: ❌ extracted folder not found")
                continue

            total_videos, total_classes = 0, 0
            for cls in sorted(os.listdir(extracted_path)):
                cls_path = os.path.join(extracted_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                videos = len(os.listdir(cls_path))
                total_videos  += videos
                total_classes += 1

            print(f"  {split}: {total_classes} classes | {total_videos} videos")


def fix_misplaced_videos(corrections: list) -> None:
    """
    Move incorrectly placed videos to the right class folder.

    Args:
        corrections: List of dicts with keys:
                     - 'src_dir':  source class folder
                     - 'dst_dir':  destination class folder
                     - 'patterns': list of glob patterns relative to src_dir
    """
    import glob
    import shutil

    for item in corrections:
        src_dir  = item["src_dir"]
        dst_dir  = item["dst_dir"]
        patterns = item["patterns"]

        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(src_dir, pat)))

        if not files:
            print(f"  No files matched in {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)
        for f in sorted(files):
            dst_path = os.path.join(dst_dir, os.path.basename(f))
            shutil.move(f, dst_path)
            print(f"  Moved: {os.path.basename(f)}")
        print(f"  ✅ Moved {len(files)} files → {dst_dir}")


# ── Example correction list (mirrors the notebook fix) ────────────────────────
NOTEBOOK_CORRECTIONS = [
    {
        "src_dir":  f"{BASE_DIR}/01/test/extracted/0005/",
        "dst_dir":  f"{BASE_DIR}/01/test/extracted/0006/",
        "patterns": ["*_16_34*_c.mp4", "*_17_11*_c.mp4"],
    },
    {
        "src_dir":  f"{BASE_DIR}/01/train/extracted/0005/",
        "dst_dir":  f"{BASE_DIR}/01/train/extracted/0006/",
        "patterns": ["*_16_34*_c.mp4", "*_16_35*_c.mp4"],
    },
]
