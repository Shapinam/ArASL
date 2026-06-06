# KArSL Sign-Recognition — Modular Project

Modular restructure of `KArSL_trial_Notebook.ipynb`. The pipeline:

1. Downloads the KArSL-502 archives from Google Drive.
2. Extracts 59 selected sign classes (1–10, 32–70, 71–80) from per-signer 7z archives.
3. Decodes every video into 30 frames, extracts MediaPipe hand + face keypoints
   (53 points × 3 coords), normalizes them into 159-D vectors, and caches the
   (30, 159) sequences as `.npy` — with 10 augmented copies per training video.
4. Builds three leave-one-signer-out experiments and saves their splits.
5. Trains an **ArSL_TGRU** model (Transformer branch + two parallel GRU branches
   → concatenate → classifier) for each experiment.
6. Reports weighted Accuracy / Precision / Recall / F1 on the unseen signer.

## Layout

```
karsl_project/
│── data_processing/
│   ├── load_data.py             # load .npy splits into arrays
│   ├── preprocess.py            # read_video, standardize_frames,
│   │                            # KeypointExtractor, normalize_frame
│   └── feature_engineering.py   # augment_video (rotate + scale)
│
│── model/
│   ├── train.py                 # compile + callbacks + fit (per experiment)
│   ├── evaluate.py              # weighted Acc/Prec/Recall/F1
│   └── predict.py               # ArSLVideoPredictor: video file -> sign
│
│── pipeline/
│   ├── data_pipeline.py         # extract -> npy -> splits
│   └── training_pipeline.py     # train + eval all 3 experiments
│
│── components/
│   ├── data_ingestion.py        # ArchiveExtractor, VideoProcessor,
│   │                            # ExperimentSplitter
│   ├── model_builder.py         # PositionalEmbedding, transformer_block,
│   │                            # gru_branch, build_arsl_tgru
│   └── evaluation.py            # ClassificationMetrics + helpers
│
│── notebooks/
│   └── exploratory_analysis.ipynb   # dataset structure + counts
│
│── scripts/
│   ├── download_dataset.py      # gdown the raw archives
│   └── fix_misplaced_videos.py  # move misplaced files (notebook cells 14-16)
│
│── utils/
│   ├── config.py                # signers, classes, hyperparameters, paths
│   ├── helper_functions.py      # ensure_dir, download, run_7z, move_files
│   ├── logging_setup.py
│   └── exception_handling.py
│
│── app.py                       # Flask inference API
│── requirements.txt
│── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# Optional: override the Google-Drive root (defaults to Colab path)
export KARSL_DRIVE_BASE=/path/to/arabic_dataset

# 1. Download archives
python -m scripts.download_dataset

# 2. Extract -> keypoints -> splits
python -m pipeline.data_pipeline

# 2b. (optional) fix the few misplaced videos in signer 01
python -m scripts.fix_misplaced_videos

# 3. Train + evaluate all 3 leave-one-signer-out experiments
python -m pipeline.training_pipeline

# 4. Serve predictions
python app.py
curl -X POST -F "video=@sample.mp4" \
     -F "model_path=/path/to/best_arsl_tgru_Exp1_Test_03.keras" \
     http://localhost:5000/predict
```

## Notebook → module mapping

| Notebook cell(s) | New location |
|---|---|
| `drive.mount`, `pip install gdown/py7zr`         | `requirements.txt`, `scripts/download_dataset.py` |
| `gdown --folder ...`                              | `scripts/download_dataset.py` |
| 7z extraction loop (cell 5)                       | `components/data_ingestion.ArchiveExtractor` |
| Directory inspection (cells 6, 7)                 | `notebooks/exploratory_analysis.ipynb` |
| MediaPipe setup + keypoint funcs (cell 12)        | `data_processing/preprocess.py` (`KeypointExtractor`, `read_video`, `standardize_frames`, `normalize_frame`) |
| Manual file moves (cells 14–16)                   | `scripts/fix_misplaced_videos.py` |
| Video → augmented .npy loop (cell 18)             | `components/data_ingestion.VideoProcessor` |
| `load_split` (cell 20)                            | `data_processing/load_data.py` |
| Three-experiment splitter (cell 22)               | `components/data_ingestion.ExperimentSplitter` |
| `PositionalEmbedding`, `transformer_block`, `gru_branch`, `build_arsl_tgru` (cells 24–27) | `components/model_builder.py` |
| `model.compile` + callbacks + `model.fit`         | `model/train.py` |
| Test eval w/ weighted metrics (cells 33, 44, 54)  | `model/evaluate.py` + `components/evaluation.py` |
| Per-experiment driver (cells 30, 36–44, 46–54)    | `pipeline/training_pipeline.py` |

## Configuration

All hyperparameters and paths live in `utils/config.py`. Override the dataset
root via `KARSL_DRIVE_BASE`, artifacts dir via `KARSL_ARTIFACTS_DIR`, and the
MediaPipe-task cache via `KARSL_MP_DIR`.

## Logging & error handling

- `utils/logging_setup.py` writes timestamped logs to `logs/` and also to
  stdout.
- `utils/exception_handling.CustomException` wraps every error with the file
  name and line number where it was raised.
