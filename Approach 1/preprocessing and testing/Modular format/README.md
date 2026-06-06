# Arabic Sign Language CNN — Modular Project

A modular restructure of the original `Full_Code_Notebook.ipynb` that trains a
custom CNN on grayscale Arabic Sign Language (ArSL) images.

The layout follows the modular-coding-in-ML best-practice approach: separation
of concerns, encapsulation, reusable components, centralized config, logging,
and explicit pipelines.

## Project structure

```
arsl_project/
│── data_processing/
│   ├── load_data.py             # Discover classes and image paths
│   ├── preprocess.py            # Per-image preprocessing (grayscale, pad, resize)
│   └── feature_engineering.py   # tf.data datasets + augmentation
│
│── model/
│   ├── train.py                 # Compile + fit
│   ├── evaluate.py              # Test-set loss/accuracy
│   └── predict.py               # ArSLPredictor for inference on new images
│
│── pipeline/
│   ├── data_pipeline.py         # ingestion -> tf.data bundles
│   └── training_pipeline.py     # Full end-to-end runner
│
│── components/
│   ├── data_ingestion.py        # DataIngestion class (load + split)
│   ├── model_builder.py         # build_arsl_cnn()
│   └── evaluation.py            # Classification report, confusion matrix, plots
│
│── notebooks/
│   └── exploratory_analysis.ipynb   # EDA (class counts, sample images, dtype checks)
│
│── utils/
│   ├── config.py                # All hyperparameters + paths
│   ├── helper_functions.py      # save/load class names, dir helpers
│   ├── logging_setup.py         # Project-wide logger
│   └── exception_handling.py    # CustomException with file+line context
│
│── app.py                       # Minimal Flask inference API
│── requirements.txt
│── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# Point at your data (default is /kaggle/input)
export ARSL_INPUT_DIR=/path/to/dataset

# Train end-to-end
python -m pipeline.training_pipeline

# Serve inference
python app.py
curl -X POST -F "image=@sample.jpg" http://localhost:5000/predict
```

## What goes where (mapping from the notebook)

| Notebook cell                        | New location |
|--------------------------------------|--------------|
| Imports + GPU check                  | distributed across modules |
| EDA (class counts, samples)          | `notebooks/exploratory_analysis.ipynb` |
| `preprocess_image`                   | `data_processing/preprocess.py` |
| Class discovery + load to memory     | `data_processing/load_data.py` + `components/data_ingestion.py` |
| Train/val/test split                 | `components/data_ingestion.py` |
| Augmentation + `make_dataset`        | `data_processing/feature_engineering.py` |
| `build_arsl_cnn`                     | `components/model_builder.py` |
| Compile, callbacks, `model.fit`      | `model/train.py` |
| Accuracy/loss plots                  | `components/evaluation.py` |
| Test eval, classification report, CM | `model/evaluate.py` + `components/evaluation.py` |
| `model.save` + class-name JSON       | `utils/helper_functions.py` + `pipeline/training_pipeline.py` |
| Inference helper                     | `model/predict.py` + `app.py` |

## Configuration

All hyperparameters and paths live in `utils/config.py`. Override via env vars
(`ARSL_INPUT_DIR`, `ARSL_OUTPUT_DIR`, `ARSL_ARTIFACTS_DIR`) or edit the file
directly.

## Logging & error handling

- `utils/logging_setup.py` writes timestamped logs to `logs/` and the console.
- `utils/exception_handling.CustomException` wraps every error with the file
  name and line number where it was raised, so traces are easy to pinpoint.
