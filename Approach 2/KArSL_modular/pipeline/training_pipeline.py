"""
Top-level training pipeline.

For every leave-one-signer-out experiment:
    1. Load the cached .npy splits
    2. Build a fresh ArSL_TGRU model (random weights)
    3. Compile + train with EarlyStopping/ReduceLROnPlateau/Checkpoint
    4. Evaluate on the held-out signer
    5. Log + store metrics

Run from project root:
    python -m pipeline.training_pipeline
"""

import json
import os
from dataclasses import asdict

import numpy as np

from components.evaluation import compute_metrics, predict_labels
from components.model_builder import build_arsl_tgru
from model.evaluate import evaluate_model
from model.train import compile_model, train_model
from utils.config import (
    ARTIFACTS_DIR,
    PROCESSED_DIR,
    SIGNER_EXPERIMENTS,
)
from utils.exception_handling import CustomException
from utils.helper_functions import ensure_dir
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def _load_experiment(exp_folder: str):
    return {
        "X_train": np.load(os.path.join(exp_folder, "X_train.npy")),
        "y_train": np.load(os.path.join(exp_folder, "y_train.npy")),
        "X_val":   np.load(os.path.join(exp_folder, "X_val.npy")),
        "y_val":   np.load(os.path.join(exp_folder, "y_val.npy")),
        "X_test":  np.load(os.path.join(exp_folder, "X_test.npy")),
        "y_test":  np.load(os.path.join(exp_folder, "y_test.npy")),
    }


def run_training_pipeline():
    try:
        ensure_dir(ARTIFACTS_DIR)
        results = {}

        for exp in SIGNER_EXPERIMENTS:
            name = exp["name"]
            exp_folder = os.path.join(PROCESSED_DIR, name)
            logger.info("=" * 60)
            logger.info("Experiment: %s | held-out signer: %s", name, exp["test"])
            logger.info("=" * 60)

            data = _load_experiment(exp_folder)
            logger.info("Shapes: train=%s val=%s test=%s",
                        data["X_train"].shape, data["X_val"].shape, data["X_test"].shape)

            model = build_arsl_tgru()
            compile_model(model)

            train_model(
                model,
                data["X_train"], data["y_train"],
                data["X_val"],   data["y_val"],
                experiment_name=name,
            )

            metrics = evaluate_model(
                model, data["X_test"], data["y_test"],
                title=f"{name} — Unseen Signer {exp['test']}",
            )
            results[name] = asdict(metrics)

        # Persist a summary across all experiments
        summary_path = os.path.join(ARTIFACTS_DIR, "experiments_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved cross-experiment summary -> %s", summary_path)
        return results
    except Exception as e:
        raise CustomException(e) from e


if __name__ == "__main__":
    run_training_pipeline()
