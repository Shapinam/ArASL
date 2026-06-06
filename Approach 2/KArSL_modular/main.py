"""
main.py
=======
End-to-end pipeline for KArSL Arabic Sign Language Recognition.

Usage (in Colab or local):
    python main.py [--stage STAGE]

Stages:
    extract      → Download & extract raw videos from Google Drive
    preprocess   → Run keypoint extraction + augmentation → save .npy files
    prepare      → Split data into train/val/test per experiment → save arrays
    train        → Train one model per experiment
    evaluate     → Evaluate all trained models and print summary
    all          → Run every stage in order (default)

Each stage can also be called independently from a notebook by importing and
calling the relevant module functions directly.
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from config.config import MODEL_DIR, SAVE_DIR, SIGNER_EXPERIMENTS
from data.extract import NOTEBOOK_CORRECTIONS, extract_dataset, fix_misplaced_videos, inspect_dataset
from data.loader import load_experiment, prepare_all_experiments
from evaluation.evaluate import evaluate_model, summarise_all_experiments
from features.augmentation import run_augmentation_pipeline
from training.trainer import train_experiment

print(f"TensorFlow {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")


# ── Stage functions ────────────────────────────────────────────────────────────

def stage_extract() -> None:
    """Extract raw videos from 7z archives."""
    print("\n[STAGE] Extracting dataset...")
    extract_dataset()
    fix_misplaced_videos(NOTEBOOK_CORRECTIONS)
    inspect_dataset()


def stage_preprocess() -> None:
    """Run MediaPipe keypoint extraction + augmentation and save .npy files."""
    print("\n[STAGE] Preprocessing videos...")
    run_augmentation_pipeline()


def stage_prepare() -> None:
    """Split .npy files into train/val/test arrays per experiment."""
    print("\n[STAGE] Preparing experiment splits...")
    prepare_all_experiments()


def stage_train() -> dict:
    """Train one model per experiment. Returns {exp_name: (model, history)}."""
    print("\n[STAGE] Training models...")
    trained = {}
    for exp in SIGNER_EXPERIMENTS:
        data = load_experiment(exp["name"])
        model, history = train_experiment(
            exp_name=exp["name"],
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
        )
        trained[exp["name"]] = (model, history)
    return trained


def stage_evaluate(trained: dict = None) -> None:
    """
    Evaluate all experiments. If trained models are not passed in,
    loads the best saved checkpoint for each.
    """
    print("\n[STAGE] Evaluating models...")
    results = []

    for exp in SIGNER_EXPERIMENTS:
        data = load_experiment(exp["name"])

        if trained and exp["name"] in trained:
            model = trained[exp["name"]][0]
        else:
            checkpoint = os.path.join(MODEL_DIR, f"best_arsl_tgru_{exp['name']}.keras")
            if not os.path.exists(checkpoint):
                checkpoint = checkpoint.replace(".keras", ".h5")
            print(f"Loading checkpoint: {checkpoint}")
            model = tf.keras.models.load_model(checkpoint)

        metrics = evaluate_model(
            model,
            X_test=data["X_test"],
            y_test=data["y_test"],
            exp_name=exp["name"],
        )
        metrics["name"] = exp["name"]
        results.append(metrics)

    summarise_all_experiments(results)


# ── Entry point ────────────────────────────────────────────────────────────────

STAGES = {
    "extract":    stage_extract,
    "preprocess": stage_preprocess,
    "prepare":    stage_prepare,
    "train":      stage_train,
    "evaluate":   stage_evaluate,
}


def main(stage: str = "all") -> None:
    if stage == "all":
        stage_extract()
        stage_preprocess()
        stage_prepare()
        trained = stage_train()
        stage_evaluate(trained)
    elif stage in STAGES:
        result = STAGES[stage]()
        # If training, also evaluate immediately
        if stage == "train":
            stage_evaluate(result)
    else:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGES)} or 'all'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KArSL pipeline")
    parser.add_argument(
        "--stage", default="all",
        choices=list(STAGES.keys()) + ["all"],
        help="Which pipeline stage to run.",
    )
    args = parser.parse_args()
    main(args.stage)
