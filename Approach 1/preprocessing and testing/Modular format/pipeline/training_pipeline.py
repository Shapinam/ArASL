"""
Top-level training pipeline.

Run from the project root:
    python -m pipeline.training_pipeline
"""

from components.evaluation import (
    collect_predictions,
    plot_confusion_matrix,
    plot_training_history,
    print_classification_report,
)
from components.model_builder import build_arsl_cnn
from model.evaluate import evaluate_model
from model.train import train_model
from pipeline.data_pipeline import run_data_pipeline
from utils.config import (
    CLASS_NAMES_PATH,
    EPOCHS,
    MODEL_PATH,
    TARGET_SIZE,
)
from utils.exception_handling import CustomException
from utils.helper_functions import ensure_dir, save_class_names
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def main():
    try:
        # 1. Data
        bundle = run_data_pipeline()

        # 2. Model
        model = build_arsl_cnn(
            input_shape=(TARGET_SIZE, TARGET_SIZE, 1),
            num_classes=bundle.num_classes,
        )
        model.summary()

        # 3. Train
        history = train_model(model, bundle.train_ds, bundle.val_ds, epochs=EPOCHS)

        # 4. Evaluate
        plot_training_history(history)
        evaluate_model(model, bundle.test_ds)

        y_true, y_pred = collect_predictions(model, bundle.test_ds)
        print_classification_report(y_true, y_pred, bundle.class_names)
        plot_confusion_matrix(y_true, y_pred, bundle.class_names)

        # 5. Persist artifacts
        ensure_dir("artifacts")
        model.save(MODEL_PATH)
        save_class_names(bundle.class_names, CLASS_NAMES_PATH)
        logger.info("Saved model -> %s", MODEL_PATH)
        logger.info("Saved class names -> %s", CLASS_NAMES_PATH)
    except Exception as e:
        raise CustomException(e) from e


if __name__ == "__main__":
    main()
