# config.py
# Central configuration — change values here and everything updates automatically

INPUT_DIR   = '/kaggle/input'
OUTPUT_DIR  = '/kaggle/working/arasl_processed'

TARGET_SIZE = 64
BATCH_SIZE  = 32
EPOCHS      = 50
RANDOM_SEED = 42

# Training hyperparameters
LEARNING_RATE   = 1e-3
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 3
EARLY_STOP_PATIENCE = 5
MIN_LR          = 1e-6

# Minimum images per class to be included
MIN_IMAGES_PER_CLASS = 10

# Output file names
MODEL_SAVE_PATH      = 'arsl_cnn_model.keras'
CLASS_NAMES_SAVE_PATH = 'class_names.json'
