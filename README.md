# ArSL CNN — Modular Training Code

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | All constants (paths, sizes, hyperparameters) |
| `data_exploration.py` | Dataset statistics and sample visualizations |
| `preprocessing.py` | Single-image preprocessing pipeline |
| `dataset.py` | Load all images, split train/val/test, build tf.data pipelines |
| `model.py` | CNN architecture definition |
| `train.py` | Compile, train, plot history |
| `evaluate.py` | Test accuracy, classification report, confusion matrix |
| `export.py` | Save model and class names for deployment |

## How to Run

### Step 1 — Explore and verify individual modules
```bash
python data_exploration.py   # explore dataset before training
python preprocessing.py      # test preprocessing on one image
python model.py              # print model summary
```

### Step 2 — Full pipeline
```bash
python train.py       # loads data → trains → saves model + artifacts
python evaluate.py    # loads saved model → prints metrics + confusion matrix
python export.py      # saves class_names.json (already done by train.py)
```

## Configuration

Edit `config.py` to change any setting:
- `TARGET_SIZE` — image resize target (default 64)
- `BATCH_SIZE` — training batch size (default 32)
- `EPOCHS` — max training epochs (default 50)
- `LEARNING_RATE` — Adam optimizer LR (default 0.001)
- `INPUT_DIR` / `OUTPUT_DIR` — Kaggle paths

## Deployment Output

After running `train.py`, two files are ready for deployment:
- `arsl_cnn_model.keras` — trained model
- `class_names.json` — class label order matching model output
