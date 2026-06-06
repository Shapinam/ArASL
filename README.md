# ArSL CNN — Modular Training Code

## Dataset
```https://www.kaggle.com/datasets/gannayasser/arabic-alphabets-sign-language-dataset-arasl```

## Deployed link
```https://ar-asl.vercel.app/```

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

# KArSL Arabic Sign Language Recognition — Modular Pipeline

## Dataset
```https://hamzah-luqman.github.io/KArSL/```

## Project Structure

```
KArSL_modular/
│
├── config/
│   └── config.py           ← ALL hyperparameters, paths, experiment definitions
│
├── data/
│   ├── extract.py          ← Dataset download, 7z extraction, video relocation
│   └── loader.py           ← .npy loading, train/val/test splits, saving
│
├── features/
│   ├── keypoints.py        ← MediaPipe init, frame reading, extraction, normalisation
│   └── augmentation.py     ← Full video → .npy pipeline (all signers × classes)
│
├── model/
│   └── architecture.py     ← ArSL-TGRU model (Transformer + dual GRU)
│
├── training/
│   └── trainer.py          ← Compile, callbacks, training loop
│
├── evaluation/
│   └── evaluate.py         ← Metrics, per-class report, summary table
│
├── main.py                 ← CLI orchestrator (--stage extract|preprocess|prepare|train|evaluate|all)
└── KArSL_pipeline.ipynb    ← Thin Colab notebook — imports modules, no business logic
```

## Quick Start

### In Colab (notebook)
1. Upload or clone this folder to `/content/KArSL_modular`.
2. Open `KArSL_pipeline.ipynb` and run cells top-to-bottom.

### From the command line / script
```python
from main import main
main("all")          # full pipeline
main("train")        # training only (expects prepare to have run)
main("evaluate")     # evaluation only (loads saved checkpoints)
```

Or via CLI:
```bash
python main.py --stage all
python main.py --stage train
```

## Configuration

Everything you need to tune is in `config/config.py`:

| Variable | Default | Description |
|---|---|---|
| `BASE_DIR` | `.../KArSL-502` | Raw video root |
| `NPY_DIR` | `.../NPY` | Preprocessed .npy output |
| `SAVE_DIR` | `.../Processed_Experiments` | Train/val/test array output |
| `SEQ_LEN` | 30 | Frames per sequence |
| `N_AUG` | 10 | Augmentation copies per training video |
| `EPOCHS` | 50 | Max training epochs |
| `BATCH_SIZE` | 16 | Mini-batch size |
| `LR` | 1e-3 | Initial learning rate |
| `N_CLASSES` | 59 | Number of sign classes |

## Model Architecture

```
Input (30, 159)
    │
    ├─ PositionalEmbedding
    │  └─ 4 × TransformerBlock (sequential)
    │        └─ LayerNorm → MHA → Residual → FFN → Residual → GAP → Dense(64) → Dropout
    │
    ├─ GRU Branch 1: GRU(64) → GRU(128) → GRU(64) → Dense(64) → Dropout
    │
    └─ GRU Branch 2: GRU(64) → GRU(128) → GRU(64) → Dense(64) → Dropout
    │
Concatenate([T, G1, G2])  →  (192,)
Dense(128, relu) → Dropout(0.5) → Dense(59, softmax)
```

## Experiments

Cross-signer evaluation (leave-one-signer-out):

| Experiment | Train signers | Test signer |
|---|---|---|
| Exp1_Test_03 | 01, 02 | 03 |
| Exp2_Test_02 | 01, 03 | 02 |
| Exp3_Test_01 | 02, 03 | 01 |


## Deployment Output

After running `train.py`, two files are ready for deployment:
- `arsl_cnn_model.keras` — trained model
- `class_names.json` — class label order matching model output
