# config.py
import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 16
TOKENIZERS_PARALLELISM = "false"

BATCH_SIZE = 124

LOSS_WEIGHTS = {
    64: 2,    # Highest priority
    128: 1.5,   # High priority
    256: 1.0,   # Medium priority
}

# should probb delete this file
RECONSTRUCTIONS_DIR = "reconstructions"
PLOT_PATH = "loss_plots.png"
SAVE_DIR = "../data/"
TRAIN_SAVE_PATH = os.path.join(SAVE_DIR, "train_embeddings.npz")
VAL_SAVE_PATH = os.path.join(SAVE_DIR, "val_embeddings.npz")
BEST_MODEL_PATH = "best_model.pth"

