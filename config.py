# config.py

import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_PATH= f"loss_curve.pth"
NUM_WORKERS = 64 
TOKENIZERS_PARALLELISM = "false"

BATCH_SIZE = 64
NUM_EPOCHS = 20

LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 0.01

STEP_SIZE = 4
GAMMA = 0.1

PATIENCE = 10

LOSS_WEIGHTS = {
    64: 2,    # Highest priority
    128: 1.5,   # High priority
    256: 1.0,   # Medium priority
}


RECONSTRUCTIONS_DIR = "reconstructions"
PLOT_PATH = "loss_plots.png"

SAVE_DIR = "../data/"
TRAIN_SAVE_PATH = os.path.join(SAVE_DIR, "train_embeddings.npz")
VAL_SAVE_PATH = os.path.join(SAVE_DIR, "val_embeddings.npz")


BEST_MODEL_PATH = "best_model.pth"

