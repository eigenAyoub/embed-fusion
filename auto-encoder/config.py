# config.py

import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_DIR = "data/e5_embed_wiki"

TRAIN_SAVE_PATH = os.path.join(SAVE_DIR, "train_embeddings.npz")
VAL_SAVE_PATH = os.path.join(SAVE_DIR, "val_embeddings.npz")

RECONSTRUCTIONS_DIR = "reconstructions"

BATCH_SIZE = 32
NUM_EPOCHS = 30

# optimizer // scheduler
LEARNING_RATE = 6e-3
WEIGHT_DECAY  = 1e-5

STEP_SIZE = 4
GAMMA = 0.1

PATIENCE = 10

# autoenc
INPUT_DIM = 1024 
COMPRESSED_DIM = 384 

#BEST_MODEL_PATH = f"models_pth/{COMPRESSED_DIM}/"
PLOT_PATH= f"loss_curve_{COMPRESSED_DIM}_6.pth"

# can we optimmize this? why 64 work worse? where do you even need it?
NUM_WORKERS = 64 

# why not this? 
TOKENIZERS_PARALLELISM = "false"
