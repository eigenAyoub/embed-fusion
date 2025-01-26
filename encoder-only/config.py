# config.py

import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PLOT_PATH= f"loss_curve.pth"

# can we optimmize this? why 64 work worse? where do you even need it?
NUM_WORKERS = 64 
TOKENIZERS_PARALLELISM = "false"

BATCH_SIZE = 64
NUM_EPOCHS = 30

# optimizer // scheduler ? why do you 
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 4e-5

STEP_SIZE = 4
GAMMA = 0.1

PATIENCE = 10

LOSS_WEIGHTS = {
    64: 2,    # Highest priority
    128: 1.5,   # High priority
    256: 1.0,   # Medium priority
}

MODEL_CATALOGUE = {
            "mxbai"           : "mixedbread-ai/mxbai-embed-large-v1",
            "bge"             : "BAAI/bge-large-en-v1.5",
            "e5"              : "intfloat/e5-large-v2"  ,
            "snowflake"       : "Snowflake/snowflake-arctic-embed-m",
            "snowflake-l"     : "Snowflake/snowflake-arctic-embed-l",
            "gte-base"        : "thenlper/gte-base",
            "gte-large"       : "thenlper/gte-large",
            "gte-small"       : "thenlper/gte-small",
            "e5-small"        : "intfloat/e5-small-v2", # (33M)
            "bge-small"       : "BAAI/bge-small-en-v1.5", # (33M)
            "jina-v3"         : "jinaai/jina-embeddings-v3"
}


RECONSTRUCTIONS_DIR = "reconstructions"
PLOT_PATH = "loss_plots.png"

SAVE_DIR = "../data/"
TRAIN_SAVE_PATH = os.path.join(SAVE_DIR, "train_embeddings.npz")
VAL_SAVE_PATH = os.path.join(SAVE_DIR, "val_embeddings.npz")


BEST_MODEL_PATH = "best_model.pth"

