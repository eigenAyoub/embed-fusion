# config.py

import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_DIR = "data"

TRAIN_SAVE_PATH = os.path.join(SAVE_DIR, "train_embeddings.npz")
VAL_SAVE_PATH = os.path.join(SAVE_DIR, "val_embeddings.npz")

RECONSTRUCTIONS_DIR = "reconstructions"

BATCH_SIZE = 32
NUM_EPOCHS = 30

# optimizer // scheduler
LEARNING_RATE = 6e-4
WEIGHT_DECAY  = 4e-5

STEP_SIZE = 4
GAMMA = 0.1

PATIENCE = 10

# autoenc
INPUT_DIM = 2048 
COMPRESSED_DIM = 1024 

#BEST_MODEL_PATH = f"models_pth/{COMPRESSED_DIM}/"
PLOT_PATH= f"loss_curve_{COMPRESSED_DIM}_6.pth"

# can we optimmize this? why 64 work worse? where do you even need it?
NUM_WORKERS = 64 

# why not this? 
TOKENIZERS_PARALLELISM = "false"


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

