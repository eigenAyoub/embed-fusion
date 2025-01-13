import os
import sys

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {
            "mxbai"           : "mixedbread-ai/mxbai-embed-large-v1",  
            "bge"             : "BAAI/bge-large-en-v1.5"                 ,
            "e5"              : "intfloat/e5-large-v2"              ,
            "snowflake-l"     : "Snowflake/snowflake-arctic-embed-l",
            "gte-base"        : "thenlper/gte-base",
            "gte-large"       : "thenlper/gte-large",
            "gte-small"       : "thenlper/gte-small",
            "snowflake-m"     : "Snowflake/snowflake-arctic-embed-m-v1.5",
            "jina-v3"         : "jinaai/jina-embeddings-v3",
            "gist":"avsolatorio/GIST-small-Embedding-v0", # (33M)
            "e5-small"        : "intfloat/e5-small-v2", # (33M)
            "bge-small"       : "BAAI/bge-small-en-v1.5", # (33M)
}

m_name = models["gist"] 

split_dir = "split_indices"
wiki_path = os.path.join(split_dir, "all_paragraphs.pkl")

train_indices = np.load(os.path.join(split_dir, "train_indices.npy"))
val_indices   = np.load(os.path.join(split_dir, "val_indices.npy"))

import pickle
with open(wiki_path, 'rb') as f:
    all_paragraphs = pickle.load(f)

#all_paragraphs = all_paragraphs[:10_000] 

num_samples = len(all_paragraphs)

print(f"Train set size: {len(train_indices)}")
print(f"Validation set size: {len(val_indices)}")
print(f"Total passages loaded: {num_samples}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer(m_name,
                            trust_remote_code = True
                            ).to("cuda")

model.eval()
embeddings = []

batch_size = 512 
num_batches = (len(all_paragraphs) + batch_size - 1) // batch_size

for i in tqdm(range(num_batches), desc="Generating Embeddings"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(all_paragraphs))
    batch_texts = all_paragraphs[start_idx:end_idx]

    batch_embeddings = model.encode(
        batch_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
        convert_to_numpy=True,
        max_length = 512,
        device="cuda"
    )

    embeddings.append(batch_embeddings)
    
embeddings = np.vstack(embeddings)

print(f"Embeddings shape: {embeddings.shape}")


# Split embeddings based on saved indices

train_data = embeddings[train_indices]
val_data = embeddings[val_indices]

# Save embeddings using compressed .npz format to save space
save_dir = f"embeddings_data/nnew_{m_name}_wiki_500k"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "train_embeddings.npy"), train_data)
np.save(os.path.join(save_dir, "val_embeddings.npy"),   val_data)

print(f"Train embeddings saved to: {save_dir}/train_embeddings.npy")
print(f"Validation embeddings saved to: {save_dir}/val_embeddings.npy")

#np.save(os.path.join(save_dir, "bge-small-all.npy"), embeddings)
