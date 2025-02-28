import os
import sys
import torch
import numpy as np

from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from eval import AdaptiveSentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_key = "mxbai"
split_dir = "split_indices"
wiki_path = os.path.join(split_dir, "all_paragraphs.pkl")

train_indices = np.load(os.path.join(split_dir, "train_indices.npy"))
val_indices   = np.load(os.path.join(split_dir, "val_indices.npy"))

import pickle
with open(wiki_path, 'rb') as f:
    all_paragraphs = pickle.load(f)


num_samples = len(all_paragraphs)

print(f"Train set size: {len(train_indices)}")
print(f"Validation set size: {len(val_indices)}")
print(f"Total passages loaded: {num_samples}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AdaptiveSentenceTransformer(
    models=[model_key],
    device="cuda",
    checkpoint_path= None,
    input_dim=None,
    compressed_dim=None,
    truncate=None,
    quantizer_path=None
)

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

train_data = embeddings[train_indices]
val_data = embeddings[val_indices]

save_dir = f"embeddings_data/f_{model_key}_wiki_500k"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "train_embeddings.npy"), train_data)
np.save(os.path.join(save_dir, "val_embeddings.npy"),   val_data)

print(f"Train embeddings saved to:      {save_dir}/train_embeddings.npy")
print(f"Validation embeddings saved to: {save_dir}/val_embeddings.npy")

