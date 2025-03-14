import os
import sys
import torch
import numpy as np

from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from eval import AdaptiveSentenceTransformer


model_key = sys.argv[1] 

split_dir = "split_indices"
wiki_path = os.path.join(split_dir, "all_paragraphs.pkl")

train_indices = torch.from_numpy(np.load(os.path.join(split_dir, "train_indices.npy"))).to("cuda")
val_indices   = torch.from_numpy(np.load(os.path.join(split_dir, "val_indices.npy"))).to("cuda")

import pickle
with open(wiki_path, 'rb') as f:
    all_paragraphs = pickle.load(f)


num_samples = len(all_paragraphs)

print(f"Train set size: {len(train_indices)}")
print(f"Validation set size: {len(val_indices)}")
print(f"Total passages loaded: {num_samples}")


model = AdaptiveSentenceTransformer(
    models=[model_key],
    device="cuda",
)

embeddings = []
batch_size = 512
num_batches = (len(all_paragraphs) + batch_size - 1) // batch_size

print(f"Generating {num_samples} embeddings for {model_key}")
print(f"Batch size used {batch_size}, number of batches required {num_batches}")

for i in tqdm(range(num_batches), desc="Generating Embeddings"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(all_paragraphs))
    batch_texts = all_paragraphs[start_idx:end_idx]

    batch_embeddings = model.encode(
        batch_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
        convert_to_numpy=False,
        max_length = 512,
    )
 
    embeddings.append(batch_embeddings)
    
embeddings = torch.cat(embeddings, dim=0) 


save_dir = f"embeddings_data/{model_key}"
os.makedirs(save_dir, exist_ok=True)

print(f"Returned Embeddings shape: {embeddings.shape}")
torch.save(embeddings, os.path.join(save_dir, "all.pth") )

train_data = embeddings[train_indices]
val_data   = embeddings[val_indices]

torch.save(train_data, os.path.join(save_dir, "train_embeddings.pth") )
torch.save(val_data, os.path.join(save_dir, "val_embeddings.pth"))

print(f"Train embeddings saved to:      {save_dir}/train_embeddings.pth")
print(f"Validation embeddings saved to: {save_dir}/val_embeddings.pth")