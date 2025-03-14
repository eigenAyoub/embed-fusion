#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import sys
import os

#model_keys = ["snowflake-m", "no-ins", "gte-small", "e5-small"]
#model_keys = ["no-ins", "gte-small", "e5-small", "bge-small"]

model_keys = ["no-ins", "gte-small"]

output = "combined-s" #"all-stars"

os.makedirs(output, exist_ok=True)

for split in ["val", "train"]:
    train_paths = [f"embeddings_data/{m_key}/{split}_embeddings.pth" for m_key in model_keys]
    train_data  = [torch.load(train_p) for train_p in train_paths]
    embeddings  = torch.cat(train_data, dim=1)
    embeddings  = F.normalize(embeddings, p=2, dim=1) 
    output_path = os.path.join(output, f"{split}_embeddings.pth")
    torch.save(embeddings, output_path)
    print(f"{split} done {output}")
    print(f"Emb of shape {embeddings.shape}")
