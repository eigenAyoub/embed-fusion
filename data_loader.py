# data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    TRAIN_SAVE_PATH,
    VAL_SAVE_PATH,
    BATCH_SIZE,
    NUM_WORKERS
)

def load_embeddings(file_path):
    return np.load(file_path)["embeddings"]

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

def get_data_loaders():
    train_data = load_embeddings(TRAIN_SAVE_PATH)
    val_data = load_embeddings(VAL_SAVE_PATH)
    
    print(f"Loaded train embeddings of shape: {train_data.shape}")
    print(f"Loaded validation embeddings of shape: {val_data.shape}")
    
    train_dataset = EmbeddingDataset(train_data)
    val_dataset = EmbeddingDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        #shuffle=True, 
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        #shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    return train_loader, val_loader


def get_data(path):
    train_data = np.load(path)
    
    print(f"Loading with batch size: {BATCH_SIZE}")
    print(f"Loaded embeddings of shape: {train_data.shape}")
    
    train_dataset = EmbeddingDataset(train_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        #shuffle=True, 
        num_workers=NUM_WORKERS,
    )
    
    
    return train_loader 


def get_data_to_gpu(path):

    data = np.load(path)
    tensor_data = torch.tensor(data).to("cuda")
    dataset = torch.utils.data.TensorDataset(tensor_data)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    print(f"Loaded data to GPU: {tensor_data.shape}")
    return loader


