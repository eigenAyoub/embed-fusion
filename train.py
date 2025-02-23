from pathlib import Path
from typing import List, Dict, Optional

import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import get_data

from loss import SimilarityLoss, SimilarityLossTopK, Similarity

from model import EncoderOnly 

from config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    STEP_SIZE,
    GAMMA,
    NUM_EPOCHS,
)


class Trainer:
    """Handles model training, validation, and visualization"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        enc_config: Dict, # in / out
        dims,             # mrl dims
        now: str,         # checkpoints tag
        device: str = DEVICE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        loss_weight: float = 1.0,
        use_topk: bool = False,
        k: int = 10,
        checkpoint_dir: str = "checkpoints",
        save_freq: int = 5,  # Save every N iepochs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.enc_config = enc_config
        self.now = now 
        self.mrl_dims = dims
        
        self.criterion = SimilarityLoss(weight=1.0)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train_epoch(self) -> float:
        """Run one epoch of training"""
        self.model.train()
        running_loss = 0.0
        
        for batch in self.train_loader:
            inputs = batch.to(self.device)
            self.optimizer.zero_grad()
           
            loss = sum(
                self.criterion(self.model(inputs, dim), inputs)  
                for dim in self.mrl_dims
            )
            
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader.dataset)

    def validate(self) -> float:
        """Run validation"""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch.to(self.device)
                loss = sum(
                    self.criterion(self.model(inputs, dim), inputs)  # Use self.criterion
                    for dim in self.mrl_dims 
                )
                running_loss += loss.item()
                
        return running_loss / len(self.val_loader.dataset)

    def save_checkpoint(self, epoch: int, val_loss: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        inDim  = self.enc_config["input_dim"]
        outDim = self.enc_config["output_dim"]

        # Regular checkpoint
        if epoch % self.save_freq == 0:
            path = self.checkpoint_dir /      f'{inDim}_{outDim}_ep_{epoch:03d}_{self.now}.pth'
            torch.save(checkpoint, path)
            print(f"Saved checkpoint to {path}")
            
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{inDim}_{outDim}_ep_{epoch:03d}_{self.now}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def train(self, num_epochs: int = NUM_EPOCHS):
        """Main training loop with checkpoint saving"""
        for epoch in range(num_epochs):

            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
            
            # Save checkpoints
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch + 1, is_best, val_loss)
            

def main():
  
    tag = "all_4"
    
    train_loader = get_data(f"generate_data/embeddings_data/{tag}/train_embeddings.npy")
    val_loader = get_data(f"generate_data/embeddings_data/{tag}/val_embeddings.npy")
    
    model_config = {
                    'input_dim':  1920,
                    'output_dim': 1024,
                }

    inDim = model_config["input_dim"]
    outDim = model_config["output_dim"]
    COMPRESSED_DIMENSIONS = [32, 64, 128, 256, 384, 512, 768, 1024, outDim]
    
    model = EncoderOnly(model_config)
    
    now = datetime.datetime.now().strftime("%H%M%S")
    
    log_line = f"run {now} {inDim} {outDim} {COMPRESSED_DIMENSIONS} Obs\n"
    with open("logs.txt", "a") as f:
        f.write(log_line)

    trainer = Trainer(
        model, 
        train_loader, 
        val_loader,
        checkpoint_dir=f"models_pth/{inDim}_{outDim}",
        enc_config=model_config,
        now = now,
        dims = COMPRESSED_DIMENSIONS,
        save_freq=5
    )


    trainer.train()

if __name__ == "__main__":
    main()
