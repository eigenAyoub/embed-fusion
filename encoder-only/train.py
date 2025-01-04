from pathlib import Path
from typing import List, Dict, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import get_data

from encoder_simple import EncoderOnly, EncoderConfig, initialize_weights


from config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    STEP_SIZE,
    GAMMA,
    NUM_EPOCHS,
    PATIENCE,
    PLOT_PATH,
    COMPRESSED_DIMENSIONS,
)

model_config = EncoderConfig.DEFAULT

inDim = model_config["input_dim"]
outDim = model_config["max_compressed_dim"]

class SimilarityLoss(nn.Module):
    """Loss based on cosine similarity between all pairs in batch"""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sim_outputs = nn.functional.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = nn.functional.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, 0)
        sim_targets = sim_targets.masked_fill(mask, 0)
        
        loss = torch.mean((sim_outputs - sim_targets) ** 2)
        return self.weight * loss

class SimilarityLossTopK(nn.Module):
    """Loss based on top-k cosine similarities between pairs in batch"""
    def __init__(self, weight: float = 1.0, k: int = 10):
        super().__init__()
        self.weight = weight
        self.k = k

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sim_outputs = nn.functional.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = nn.functional.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, -float('inf'))
        sim_targets = sim_targets.masked_fill(mask, -float('inf'))

        topk_values, topk_indices = sim_targets.topk(self.k, dim=-1)
        topk_sim_outputs = sim_outputs.gather(1, topk_indices)
        loss = torch.mean((topk_sim_outputs - topk_values) ** 2)
        return self.weight * loss

class Trainer:
    """Handles model training, validation, and visualization"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = DEVICE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        loss_weight: float = 1.0,
        use_topk: bool = False,
        k: int = 10,
        checkpoint_dir: str = "checkpoints",
        save_freq: int = 5,  # Save every N epochs

    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = (
            SimilarityLossTopK(weight=loss_weight, k=k) if use_topk 
            else SimilarityLoss(weight=loss_weight)
        )
        
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
                self.criterion(self.model(inputs, dim), inputs)  # Use self.criterion
                for dim in COMPRESSED_DIMENSIONS
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
                    for dim in COMPRESSED_DIMENSIONS
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

        tag = f"epoch_{epoch:03d}_loss_{val_loss:.6f}.pth"

        # Regular checkpoint
        if epoch % self.save_freq == 0:
            path = self.checkpoint_dir / f'checkpoint_{epoch:03d}.pth'
            torch.save(checkpoint, path)
            print(f"Saved checkpoint to {path}")
            
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / tag 
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
            
        self.plot_losses()

    def plot_losses(self):
        """Plot training history"""
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        plt.close()

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    train_loader = get_data("bge-arctic-train.npy")
    val_loader = get_data("bge-arctic-val.npy")
    
    model = EncoderOnly(EncoderConfig.DEFAULT)
    model.apply(initialize_weights)
    
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader,
        checkpoint_dir=f"models_pth/{inDim}_{outDim}",
        save_freq=5
    )
    trainer.train()


if __name__ == "__main__":
    main()