# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path

from data_loader import get_data_loaders
from model import AutoEncoder, initialize_weights
from config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    STEP_SIZE,
    GAMMA,
    NUM_EPOCHS,
    #BEST_MODEL_PATH,
    PATIENCE,
    RECONSTRUCTIONS_DIR,
    PLOT_PATH,
    COMPRESSED_DIM,
    INPUT_DIM
)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def plot_losses(train_losses_flat, val_losses, num_epochs=NUM_EPOCHS):
    """
    Plots training loss (per batch) and validation loss (per epoch).
    """
    plt.figure(figsize=(14, 6))

    # Plot Training Loss per Batch
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_flat, label='Train Loss (First 3 Epochs)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch (First 3 Epochs)')
    plt.legend()

    # Plot Validation Loss per Epoch
    plt.subplot(1, 2, 2)
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, val_losses[:num_epochs], marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch (First 3 Epochs)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("hey.png")
    plt.show()

def train():
    # Set TOKENIZERS_PARALLELISM environment variable
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Prepare data loaders
    train_loader, val_loader = get_data_loaders()

    # Initialize the model
    model = AutoEncoder(INPUT_DIM, COMPRESSED_DIM).to(DEVICE)
    model.apply(initialize_weights)

    # loss and optimizer
    
    class CombinedLoss(nn.Module):
        def __init__(self, mse_weight=1.0, cosine_weight=1.0):
            super(CombinedLoss, self).__init__()
            self.mse = nn.MSELoss()
            self.cosine = nn.CosineSimilarity(dim=1)
            self.mse_weight = mse_weight
            self.cosine_weight = cosine_weight

        def forward(self, output, target):
            loss_mse = self.mse(output, target)
            loss_cosine = 1 - self.cosine(output, target).mean()
            return self.mse_weight * loss_mse + self.cosine_weight * loss_cosine

    # Use CombinedLoss instead of MSELoss
    #criterion = CombinedLoss(mse_weight=0.99, cosine_weight=0.1)   
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Training Loop 
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    trigger_times = 0

    ensure_dir(RECONSTRUCTIONS_DIR)

    train_losses_batches = []  
    val_losses_epochs = []     

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_inputs in train_loader:

            batch_inputs = batch_inputs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_inputs.size(0)
            train_losses_batches.append(loss.item())
             
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
            
        # Validation
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for batch_inputs in val_loader:
                batch_inputs = batch_inputs.to(DEVICE)
                outputs, _ = model(batch_inputs)
                loss = criterion(outputs, batch_inputs)
                val_running_loss += loss.item() * batch_inputs.size(0)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_epoch_loss:.6f}')
        
        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            trigger_times = 0

            parent_dir     = Path(f"models_pth/{INPUT_DIM}_{COMPRESSED_DIM}")
            checkpoint_tag = f"{val_epoch_loss:.6f}"[2:] + ".pth"
            checkpoint_dir = parent_dir / checkpoint_tag 
            
            parent_dir.mkdir(parents=True, exist_ok=True)


            torch.save(model.state_dict(), checkpoint_dir)
            print(f'Best model saved with Val Loss: {best_val_loss:.6f}')
        else:
            trigger_times += 1
            print(f'EarlyStopping counter: {trigger_times} out of {PATIENCE}')
            if trigger_times >= PATIENCE:
                print('Early stopping triggered')
                break
        
        #if (epoch + 1) % 10 == 0:
        #    visualize_reconstructions(model, val_loader, epoch+1)

    plot_losses(train_losses_batches, val_losses, num_epochs=5)
    
    
def visualize_reconstructions(model, data_loader, epoch):
    model.eval()
    with torch.no_grad():
        batch_inputs = next(iter(data_loader))
        batch_inputs = batch_inputs.to(DEVICE)
        sample = batch_inputs[:5]
        reconstructed, _ = model(sample)
        sample = sample.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        for i in range(5):
            plt.figure(figsize=(12, 4))
            
            # Plot Original
            plt.subplot(1, 2, 1)
            plt.title('Original')
            plt.plot(sample[i])
            plt.xlabel('Feature Dimension')
            plt.ylabel('Value')
            
            # Plot Reconstructed
            plt.subplot(1, 2, 2)
            plt.title('Reconstructed')
            plt.plot(reconstructed[i])
            plt.xlabel('Feature Dimension')
            plt.ylabel('Value')
            
            # Adjust layout for better spacing
            plt.tight_layout()
            
            # Define the filename with epoch and sample index
            filename = f'reconstruction_epoch_{epoch}_sample_{i+1}.png'
            filepath = os.path.join(RECONSTRUCTIONS_DIR, filename)
            
            # Save the figure
            plt.savefig(filepath)
            plt.close()  # Close the figure to free memory
            
            print(f'Saved reconstruction plot: {filepath}')

if __name__ == "__main__":
    train()

