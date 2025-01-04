# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path

from data_loader import get_data_loaders, get_data

from encoder_only import EncoderOnly, initialize_weights, EncoderConfig


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
    COMPRESSED_DIMENSIONS,  # Updated import
    LOSS_WEIGHTS,
    INPUT_DIM,
    MAX_COMPRESSED_DIM
)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

print("COMPRESSED_DIMENSIONS", COMPRESSED_DIMENSIONS)

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
    plt.title('Validation Loss per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()

def train():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Prepare data loaders
    train_loader = get_data("bge-arctic-train.npy")
    val_loader = get_data("bge-arctic-val.npy")

    encoder_config = EncoderConfig.DEFAULT 

    model = EncoderOnly(encoder_config).to(DEVICE)

    model.apply(initialize_weights)

    class SimilarityLoss(nn.Module):
        def __init__(self, weight=1.0):
            super(SimilarityLoss, self).__init__()
            self.weight = weight

        def forward(self, model_output, targets):

            sim_outputs = nn.functional.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
            sim_targets = nn.functional.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

            mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
            sim_outputs = sim_outputs.masked_fill(mask, 0)
            sim_targets = sim_targets.masked_fill(mask, 0)
            
            loss = torch.mean((sim_outputs - sim_targets) ** 2)

            return self.weight * loss


    class SimilarityLossTopK(nn.Module):
        def __init__(self, weight=1.0, k=10):
            super(SimilarityLossTopK, self).__init__()
            self.weight = weight
            self.k = k  # Number of top similarities to consider

        def forward(self, model_output, targets):
            #assert model_output.shape == targets.shape, f"Input dimensions mismatch: got {model_output.shape} and {targets.shape}"

            # Compute pairwise cosine similarities in the adapted and original spaces
            sim_outputs = nn.functional.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
            sim_targets = nn.functional.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

            # Exclude self-similarities by masking the diagonal
            mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
            sim_outputs = sim_outputs.masked_fill(mask, -float('inf'))
            sim_targets = sim_targets.masked_fill(mask, -float('inf'))

            # Identify the indices of the top k similarities in the original space
            topk_values, topk_indices = sim_targets.topk(self.k, dim=-1)

            # Gather the corresponding similarities in the adapted space
            topk_sim_outputs = sim_outputs.gather(1, topk_indices)

            # Compute the loss over the top k similarities
            loss = torch.mean((topk_sim_outputs - topk_values) ** 2)

            return self.weight * loss


    # Define the loss function and optimizer
    #criterion = nn.MSELoss()
    #criterion = SimilarityLossTopK(weight=1.0, k=15)

    criterion = SimilarityLoss(weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Training Loop 
    train_losses_batches = []  
    val_losses_epochs = []     
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    trigger_times = 0

    ensure_dir(RECONSTRUCTIONS_DIR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch_inputs = batch.to(DEVICE)  
            optimizer.zero_grad()
            
            batch_loss = 0.0
            
            for dim in COMPRESSED_DIMENSIONS:
                outputs = model(batch_inputs, compressed_dim=dim)
                loss = criterion(outputs, batch_inputs)
                batch_loss += loss 

            # Backpropagate the total loss
            batch_loss.backward()
            optimizer.step()
            
            # Accumulate the loss for monitoring
            running_loss += batch_loss.item() * batch_inputs.size(0)
            train_losses_batches.append(batch_loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
            
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_inputs = batch.to(DEVICE)
                batch_loss = 0.0
                for dim in COMPRESSED_DIMENSIONS:
                    outputs = model(batch_inputs, compressed_dim=dim)
                    loss = criterion(outputs, batch_inputs)
                    batch_loss += loss
                val_running_loss += batch_loss.item() * batch_inputs.size(0)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_epoch_loss:.6f}')
        
        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            trigger_times = 0

            parent_dir     = Path(f"models_pth/{INPUT_DIM}_{MAX_COMPRESSED_DIM}")
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
        
        # Optionally visualize reconstructions every few epochs
        # if (epoch + 1) % 10 == 0:
        #     visualize_reconstructions(model, val_loader, epoch+1)

    # Optionally plot the losses
    # plot_losses(train_losses_batches, val_losses, num_epochs=NUM_EPOCHS)

def visualize_reconstructions(model, data_loader, epoch):
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        batch_inputs = batch[0].to(DEVICE)
        sample = batch_inputs[:5]
        # Choose a specific compressed dimension to visualize
        selected_dim = 128  # Example
        reconstructed = model(sample, compressed_dim=selected_dim)
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
            plt.title(f'Reconstructed (dim={selected_dim})')
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
