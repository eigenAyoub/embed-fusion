import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import SimilarityLoss
from data_loader import get_data_to_gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class HardThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, beta=10):
        # Save inputs for backward; beta controls the steepness of the surrogate gradient.
        ctx.save_for_backward(x, threshold)
        ctx.beta = beta
        # Hard threshold: output 1 if (x - threshold) > 0, else 0.
        return (x - threshold > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        beta = ctx.beta
        # Surrogate gradient: derivative of a steep sigmoid.
        sig = torch.sigmoid(beta * (x - threshold))
        grad_x = grad_output * beta * sig * (1 - sig)
        # The threshold is broadcasted along the batch dimension, so sum the gradients over that dimension.
        grad_threshold = -(grad_output * beta * sig * (1 - sig)).sum(dim=0)
        return grad_x, grad_threshold, None  # No gradient for beta

# The network that maps from 768 to 4096 dimensions with learnable per-feature thresholds.
class BinaryEmbeddingNet(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=4096, beta=10):
        super(BinaryEmbeddingNet, self).__init__()

        self.fc = nn.Linear(input_dim, embedding_dim)
        self.threshold = nn.Parameter(torch.zeros(embedding_dim))
        self.beta = beta
    
    def forward(self, x):
        x = self.fc(x)
        # Apply the custom hard-threshold function.
        binary = HardThresholdSTE.apply(x, self.threshold, self.beta)
        return binary

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:

        student_inputs = batch[0]
        optimizer.zero_grad()

        outputs = model(student_inputs)

        loss = criterion(outputs, student_inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * student_inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            student_inputs = batch[0]
            outputs = model(student_inputs)
            loss = criterion(outputs, student_inputs)
            running_loss += loss.item() * student_inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def main():

    tag = "no-ins-gte-small"
    student_train_path = f"generate_data/{tag}/train_embeddings.npy"
    student_val_path = f"generate_data/{tag}/val_embeddings.npy"
    
    # Hyperparameters.
    input_dim = 768
    embedding_dim = 4096

    beta = 10

    num_epochs = 10
    learning_rate = 1e-3
    device = "cuda" 
    
    val_loader   = get_data_to_gpu(student_val_path)
    train_loader = get_data_to_gpu(student_train_path)
    
    model = BinaryEmbeddingNet(input_dim=input_dim, embedding_dim=embedding_dim, beta=beta).to(device)
    criterion = SimilarityLoss(weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()

