import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime 

from loss import SimilarityLoss
from data_loader import get_data_to_gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



class TheSimpleNet(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=4096):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
    
    def forward(self, x):
        z = self.fc(x)
        binary = (z > 0).float() 
        return binary
    
    def encode(self, x):
        return self.fc(x)
    


"""
Move this somewhere:

## To simply generate a (hopefully better threshold):
## For the combined-s:

import numpy as np

tag = "no-ins-gte-small"
student_val_path = f"generate_data/{tag}/val_embeddings.npy"
val_ds = np.load(student_val_path)
val_ds = torch.tensor(val_ds) # please switch all to torch.

m = TheSimpleNet(768, 4096)
m.load_state_dict(torch.load("chillDude.pth"))

acts = m.encode(val_ds)  

thresholds = acts.median(dim=0)

encoded = (acts < thresholds).float()
"""



    
    

class TheSigNet(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=4096):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
    
    def forward(self, x):
        z = self.fc(x)
        return F.sigmoid(z)

class TheTanNet(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=4096):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
    
    def forward(self, x):
        z = self.fc(x)
        return F.tanh(z)
    
    
tanhLoss   = lambda x : F.mse_loss(x.abs(), torch.ones_like(x))
sigLoss    = lambda x : (x*(1-x)).mean()

bitBalance = lambda batch : F.mse_loss(batch.mean(dim=0), torch.full_like(batch.mean(dim=0),1/2))


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    collected_outputs = [] 
    
    for i, batch in enumerate(dataloader):

        student_inputs = batch[0]
        optimizer.zero_grad()

        outputs = model(student_inputs)
        
        if i < 10 :
            collected_outputs.append(outputs.detach().cpu()) 

        #loss = criterion(outputs, student_inputs) + 0.01*sigLoss(outputs) + 0.05*bitBalance(outputs)
        loss = criterion(outputs, student_inputs) + 0.1*bitBalance(outputs)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * student_inputs.size(0)
    
    aggregated = torch.cat(collected_outputs, dim=0)
    print(aggregated)
    plot_output_distribution(aggregated, epoch, "SigNet")

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            student_inputs = batch[0]
            outputs = model(student_inputs)
            loss = criterion(outputs, student_inputs) + 0.1*bitBalance(outputs)
            running_loss += loss.item() * student_inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def plot_output_distribution(outputs, epoch, model_name):
    # Flatten the outputs to a 1D array
    outputs_flat = outputs.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(outputs_flat, bins=50, density=True, alpha=0.75)
    plt.title(f"{model_name} Output Distribution at Epoch {epoch}")
    plt.xlabel("Output Value")
    plt.ylabel("Density")

    filename = f"figs/{model_name}_epoch_{epoch}.png"
    plt.savefig(filename)
    plt.close()



def main():

    tag = "no-ins-gte-small"
    student_train_path = f"generate_data/{tag}/train_embeddings.npy"
    student_val_path = f"generate_data/{tag}/val_embeddings.npy"
    

    input_dim = 768
    embedding_dim = 4096

    num_epochs = 40
    learning_rate = 4e-4

    device = "cuda" 
    
    val_loader   = get_data_to_gpu(student_val_path)
    train_loader = get_data_to_gpu(student_train_path)
   
    run_id = datetime.datetime.now().strftime("%H%M%S")
    print(run_id)

    model = TheSigNet(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    criterion = SimilarityLoss(weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss   = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        torch.save(checkpoint, f'checks/checkpoint_epoch_{epoch}_{run_id}.pth')
        print(f"Checkpoint saved for epoch {epoch}")


if __name__ == "__main__":
    main()

