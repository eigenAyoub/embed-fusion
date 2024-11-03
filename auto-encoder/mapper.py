import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
path_1 = "../data/mix_train_embeddings.npy"

path_768 = "mapper_data/autoencoder_train_768.npy"
path_1024 = "mapper_data/autoencoder_train_1024.npy"

path_bge_small = "../generate_data/embeddings_data/new_bge-small_wiki_500k/train_embeddings.npy"
path_e5_small = "../generate_data/embeddings_data/new_e5-small_wiki_500k/train_embeddings.npy"

path_e5 = "../generate_data/embeddings_data/new_e5_wiki_500k/train_embeddings.npy"
path_mxbai = "../generate_data/embeddings_data/new_mxbai_wiki_500k/train_embeddings.npy"

data_2 = np.load(path_bge_small)
data_3 = np.load(path_e5_small)

input_data = np.concatenate((data_2, data_3), axis=1)

e5_data = np.load(path_e5)
mxbai_data = np.load(path_mxbai)
original_data = np.concatenate((e5_data, mxbai_data), axis=1)

target_768 = np.load(path_768)    # N x 768
target_1024 = np.load(path_1024)  # N x 1024

# Normalize the data (optional)
# scaler_input = StandardScaler()
# input_data = scaler_input.fit_transform(input_data)
# scaler_target_768 = StandardScaler()
# target_768 = scaler_target_768.fit_transform(target_768)
# scaler_target_1024 = StandardScaler()
# target_1024 = scaler_target_1024.fit_transform(target_1024)

# Convert to tensors
input_tensor = torch.from_numpy(input_data).float()
target_768_tensor = torch.from_numpy(target_768).float()
target_1024_tensor = torch.from_numpy(target_1024).float()
original_tensor = torch.from_numpy(original_data).float()

# Split into training and validation sets
x_train, x_val, y_train_768, y_val_768, y_train_1024, y_val_1024, original_train, original_val = train_test_split(
    input_tensor, target_768_tensor, target_1024_tensor, original_tensor, test_size=0.1, random_state=42
)

# Create DataLoaders
batch_size = 16

train_dataset = TensorDataset(x_train, y_train_768, y_train_1024, original_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(x_val, y_val_768, y_val_1024, original_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the model
class MappingNet(nn.Module):
    def __init__(self):
        super(MappingNet, self).__init__()

        self.layer_768_2048 = nn.Sequential(
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer_2048_1024 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer_1024_768 = nn.Sequential(
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer_768_1024 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer_1024_2048 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
#        self.final_layer = nn.Linear(1024, 768)
        
        self._initialize_weights()
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        out1 = self.layer_768_2048(x)       # 2048
        out2 = self.layer_2048_1024(out1)   # 1024
        out3 = self.layer_1024_768(out2)  + x  # 768
        out4 = self.layer_768_1024(out3)    # 1024
        out5 = self.layer_1024_2048(out4)   # 2048
        return out3, out2, out4, out1, out5

# then  add residual connections.
# slowly: +x then +out2 the +out5
# remove leaky relu on the 768 layer 
# add a new filler layer between 2048 and 1024 (another 1024 that is not used for the loss)


# Initialize the model
model = MappingNet()
model._initialize_weights()

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)

# Training loop with validation and early stopping
num_epochs = 30
best_val_loss = float('inf')
patience = 5
trigger_times = 0


for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    step = 0 
    for batch_inputs, batch_tar_768, batch_tar_1024, batch_original in train_loader:
        
        batch_inputs   = batch_inputs.to(device)
        batch_tar_768  = batch_tar_768.to(device)
        batch_tar_1024 = batch_tar_1024.to(device)
        batch_original = batch_original.to(device)

        step += 1

        optimizer.zero_grad()

        out3, out2, out4, out1, out5 = model(batch_inputs)
        
        #loss1024 = criterion(out2, out4)  
        loss2048 = criterion(out1, out5)  
        loss2    = criterion(out1, batch_original) 
        loss3    = criterion(out5, batch_original) 

        #rest     = criterion(out5, batch_original)
        
        #loss =  loss1024 + loss2048 + loss2 + loss3
        loss =   loss2048 + loss2  + loss3

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        #if step % 100 == 0:
        #    avg_loss_so_far = total_train_loss / step
        #    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}], "
        #          f"Current Batch Loss: {loss.item():.6f}, "
        #          f"Average Loss So Far: {avg_loss_so_far:.6f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_tar_768, batch_tar_1024, batch_original in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_tar_768 = batch_tar_768.to(device)
            batch_tar_1024 = batch_tar_1024.to(device)
            batch_original = batch_original.to(device)


            #loss1024 = criterion(out2, out4)  
            loss2048 = criterion(out1, out5)  
            loss2    = criterion(out1, batch_original) 
            loss3    = criterion(out5, batch_original) 

            loss =   loss2048 + loss2  + loss3

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.6f}, "
          f"Val Loss: {avg_val_loss:.6f}")
#    print(f"Reconstruction loss: {loss2048.item(), loss1024.item()}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        # Save the best model
        print("> Model saved!")
        torch.save(model.state_dict(), 'best_model_jj.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

print("Training completed.")