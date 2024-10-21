import numpy as np
import torch
from torch import nn



n_0 = 1792 
n_1 = 1280

n_0 = 1792 
n_1 = 1024 

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1, compressed_dim=1):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_0),
            nn.BatchNorm1d(n_0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_0, n_1),
            nn.BatchNorm1d(n_1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_1, compressed_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(compressed_dim, n_1),
            nn.BatchNorm1d(n_1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_1, n_0),
            nn.BatchNorm1d(n_0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_0, input_dim),
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed, compressed

device = torch.device("cuda")



# Load the data
data_1 = np.load("../generate_data/embeddings_data/new_e5_wiki_500k/train_embeddings.npy")
data_2 = np.load("../generate_data/embeddings_data/new_mxbai_wiki_500k/train_embeddings.npy")

data_3 = np.load("../generate_data/embeddings_data/new_e5_wiki_500k/val_embeddings.npy")
data_4 = np.load("../generate_data/embeddings_data/new_mxbai_wiki_500k/val_embeddings.npy")

# Concatenate the data along axis 1
data_train = np.concatenate((data_1, data_2), axis=1)
data_val   = np.concatenate((data_3, data_4), axis=1)

# Initialize the AutoEncoder and load the saved state

INPUT_DIM = 2048 
COMPRESSED_DIM = 1024 
CHECKPOINT_PATH = "008700.pth"
autoencoder_path = f"models_pth/{INPUT_DIM}_{COMPRESSED_DIM}/{CHECKPOINT_PATH}"

autoencoder = AutoEncoder(input_dim=INPUT_DIM, compressed_dim=COMPRESSED_DIM)
autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))

autoencoder.to(device)
autoencoder.eval()  # Set to evaluation mode

# Convert NumPy arrays to PyTorch tensors
data_train_tensor = torch.from_numpy(data_train).float().to(device)
data_val_tensor = torch.from_numpy(data_val).float().to(device)

# Perform inference without tracking gradients
with torch.no_grad():
    autoencoder_train  = autoencoder(data_train_tensor)[1]
    autoencoder_val    = autoencoder(data_val_tensor)[1]

autoencoder_train_np = autoencoder_train.cpu().numpy()
autoencoder_val_np   = autoencoder_val.cpu().numpy()

np.save(f"autoencoder_train_{COMPRESSED_DIM}.npy", autoencoder_train_np)
np.save(f"autoencoder_val_{COMPRESSED_DIM}.npy", autoencoder_val_np)

print("Autoencoder outputs have been saved successfully.")

print(autoencoder_train_np.shape)
print(autoencoder_val_np.shape)

#mse = nn.MSELoss()
#print(mse(data_train_tensor, autoencoder_train))
#print(mse(data_val_tensor, autoencoder_val))