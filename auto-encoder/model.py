# model.py

import torch.nn as nn
from config import INPUT_DIM, COMPRESSED_DIM


n_0 = 1792 
n_1 = 1280
n_0 = 1792 
n_1 = 1024 

n0 = 768
n1 = 512
n2 = 384

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1, compressed_dim=1):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1024, n0),
            nn.BatchNorm1d(n0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n0, n1),
            nn.BatchNorm1d(n1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n1, n2),
            nn.BatchNorm1d(n2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n2, 128),

        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(128, n2),
            nn.BatchNorm1d(n2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n2, n1),
            nn.BatchNorm1d(n1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n1, n0),
            nn.BatchNorm1d(n0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n0, 1024),
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed, compressed

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
