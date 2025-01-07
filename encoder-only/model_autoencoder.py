# model.py

import torch.nn as nn
from config import INPUT_DIM, COMPRESSED_DIM

input_dim = INPUT_DIM 
compressed_dim = COMPRESSED_DIM 

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, compressed_dim=COMPRESSED_DIM):
        super(AutoEncoder, self).__init__()

        n0 = 1024 # First hidden layer size
        n1 = 768  # Second hidden layer size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n0),           # encoder.0
            nn.BatchNorm1d(n0),                 # encoder.1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n0, n1),                  # encoder.3
            nn.BatchNorm1d(n1),                 # encoder.4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n1, compressed_dim),      # encoder.6
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, n1),      # decoder.0
            nn.BatchNorm1d(n1),                 # decoder.1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n1, n0),                  # decoder.3
            nn.BatchNorm1d(n0),                 # decoder.4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n0, input_dim),           # decoder.6
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
