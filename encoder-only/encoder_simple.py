# model.py

import torch.nn as nn
import torch


class EncoderConfig:
    DEFAULT = {
        'input_dim': 1152,
        'hidden1_dim': 896,
        #'hidden2_dim': 896,
        'max_compressed_dim': 768
    }

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super(EncoderOnly, self).__init__()
        
        # Use provided config or defaults
        cfg = config or EncoderConfig.DEFAULT
        
        self.encoder = nn.Sequential(
            nn.Linear(cfg['input_dim'], cfg['hidden1_dim']),
            nn.BatchNorm1d(cfg['hidden1_dim']),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Linear(cfg['hidden1_dim'], cfg['hidden2_dim']),
            #nn.BatchNorm1d(cfg['hidden2_dim']),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.Linear(cfg['hidden2_dim'], cfg['max_compressed_dim'])
            nn.Linear(cfg['hidden1_dim'], cfg['max_compressed_dim'])
        )

    def forward(self, x, compressed_dim):
        encoded = self.encoder(x)
        return encoded[:, :compressed_dim]

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming/He initialization - good for LeakyReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0 )