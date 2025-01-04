from dataclasses import dataclass
from typing import List, Optional
import torch.nn as nn
import torch

@dataclass
class EncoderConfig:
    """Enhanced configuration for Encoder architecture"""
    DEFAULT = {
        'input_dim': 1152,
        'hidden_dims': [1024, 896],  # List of hidden dimensions
        'max_compressed_dim': 512,
        'dropout': 0.1,
        'activation': 'leaky_relu',
        'use_layer_norm': True,
        'use_residual': True
    }

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        # Activation function mapping
        activations = {
            'leaky_relu': lambda: nn.LeakyReLU(0.2, inplace=True),
            'gelu': nn.GELU,
            'relu': nn.ReLU
        }
        activation = activations[cfg['activation']]()
        
        # Build encoder layers
        layers = []
        prev_dim = cfg['input_dim']
        
        for hidden_dim in cfg['hidden_dims']:
            # Main layer block
            block = []
            block.append(nn.Linear(prev_dim, hidden_dim))
            
            if cfg['use_layer_norm']:
                block.append(nn.LayerNorm(hidden_dim))
            else:
                block.append(nn.BatchNorm1d(hidden_dim))
                
            block.append(activation)
            block.append(nn.Dropout(cfg['dropout']))
            
            # Add residual connection if dimensions match
            if cfg['use_residual'] and prev_dim == hidden_dim:
                layers.append(ResidualBlock(nn.Sequential(*block)))
            else:
                layers.extend(block)
                
            prev_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(prev_dim, cfg['max_compressed_dim']))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, compressed_dim: int) -> torch.Tensor:
        encoded = self.encoder(x)
        return encoded[:, :compressed_dim]

class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)