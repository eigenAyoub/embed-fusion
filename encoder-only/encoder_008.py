from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class EncoderConfig:
    """Enhanced configuration for modern encoder"""
    DEFAULT = {
        'input_dim': 1152,
        'hidden_dims': [1024, 896, 768],
        'max_compressed_dim': 512,
        'num_heads': 4,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'use_layer_norm': True,
        'use_residual': True,
        'activation': 'gelu'
    }

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        self.input_proj = nn.Linear(cfg['input_dim'], cfg['hidden_dims'][0])
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, cfg['hidden_dims'][0]))
        
        layers = []
        # Create sequential blocks for each dimension transition
        for i in range(len(cfg['hidden_dims']) - 1):
            layers.extend([
                nn.LayerNorm(cfg['hidden_dims'][i]),
                nn.Linear(cfg['hidden_dims'][i], cfg['hidden_dims'][i+1]),
                nn.GELU(),
                nn.Dropout(cfg['dropout'])
            ])
        
        self.blocks = nn.Sequential(*layers)
        
        # Final normalization and projection
        self.final_norm = nn.LayerNorm(cfg['hidden_dims'][-1])
        self.out_proj = nn.Linear(cfg['hidden_dims'][-1], cfg['max_compressed_dim'])

    def forward(self, x: Tensor, compressed_dim: int) -> Tensor:
        # Project input
        x = self.input_proj(x)
        
        # Process through blocks
        x = self.blocks(x)
        
        # Final layer norm and projection
        x = self.final_norm(x)
        x = self.out_proj(x)
        
        return x[:, :compressed_dim]

