from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class EncoderConfig:
    """Configuration for cross-attention encoder"""
    DEFAULT = {
        'input_dim': 1152,  # Total dimension (386 + 766)
        'query_dim': 384,   # First model output
        'kv_dim': 768,     # Second model output
        'hidden_dims': [600, 512],
        'max_compressed_dim': 512,
        'num_heads': 24,
        'dropout': 0.1,
        'attention_dropout': 0.1
    }

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.kv_dim = kv_dim
        
        # Calculate dimensions - make hidden_dim match kv_dim for consistency
        self.head_dim = 32  # Standard size per head (128/4)
        hidden_dim = self.head_dim * num_heads  # 32 * 20 = 640  
        self.scale = self.head_dim ** -0.5
        
        # Project query from query_dim to hidden_dim
        self.q_proj = nn.Linear(query_dim, hidden_dim)  # [768 -> 128]
        self.k_proj = nn.Linear(kv_dim, hidden_dim)     # [128 -> 128]
        self.v_proj = nn.Linear(kv_dim, hidden_dim)     # [128 -> 128]
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, kv_dim)       # [128 -> 128]
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        B = q.shape[0]  # Batch size
        
        # Project to hidden dimension
        q = self.q_proj(q)  # [B, 128]
        k = self.k_proj(kv)  # [B, 128]
        v = self.v_proj(kv)  # [B, 128]
        
        # Reshape for multi-head attention: [B, 128] -> [B, 4, 32]
        q = q.reshape(B, self.num_heads, self.head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim)
        
        # Rest of the code remains the same
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).reshape(B, -1)
        x = self.proj(x)
        return self.proj_drop(x)

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        self.query_dim = cfg['query_dim']
        self.kv_dim = cfg['kv_dim']
        
        # Cross attention between two embedding spaces
        self.cross_attn = CrossAttention(
            query_dim=self.query_dim,
            kv_dim=self.kv_dim,
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout']
        )
        
        # Processing layers
        layers = []
        prev_dim = self.kv_dim

        # In EncoderOnly, replace the loop with:
        for hidden_dim in cfg['hidden_dims']:
            layers.extend([
                nn.LayerNorm(prev_dim),
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg['dropout'])
            ])
            prev_dim = hidden_dim
        
        self.blocks = nn.Sequential(*layers)
        
        # Final projection
        self.final_norm = nn.LayerNorm(prev_dim)
        self.out_proj = nn.Linear(prev_dim, cfg['max_compressed_dim'])
        
    def forward(self, x: Tensor, compressed_dim: int) -> Tensor:
        # Split input into query and key-value components
        query = x[:, :self.query_dim]
        key_value = x[:, self.query_dim:]
        
        # Cross attention between components
        x = self.cross_attn(query, key_value)
        
        # Process through blocks
        x = self.blocks(x)
        
        # Final projection
        x = self.final_norm(x)
        x = self.out_proj(x)
        
        return x[:, :compressed_dim]
