import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderConfig:
    DEFAULT = {
        'input_dim': 1152,
        'hidden_dim': 1024,  # new hidden dimension for deeper MLP
        'output_dim': 512,
        'dropout': 0.1
    }

class ResidualMLPBlock(nn.Module):
    """
    Example residual block. You can remove or replace this block with 
    simple linear -> BN -> ReLU sequences if you prefer.
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Residual connection
        return self.activation(x + self.block(x))

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT

        self.preprocess = nn.Sequential(
            # BN on input
            nn.BatchNorm1d(cfg['input_dim']),
            nn.Linear(cfg['input_dim'], cfg['hidden_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg['dropout'])
        )

        # One or more residual blocks
        self.resblock1 = ResidualMLPBlock(
            dim=cfg['hidden_dim'],
            hidden_dim=cfg['hidden_dim'],
            dropout=cfg['dropout']
        )
        
        # Final projection to the output embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(cfg['hidden_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization for Linear, and standard initialization for BatchNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, dim):
        # Example forward:
        x = self.preprocess(x)
        x = self.resblock1(x)
        x = self.projection(x)

        # L2 normalize final embedding
        out = F.normalize(x[:,:dim], p=2, dim=1)
        return out


class SimilarityLossF(nn.Module):
    """
    Example: pairwise similarity MSE, excluding the diagonal from the average.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute cosine similarity for all pairs
        sim_outputs = F.cosine_similarity(
            model_output.unsqueeze(1), 
            model_output.unsqueeze(0), 
            dim=-1
        )
        sim_targets = F.cosine_similarity(
            targets.unsqueeze(1), 
            targets.unsqueeze(0), 
            dim=-1
        )

        # Create mask for the diagonal (self-similarities)
        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()

        # Exclude diagonal from the average
        sim_outputs = sim_outputs[~mask]
        sim_targets = sim_targets[~mask]

        loss = torch.mean((sim_outputs - sim_targets) ** 2)
        return self.weight * loss

