import torch.nn as nn

class EncoderConfig:
    DEFAULT = {
        'input_dim': 1152,
        'hidden_dims': [1024, 896, 832, 768],  # Progressive reduction
        'max_compressed_dim': 768,
        'dropout': 0.1
    }

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return x + self.block(x)

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        layers = []
        in_dim = cfg['input_dim']
        
        # Progressive dimension reduction
        for hidden_dim in cfg['hidden_dims']:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                ResidualBlock(hidden_dim, hidden_dim // 2),
                nn.Dropout(cfg['dropout'])
            ])
            in_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x, compressed_dim):
        encoded = self.encoder(x)
        return encoded[:, :compressed_dim]
    

def initialize_weights(model):
    """Initialize network weights."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Kaiming/He initialization for LeakyReLU
            nn.init.kaiming_normal_(
                m.weight, 
                mode='fan_in',
                nonlinearity='leaky_relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)